#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional
# from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Any
from typing import List

import numpy as np
import torch
from typeguard import check_argument_types
from funasr.modules.beam_search.beam_search import BeamSearchPara as BeamSearch
from funasr.modules.beam_search.beam_search import Hypothesis
from funasr.modules.scorers.ctc import CTCPrefixScorer
from funasr.modules.scorers.length_bonus import LengthBonus
from funasr.tasks.asr import ASRTaskParaformer as ASRTask
from funasr.tasks.lm import LMTask
from funasr.text.build_tokenizer import build_tokenizer
from funasr.text.token_id_converter import TokenIDConverter
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.set_all_random_seed import set_all_random_seed


from funasr.utils import asr_utils, wav_utils, postprocess_utils
from funasr.models.frontend.wav_frontend import WavFrontend

from modelscope.utils.logger import get_logger
import soundfile
logger = get_logger()

header_colors = '\033[95m'
end_colors = '\033[0m'

global_asr_language: str = 'zh-cn'
global_sample_rate: Union[int, Dict[Any, int]] = {
    'audio_fs': 16000,
    'model_fs': 16000
}

class Speech2Text:
    """Speech2Text class

    Examples:
            >>> import soundfile
            >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
            >>> audio, rate = soundfile.read("speech.wav")
            >>> speech2text(audio)
            [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            asr_train_config: Union[Path, str] = None,
            asr_model_file: Union[Path, str] = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            device: str = "cpu",
            maxlenratio: float = 0.0,
            minlenratio: float = 0.0,
            dtype: str = "float32",
            beam_size: int = 20,
            ctc_weight: float = 0.5,
            lm_weight: float = 1.0,
            ngram_weight: float = 0.9,
            penalty: float = 0.0,
            nbest: int = 1,
            frontend_conf: dict = None,
            **kwargs,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        if asr_model.frontend is None and frontend_conf is not None:
            frontend = WavFrontend(**frontend_conf)
            asr_model.frontend = frontend
        # logging.info("asr_model: {}".format(asr_model))
        # logging.info("asr_train_args: {}".format(asr_train_args))
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        token_list = asr_model.token_list
        scorers.update(
            ctc=ctc,
            length_bonus=LengthBonus(len(token_list)),
        )

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        # transducer is not supported now
        beam_search_transducer = None

        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )
        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        )
        device='cuda'
        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        # logging.info(f"Beam_search: {beam_search}")
        # logging.info(f"Decoding device={device}, dtype={dtype}")

        # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)
        # logging.info(f"Text tokenizer: {tokenizer}")

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        has_lm = lm_weight == 0.0 or lm_file is None
        if ctc_weight == 0.0 and has_lm:
            beam_search = None
        self.beam_search = beam_search
        self.beam_search_transducer = beam_search_transducer
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None
    ):
        """Inference

        Args:
                speech: Input speech data
        Returns:
                text, token, token_int, hyp

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        # lengths: (1,)
        # if len(speech.size()) < 3:
        #     speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        #     speech_lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        lfr_factor = max(1, (speech.size()[-1]//80)-1)
        
        batch = {"speech": speech, "speech_lengths": speech_lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, enc_len = self.asr_model.encode(**batch)
        if isinstance(enc, tuple):
            enc = enc[0]
        # assert len(enc) == 1, len(enc)
        enc_len_batch_total = torch.sum(enc_len).item()

        predictor_outs = self.asr_model.calc_predictor(enc, enc_len)
        pre_acoustic_embeds, pre_token_length = predictor_outs[0], predictor_outs[1]
        pre_token_length = pre_token_length.round().long()
        decoder_outs = self.asr_model.cal_decoder_with_predictor(enc, enc_len, pre_acoustic_embeds, pre_token_length)
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]

        results = []
        b, n, d = decoder_out.size()
        for i in range(b):
            x = enc[i, :enc_len[i], :]
            am_scores = decoder_out[i, :pre_token_length[i], :]
            if self.beam_search is not None:
                nbest_hyps = self.beam_search(
                    x=x, am_scores=am_scores, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
                )
    
                nbest_hyps = nbest_hyps[: self.nbest]
            else:
                yseq = am_scores.argmax(dim=-1)
                score = am_scores.max(dim=-1)[0]
                score = torch.sum(score, dim=-1)
                # pad with mask tokens to ensure compatibility with sos/eos tokens
                yseq = torch.tensor(
                    [self.asr_model.sos] + yseq.tolist() + [self.asr_model.eos], device=yseq.device
                )
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)]
                
            for hyp in nbest_hyps:
                assert isinstance(hyp, (Hypothesis)), type(hyp)
    
                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()
    
                # remove blank symbol id, which is assumed to be 0
                token_int = list(filter(lambda x: x != 0, token_int))
    
                # Change integer-ids to tokens
                token = self.converter.ids2tokens(token_int)
    
                if self.tokenizer is not None:
                    text = self.tokenizer.tokens2text(token)
                else:
                    text = None
    
                results.append((text, token, token_int, hyp, enc_len_batch_total, lfr_factor))

        # assert check_return_type(results)
        return results


def inference(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        beam_size: int,
        ngpu: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        log_level: Union[int, str],
        data_path_and_name_and_type,
        asr_train_config: Optional[str],
        asr_model_file: Optional[str],
        audio_lists: Union[List[Any], bytes] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        token_type: Optional[str] = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        bpemodel: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        streaming: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        num_workers: int = 1,
        frontend_conf: dict = None,
        fs: Union[dict, int] = 16000,
        lang: Optional[str] = None,
        **kwargs,
):
    assert check_argument_types()
    print('305',data_path_and_name_and_type) #305 ['speech', 'sound', 'C:\\Users\\deep\\.cache\\modelscope\\hub\\damo\\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\\am.mvn']
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    print('315')
    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"
    hop_length: int = 160
    sr: int = 16000
    if isinstance(fs, int):
        sr = fs
    else:
        if 'model_fs' in fs and fs['model_fs'] is not None:
            sr = fs['model_fs']
    # data_path_and_name_and_type for modelscope: (data from audio_lists)
    # ['speech', 'sound', 'am.mvn']
    # data_path_and_name_and_type for funasr:
    # [('/mnt/data/jiangyu.xzy/exp/maas/mvn.1.scp', 'speech', 'kaldi_ark')]
    if isinstance(data_path_and_name_and_type[0], Tuple):
        features_type: str = data_path_and_name_and_type[0][1]
    elif isinstance(data_path_and_name_and_type[0], str):
        features_type: str = data_path_and_name_and_type[1]
    else:
        raise NotImplementedError("unknown features type:{0}".format(data_path_and_name_and_type))
    if features_type != 'sound':
        frontend_conf = None
        flag_modelscope = False
    else:
        flag_modelscope = True
    if frontend_conf is not None:
        if 'hop_length' in frontend_conf:
            hop_length = frontend_conf['hop_length']
    print('345')
    finish_count = 0
    file_count = 1
    if flag_modelscope and not isinstance(data_path_and_name_and_type[0], Tuple):
        data_path_and_name_and_type_new = [
            audio_lists, data_path_and_name_and_type[0], data_path_and_name_and_type[1]
        ]
        print('audio_lists',audio_lists)
        if isinstance(audio_lists, bytes):
            file_count = 1
        else:
            file_count = len(audio_lists)
        if len(data_path_and_name_and_type) >= 3 and frontend_conf is not None:
            mvn_file = data_path_and_name_and_type[2]
            mvn_data = wav_utils.extract_CMVN_featrures(mvn_file)
            frontend_conf['mvn_data'] = mvn_data
    # 1. Set random-seed
    set_all_random_seed(seed)
    print('362')
    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        frontend_conf=frontend_conf,
    )
    # print('speech2text_kwargs',speech2text_kwargs)
# {'asr_train_config': 'C:\\Users\\deep\\.cache\\modelscope\\hub\\damo\\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\\config.yaml', 'asr_model_file': 'C:\\Users\\deep\\.cache\\modelscope\\hub\\damo\\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\\model.pb', 'lm_train_config': 'C:\\Users\\deep\\.cache\\modelscope\\hub\\damo\\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\\config_lm.yaml', 'lm_file': 'C:\\Users\\deep\\.cache\\modelscope\\hub\\damo\\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\\lm.pb', 'token_type': None, 'bpemodel': None, 'device': 'cuda', 'maxlenratio': 0.0, 'minlenratio': 0.0, 'dtype': 'float32', 'beam_size': 1, 'ctc_weight': 0.0, 'lm_weight': 0.0, 'ngram_weight': 0.9, 'penalty': 0.0, 'nbest': 1, 'frontend_conf': {'fs': 16000, 'win_length': 400, 'hop_length': 160, 'window': 'hamming', 'n_mels': 80, 'lfr_m': 7, 'lfr_n': 6, 'mvn_data': array([[ -8.311879 ,  -8.600912 ,  -9.615928 , ..., -13.3996   ,
#         -12.7767   , -11.71208  ],
#        [  0.155775 ,   0.154484 ,   0.1527379, ...,   0.1495501,
#           0.1499738,   0.1509654]])}}

    # input('')
    speech2text = Speech2Text(**speech2text_kwargs)

    # 3. Build data-iterator

    print('394 flag_modelscope')
    input(data_path_and_name_and_type_new)
    loader = ASRTask.build_streaming_iterator_modelscope(
        data_path_and_name_and_type_new,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
        sample_rate=fs
    )

    print('speech2text.asr_train_args',speech2text.asr_train_args)
    input('')
    forward_time_total = 0.0
    length_total = 0.0
    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    asr_result_list = []
    for keys, batch in loader:
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        print('_bs',_bs)
        input('')
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        # batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}

        # logging.info("decoding, utt_id: {}".format(keys))
        # N-best list of (text, token, token_int, hyp_object)
        print('keys',keys)
        input('')
        print('batch',batch)
        input('')
        time_beg = time.time()
        # audio, rate = soundfile.read("asr_example_zh.wav")
        
        results = speech2text(**batch)
        # batch_ = {"speech": torch.tensor(np.array([audio],dtype=np.float32)), "speech_lengths": torch.tensor(np.array([len(audio)]))}
        # print('batch_',batch_ );input('')
        # results = speech2text(**batch_)
        print('results',results)
        print('used time',time.time()-time_beg)
        input('')
        time_end = time.time()
        forward_time = time_end - time_beg
        lfr_factor = results[0][-1]
        length = results[0][-2]
        forward_time_total += forward_time
        length_total += length
        logging.info(
            "decoding, feature length: {}, forward_time: {:.4f}, rtf: {:.4f}".
                format(length, forward_time, 100 * forward_time / (length*lfr_factor)))
        
        for batch_id in range(_bs):
            result = [results[batch_id][:-2]]
            print('result',result)
            input('')
    
            key = keys[batch_id]
            for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), result):
                if text is not None:
                    text_postprocessed = postprocess_utils.sentence_postprocess(token)
                    print('text is not None')
                    item = {'key': key, 'value': text_postprocessed}
                    print('item',item)
                    input('')
                    asr_result_list.append(item)
                    finish_count += 1
                    asr_utils.print_progress(finish_count / file_count)
    
                logging.info("decoding, utt: {}, predictions: {}".format(key, text))

    logging.info("decoding, feature length total: {}, forward_time total: {:.4f}, rtf avg: {:.4f}".
                 format(length_total, forward_time_total, 100 * forward_time_total / (length_total*lfr_factor)))
    return asr_result_list


# 394 flag_modelscope
# speech2text.asr_train_args Namespace(config='examples/aishell_demo/paraformer_s2/conf/train_asr_paraformer_sanm_50e_16d_2048_512_lfr6.yaml', print_config=False, log_level='INFO', dry_run=False, iterator_type='sequence', output_dir='/nfs/FunASR_results/paraformer/1m-1gpu/baseline_train_asr_paraformer_sanm_50e_16d_2048_512_lfr6_fbank_zh_char_local', ngpu=1, seed=0, num_workers=16, num_att_plot=3, dist_backend='nccl', dist_init_method='file:///nfs/FunASR_results/paraformer/1m-1gpu/baseline_train_asr_paraformer_sanm_50e_16d_2048_512_lfr6_fbank_zh_char_local/ddp_init', dist_world_size=1, dist_rank=0, local_rank=0, dist_master_addr=None, dist_master_port=None, dist_launcher=None, multiprocessing_distributed=True, unused_parameters=True, sharded_ddp=False, cudnn_enabled=True, cudnn_benchmark=False, cudnn_deterministic=True, collect_stats=False, write_collected_feats=False, max_epoch=20, patience=None, val_scheduler_criterion=['valid', 'acc'], early_stopping_criterion=['valid', 'loss', 'min'], best_model_criterion=[['valid', 'acc', 'max']], keep_nbest_models=5, nbest_averaging_interval=0, grad_clip=5, grad_clip_type=2.0, grad_noise=False, accum_grad=1, no_forward_run=False, resume=True, train_dtype='float32', use_amp=False, log_interval=50, use_matplotlib=True, use_tensorboard=True, use_wandb=False, wandb_project=None, wandb_id=None, wandb_entity=None, wandb_name=None, wandb_model_log_interval=-1, detect_anomaly=False, pretrain_path=None, init_param=['/nfs/init_model/paraformer_9k_bigmodel.pth'], ignore_init_mismatch=False, freeze_param=[], num_iters_per_epoch=None, batch_size=20, valid_batch_size=None, batch_bins=1000, valid_batch_bins=None, train_shape_file=['/nfs/dataset/data/asr_stats_fbank_zh_char/train/speech_shape', '/nfs/dataset/data/asr_stats_fbank_zh_char/train/text_shape.char'], valid_shape_file=['/nfs/dataset/data/asr_stats_fbank_zh_char/dev/speech_shape', '/nfs/dataset/data/asr_stats_fbank_zh_char/dev/text_shape.char'], batch_type='length', valid_batch_type=None, fold_length=[512, 150], sort_in_batch='descending', sort_batch='descending', multiple_iterator=False, chunk_length=500, chunk_shift_ratio=0.5, num_cache_chunks=1024, dataset_type='small', dataset_conf={'filter_conf': {'min_length': 10, 'max_length': 250, 'min_token_length': 1, 'max_token_length': 200}, 'shuffle': True, 'shuffle_conf': {'shuffle_size': 10240, 'sort_size': 500}, 'batch_conf': {'batch_type': 'token', 'batch_size': 6000}, 'num_workers': 16}, train_data_file='/nfs/dataset/data/dump/fbank/train/ark_txt.scp', valid_data_file='/nfs/dataset/data/dump/fbank/dev/ark_txt.scp', train_data_path_and_name_and_type=[['/nfs/dataset/data/dump/fbank/train/feats.scp', 'speech', 'kaldi_ark'], ['/nfs/dataset/data/dump/fbank/train/text', 'text', 'text']], valid_data_path_and_name_and_type=[['/nfs/dataset/data/dump/fbank/dev/feats.scp', 'speech', 'kaldi_ark'], ['/nfs/dataset/data/dump/fbank/dev/text', 'text', 'text']], allow_variable_data_keys=False, max_cache_size=0.0, max_cache_fd=32, valid_max_cache_size=None, optim='adam', optim_conf={'lr': 0.0005}, scheduler='warmuplr', scheduler_conf={'warmup_steps': 30000}, use_pai=False, num_worker_count=1, access_key_id=None, access_key_secret=None, endpoint=None, bucket_name=None, oss_bucket=None, token_list=['<blank>', '<s>', '</s>', 'and@@', '筑', '陨', '眺', '塘', '檩', '衷', '氧', '孔', '阖', '邠', '坎', '喵', '曰', '鼠', '隐', '腊', '族', '矧', '敉', '俜', '似', '怫', '塔', 'price', '春', '罍', '娅', '棉', '弃', '茱', '应', '汈', '擦', '贺', '鹇', 'these', '迅', '诬', 'do@@', '盍', '秕', '啃', '颟', '辑', '彘', 'ps', '斜', '瞭', '铟', '漭', '蹇', '旆', '窳', '臊', '览', '嘿', '淖', '尴', '袆', '斧', '筹', '媵', '挞', '臧', '齐', '璨', '笥', '滂', '即', '愔', '思', 'gr@@', '幅', '祛', '箬', '礁', '茅', '北', '澡', '俭', '蘅', 'ing', '肺', '肢', '巢', '九', '蠓', '路', '藻', '沱', 'ness', '璐', '积', '寞', '栳', '舆', '医', '眷', '岳', '勘', '璃', '黔', '犇', ' 哎', '罡', 'k', '丝', 'de', '跣', '梦', '需', '毅', '峡', '竞', '砦', '研', '眙', '滋', '鹳', '肝', '阼', 'per', '忱', '乏', '废', '邦', '輶', '驯', '夫', '寳', '忪', '崑', '睾', '逗', '峰', '越', '狗', '蟒', '笆', '适', '洇', '缶', 'ore', '辎', '粤', '蹴', '黄', '浞', 'comp@@', '犀', '藏', '本', '嗖', '黻', '這', '绰', '鉏', '麼', '喆', '袪', '刚', '侦', 'ic@@', '骧', '瓫', '柤', '桃', '鲴', '褙', '韵', '妓', '甍', 'to@@', '轱', '塑', '坯', '貌', 'n@@', '蕞', '疡', '伏', '酉', '暇', '霖', '了', '萸', '嘶', 'nu@@', '挼', 'by', '聩', '袓', '嶓', '桎', '抖', '攥', '鞬', '毳', '旗', '庸', '呋', '诲', 'tting', '狭', '魄', '伎', '喋', '樱', '翎', '怯', '场', '睽', '盒', 'times', '鲳', '爆', '绵', '皋', '尢', '嘎', '渝', '迂', '嘁', '袷', '始', '奚', '台', '禄', '挢', '座', '绀', '漱', '龃', 'tu@@', '榫', '诛', 'minu@@', '萼', '裛', '玠', '谵', '亳', '副', 'any@@', '诊', '唑', '頫', '斿', '赁', '骇', '训', '母', '床', '微', '椰', '迢', '埃', '辏', '汩', '叟', '辔', '隙', '遵', 'pp@@', '翕', '佬', '栖', '踉', '皕', '苏', '痂', '奋', '阄', '悌', '点', '碡', '茎', '睫', '闫', 'few', '籁', '孰', '拥', 'for', '曾', '疲', '辞', '赖', '姆', '与', '诰', '怨', '沙', 'with@@', '睱', '谓', '晰', '嗌', 'id', 'aga@@', '実', '魏', '鲟', '寅', '滗', '珲', '腑', '冠', '夺', '娶', '宇', '侩', '筘', 'an', '磺', '邛', '着', '踵', 'ite', '狝', 'ele@@', '蓼', '猿', '豆', '蔷', '沽', '去', '铥', '癀', '站', '甪', '璧', '范', '哓', '菏', '龠', '岷', '嫉', '拧', '札', '戒', '琏', '绪', '澌', '楠', '莱', 'cer@@', 'here', '别', '帻', '嗉', '假', '拽', '髭', '穰', '勋', '栓', '塬', 'ou@@', '橫', '刻', '侣', '鎉', 'bre@@', '趸', '稳', '岌', '拎', '落', '岙', '氨', '桴', '鬶', 'clu@@', '蚣', '肌', '讹', '骼', '忧', '雹', '0@@', '算', '腔', '璇', '酣', '锭', '蟾', '逦', '椟', '频', 'ts', '矫', '拨', '珅', '侨', '蚨', '皯', '翛', '儋', '恳', '瀍', ' 敌', '砬', '奁', '耵', '烂', '绿', '缆', '辣', 'time', '蔗', 'too', '圾', '骞', '慥', '啶', '帔', '楢', 'bi@@', '蚤', '浛', 'ine', '綦', 'old', '肩', '擎', 'ling', '瘆', '娲', 'prob@@', '暑', '鲨', '焚', '剽', '玚', '乔', '纪', 'bo@@', '熏', '毗', '鳙', '鞥', '絜', '糠', '菔', '廛', '谪', '冬', '遐', '衽', 'ich', '柿', '峭', '渴', '亓', '荠', '蝠', '扆', '鄞', '诎', '尙', '摸', '牙', '薤', 'da@@', 'w', '呜', '陂', '磻', '匹', 'while', '迁', '良', '郤', 'ct@@', '蚕', '浥', '鲋', '腱', 'pres@@', '氕', '颉', '夥', 'ter@@', '左', '侉', '妍', '嚒', '殖', '私', 'bas@@', '锱', '篪', '吻', '鄅', '鳗', '疳', 'cor@@', '毛', '歼', '邵', '圪', 'inn', '舛', '埗', '貉', '帐', '妮', 'ged', '窒', 'put', '诉', '堌', '气', '國', '摩', '沫', '谁', '转', '语', '琵', '羊', '檠', '慎', '踮', '啾', '瞻', '山', '播', '筌', '财', '飚', '苟', '扣', 'mis@@', '桌', '侑', 'jo@@', 'ke@@', '冶', '滔', '蠹', '呓', '捷', '证', '崞', '阵', '掐', '劳', '皆', '巧', '肏', '肆', '葆', '檄', '画', '狨', '谧', '傑', '诱', '纭', '荥', '厥', 'bri@@', '绁', '睇', '布', '江', '噎', '灞', '鄏', '煊', '蒹', '厮', '馓', '狴', '碍', '穷', 'lar@@', '食', '迩', 'meeting', 'tter', '趣', '辜', '椁', '汽', '燠', 'ate', '礴', '骈', 'will', 'not', '煽', '嗳', '秆', '勤', '陆', '键', '墎', '官', '蘧', '酤', '唐', '颓', '仝', 'iting', '田', '楫', '瑁', '鄢', 'our', '诘', 'venue', '霜', '镏', '痫', '娟', '婢', '埋', '汕', '铋', '徭', '隰', '猊', '卞', '慑', 'said', '裘', 'bus@@', '召', '粱', '返', '缨', '纻', '磋', '炴', '凊', '曝', '兀', '洼', '杓', '榆', 'hotels', '睹', '糯', '窘', '葩', '帮', '荷', '塌', '矍', '圯', 'er@@', ' 蚪', '篼', '咳', '吸', '喃', '岩', '嚟', '谱', '崆', '蟭', 'wal@@', '姗', '谛', '东', '菱', 'ction', '肫', '祥', '述', '璋', '仙', '唇', '硕', '嘉', '醒', '兔', '恒', '银', '収', 'cre@@', '弭', '曌', '螯', '苡', '疾', '鸱', '权', '搔', '途', '茂', '皑', '补', '肃', 'ns', 'market', '讠', '阈', '机', '苒', '髙', 'is@@', '幢', '郐', '萎', '帛', '烖', '襟', '崟', '溶', '螺', '铲', '仉', '舌', '敖', '倍', '锏', '鸦', '夌', '埌', '腰', '雠', 'ol@@', '涙', '翳', '夹', '疏', 'el', '驭', '鹯', ' 瑜', '薛', '奇', '娴', '靳', '镥', '伍', '暨', 'what', '萧', '皴', '烦', '敷', '样', 'ri@@', '睆', '伦', '後', '祺', '贾', '汹', '箸', '掂', '械', 'ob@@', '赜', '黎', '潵', '鲽', '牒', '罐', 'inter@@', '冰', '成', '薳', 'if', '堡', '搜', '漪', ' 赕', '症', '参', '鲫', '克', '浴', '薹', '梫', '调', '劢', '茹', '飐', 'r', 'her', '鲊', '妪', '惟', '榜', '谒', '梃', 'twenty', 'ni@@', '猥', '兴', '氓', '肘', '饕', '陵', 'within', '轲', 'cted', '単', '嚗', '唼', 'inte@@', '钋', '抒', '郓', '牖', '蔻', '邂', '懦', '邻', '茸', '辈', '种', '黹', 'hard', '悦', '综', 'ved', 'oo@@', 'used', '溱', '藁', '士', '猝', '芹', 'char@@', 'we', '呒', 'ss@@', '岐', '桖', '迟', '荞', '哏', '仵', '抟', '淇', 'il', '电', '褀', '果', 'bir@@', 'ges', '暻', '椊', '吴', 'sal@@', '獐', '煲', '箥', '爰', '呣', '邶', '尻', '水', 'sh@@', 'fore', 'mat@@', '浅', '枘', '稷', '艨', '唧', 'can@@', 'eng', 'know', '桠', '亵', 'tou@@', '伥', '虏', '綮', 'sha@@', '腌', '蔓', '莶', '戬', '鋆', '酥', '璜', '阐', '贶', '远', '验', 'dre@@', '赎', '惭', '裴', '晌', '铡', '愆', '隧', '祐', '貔', '片', 'al', '抾', '柰', '探', '喈', '蟮', '潎', '批', '胴', '郁', '蓿', '稿', '傲', '垒', '甙', '靼', 'chic@@', '效', '栗', '齁', '仲', '担', 'har@@', '仟', '砂', 'i', '舰', '吝', '慆', '芸', 'why', 'tic', '哥', '佩', '铀', '妊', 'x', '刍', '殄', '躬', '莙', '客', '拚', '葬', '哼', '婵', 'ance', '棺', '没', '崎', '曛', '宛', '斌', '掠', '醇', '庾', '黥', '乗', '旉', '铰', '寨', '钨', '沥', 'gh@@', '吇', '嬖', '厨', '箔', 'ah', '譞', '鲖', '蒗', '銶', '角', '脏', '夬', 'who', '毂', '赓', '嘢', '蹶', '駃', 'avenue', '呃', "that's", '訾', '蒜', '就', '抿', '霫', 'que@@', '牁', '叻', '绌', '捣', '埭', '蛩', '迤', 'ir', '最', '不', '鹚', '钧', '晃', '钍', '岍', '烯', '授', '笨', '馔', '甸', '啰', '赌', '蠼', '荆', '濉', '摹', '剔', '浪', '瓦', '涤', '阬', 'eng@@', '墨', '鲢', '老', '拮', '轿', '弈', '秸', 'ken', '省', '穹', '跨', '芤', '剰', '湍', '吥', '喧', '借', '伯', '咋', '噢', '剩', '略', '图', '毕', '爻', '箭', 'ans@@', 'no', '缣', 'fic@@', '必', '礻', '视', '侔', '乸', '缎', '比', '殉', '禅', '蹈', '茶', '沔', '腹', '更', '倢', '骑', '俦', '一', '巉', '糌', 'there', '笾', '泺', '虫', '随', '室', '谙', '淞', 'even', '嘌', '掉', '进', '栈', '隋', '钳', '饲', '裾', '搞', '朽', '嚏', '垱', '倘', 'sy@@', '蒂', '訚', '火', '葱', '踹', 'only', 'den@@', '胰', '曦', '汨', '奴', '院', '晶', '臇', '赭', '蚵', '便', '藜', '鍪', '穆', '尿', 'find', '偾', '项', '嬅', '济', 'area', ' 皿', '蹽', 'af@@', '曈', 'ger', '袭', '温', '包', '惎', '枝', '槁', '跑', '汇', '嫦', '崒', '颇', '丐', '丛', '哠', '鲲', '佯', '疱', '來', '彝', '件', '鸫', '张', '缋', '檎', '港', '尸', 'comm@@', '瘘', '囍', '锅', '惫', '衔', '蔚', '龚', '酱', 'ina', '尚', '孪', '蔵', '帧', '弯', '迄', '訇', '恕', '紡', '吱', '觐', '印', 'need', '叭', '茫', '汶', '邢', '磅', '焜', '蜣', '米', '俎', 'ath', '蛔', '组', '壹', '诈', 'ing@@', '希', '茨', '砧', 'has', '蝶', '矛', '拖', '乍', '浇', 'another', '输', '朗', '殡', '壶', '灿', '礌', '钡', '瓤', '序', '误', '毖', '静', '鸾', '墚', '璟', '咱', '惘', '化', '腾', '苍', ' 苼', '七', '芾', '囝', '淄', '馆', '榉', '荸', '摧', '醋', '缦', '帘', '蛋', '曙', '萩', '莉', '犸', '拜', '特', '蕊', '并', '冼', '埝', '茴', '佶', '噶', 'ked', 'port', '柠', '吶', '竿', '鞧', '糙', '栻', '褂', '杉', '陛', 'shi@@', '朋', '升', '钛', '拭', 'walk', '钱', '岸', '衲', '若', '燕', '墩', '戛', 'ations', '诳', '冨', '强', '掌', '腺', '淤', '鼍', '妥', '亥', '俵', '鹩', '占', '佤', '棋', 'does', 'tes', '拒', '劼', '绩', 'ren@@', '货', 'g@@', '深', '钯', '棬', '墟', '疼', '骊', '摅', '祧', '兊', '坠', 'int', 'use', '泞', '赦', '甾', '葺', '辘', '炆', '旭', '鸯', '茆', '融', '艄', '晖', '钺', '勉', '嘘', '龛', '蕙', '渀', '钞', '写', '弋', '颦', '灌', '埚', '鲷', '亡', '矩', '轰', 'a', '单', '觚', '呯', '祏', 'rec@@', ' 逢', '憧', '蒽', '內', '乡', '鸠', '卜', '庄', '仰', 'how', '铓', '踝', '隆', '避', '豌', 'low@@', 'ak', '劣', '哺', '头', 'proble@@', 'es', '说', '哇', '折', '祝', '偻', '揆', '的', '盎', '初', '骝', '荻', '饷', '耽', '莸', 'just', '簰', '现', 'pl@@', '籍', '珉', '蕲', '臌', '闪', '崮', 'gra@@', '琯', '圆', '瓴', '赬', '镧', '被', '共', '芯', '蚧', 'stu@@', 'mee@@', '沧', '伲', '觌', '筏', '庑', 'still', '题', 'wat@@', '4', '绱', '入', '亚', 'sho@@', '珫', '饴', '點', 'than', 'good', 'l@@', '梁', '忿', '荐', '躺', '蹡', '呕', '圩', '唷', '陌', 'ue', '鲭', '碗', '怪', '飘', 'country', '粑', '怹', '飕', '烨', '吹', '嵇', '驺', '纰', 'in@@', '间', '馈', '榑', '窜', '泗', '硪', '躏', 'th', '耸', '贞', 'wom@@', '排', '箩', '绽', '舵', '焉', '振', '镶', 'thirty', '闲', '摁', '堰', '牵', '栋', '堤', '馀', '盟', 't', '旄', '凇', '洣', '録', '韭', 'por@@', '孑', '茄', '闺', '淀', '坡', '烟', '洺', 'gre@@', '敦', '哉', '到', 'ding', '遑', '钒', '壳', 'lo', '纾', '砲', '灶', 'lee', '玘', 'up', '梵', '旖', '佗', '竽', '绋', '砩', '酒', '苯', '焕', '祚', '苁', '嗓', 'ail@@', '殽', 'om@@', '棨', '翼', '墼', '萄', '垭', '碱', 'cts', '渲', '矱', '掇', 'best', '锃', '谶', '喜', '雌', '辊', '啀', '嗞', '谢', '疹', '玎', '唤', '兆', '彳', '溧', '丕', '棒', '桁', '樓', '跟', '蝼', '哭', '啭', '替', '乩', '箪', '城', '朾', 'ear@@', '鲌', 'ship', '吕', '粉', '舜', '伛', '觏', '燮', '铊', '硝', '撤', '瘝', 'thanks', '锵', '圣', 'contin@@', '侬', '浮', '棵', '歭', 'ici@@', '珞', '褔', '券', '演', '箫', '缵', '篾', '鲮', '砒', '含', '郡', '快', '栏', '瘟', '饤', 'tw@@', '拃', '盹', '壕', '桯', ' 嗪', '鞨', '甏', '锫', '涕', '冕', '鄣', '淌', '辰', '唿', '暲', '蚀', '跋', '郸', '镀', 'ku@@', '赔', '姺', '课', '础', ' 耷', '涪', 'day', '笳', 'away', '稞', '鹈', '珍', '毯', '酮', '汀', '梆', '嫫', '准', 'ces', '巷', '晋', '肉', '莆', '痢', '缗', '怜', '鄙', '搠', 'fri@@', '仳', '该', '宓', '珂', '圉', '弨', '悬', 'buil@@', '绸', '太', '外', '祢', '蓍', '圹', ' 侓', '跸', '谊', '获', '髈', '迮', '鹤', '卦', '嗻', '佐', '愠', '媲', '殍', '齉', '妹', '残', '嗄', '钾', 'court', '踞', '脯', '菖', '琼', '傻', '三', '虿', '唪', '逶', '鲤', '镌', '肇', '弘', '李', '履', '恩', '蒌', '夙', '环', '坒', 'gar@@', 'ans', '嘣', '嵯', '命', '酢', '屏', '鈇', '麟', '旨', '旼', '疮', 'with', '解', '屈', '趴', '蠛', '密', '瞩', '屎', '显', '魁', '衯', '钇', '酩', '鳌', '戆', '芈', '十', 'gu@@', '陪', '黑', '缌', 'ch@@', '摇', '梨', '胼', '撷', '疤', '砟', 'el@@', '唛', '芪', '速', 'ol', '细', '馥', '犰', 'bal@@', '鲇', '韂', '焰', '胗', '粹', '枌', '嵬', '古', 'she', 'through', '筛', '翀', '协', 'se', '魃', '格', 'mes', '晥', '跱', '掺', '阕', '智', '松', 'st@@', '靠', '斟', '粒', '舞', '瀣', '棅', '茭', '韫', '鐎', '灵', '龢', '卷', 'lion', '曹', '哒', '皝', '哲', 'pe@@', '患', '逸', '涠', '蛰', '佣', '猇', '狈', 'nine', '囫', '风', '态', '慈', '慜', '俨', '汲', '肛', '隶', '坩', '赍', '海', '癫', 'my', '委', 'ill', '胤', '覩', '臬', '矶', '炷', '衬', '前', '馊', '伽', '艳', '妗', '肠', '檗', 'soon', '氙', '琅', '谏', 'light', '変', 'seven@@', '旸', '芭', 'en@@', '烧', '诃', '攘', '陧', '觅', '铑', '氐', '余', 'night', 'hou@@', '鹀', '膜', '炙', '抨', '珊', 'ses', '漩', 'both', '桉', '笺', '鎛', 'led', '披', '膛', '蜻', '菽', '娼', '団', '揽', '测', 'f@@', '芎', '吅', 'sion', '遹', '瓘', '慕', '他', '鄩', '矽', 'thou@@', '沒', '唁', '匿', '设', '嵖', '髹', 'ine@@', '恸', '窣', '-@@', '街', '膝', '碑', 'national', 'it@@', '瘢', 'ci@@', '侮', 'l', '陉', '照', '原', '厐', '悚', '答', '犷', '罔', '绘', '敞', 'ys', '捆', '殚', '填', '挟', 'tal@@', '萌', '卑', '甃', '吉', '蜮', '帑', '笖', 'new', '昺', '诤', '襜', '矗', '藠', '苓', 'th@@', '哝', 'its', '蚋', 'ran@@', '澳', 'eight', '贱', '傕', '亦', '续', '槐', '筚', '追', '醺', '錾', '蹒', '玟', 'look', '圃', '颗', '旎', '圮', '绷', 'op@@', '咙', '槃', '冫', '乳', '鸣', '柴', '蚴', '擞', '锴', '姣', '惯', '管', '奎', 'ra@@', '瞠', '侍', '恵', '岬', '喎', '摭', ' 卺', 'wee@@', '羧', 'cep@@', 'fron@@', '妁', '很', '禹', '巯', '夼', '鄫', 're@@', '动', '迓', '狐', '瑕', '棹', '屹', '皈', '陔', '殛', '仿', '蝥', '缘', '镫', '品', 'ase', 'row', '缜', 'stance', '予', 'custom@@', '抬', '鞶', '蛘', '埏', '漂', '凝', '虻', '姒', '痒', '邝', 'ss', '战', '悴', 'spe@@', '羝', '吮', '锗', '湘', '端', '淸', '孢', '3', '郴', '卬', 'fif@@', '濯', '射', '簏', '锌', '啖', '懑', '霪', '棻', '簺', '怅', 'g', '毎', '犳', 'ffe@@', '镉', '閦', '吲', '驹', 'are@@', '埇', '心', '漴', '娃', '侯', '蔽', '值', '鲧', 'fr@@', 'ful', '嘹', '滥', '騠', 'ility', '喹', '悉', '嗽', '些', '硁', 'mer@@', '磊', '霆', '麝', '曲', '蜃', 'police', '镩', '笪', '苾', '靑', '凼', '多', '质', '缇', '嗫', '沏', 'ened', '花', '诹', '尉', '珥', '崩', 'ld', 'x@@', '揎', '纷', '缂', '轹', '庙', '渚', '鸪', '乒', '惧', 'peop@@', '歌', '唾', '樘', '膺', 'fro@@', '哂', '腼', '霄', '坞', '霰', '掎', '娿', '镬', '巨', '碇', '藩', '活', '荤', '团', '缪', '钵', '飞', '儡', '苤', '貊', '柄', '蓠', '防', '贮', '碾', '狞', '艏', '喏', '稚', '映', "i'm", '谤', '蜿', '车', '乂', '寕', '啧', '虔', 'ster@@', ' 垣', '嗛', '讪', 'ves', 'again', '隗', '帜', '嗾', '绂', '公', '卮', '抱', '仕', '以', '栘', '拊', '萤', 'him', '荪', '淬', '7', '鋹', '敢', '颖', 'ment', '嫩', '棕', 'show', '跩', 'out@@', '汤', '迕', '榨', '暗', '糍', '晡', '9', '稂', '曼', '蒺', 'ture', '鬄', '逅', '岚', '芟', '昶', '埤', '幺', '猖', '伙', 'pub@@', '南', '荨', '趁', '淑', '嘲', '悔', '藉', '争', '渔', 'pool', '簟', '谀', '噘', '窀', '祟', '阜', '涸', '掖', '癃', '疑', '搢', '漏', '锉', '钹', '耱', '踢', '骎', '稣', ' 锲', '繇', '缊', '劈', '啻', '蕴', '仔', '昝', '且', '滚', '柢', '镊', '响', '凰', '噗', '瑴', '嗔', '简', '蜇', '有', '豢', 'ap@@', '啓', '翅', '愤', 'peri@@', '蚶', '弄', '禨', '蚡', '坝', '换', '纨', '蹑', 'for@@', '草', '荛', '懈', '奉', '鳊', '疗', '搂', '串', '幸', '岽', '牍', '蝰', '絶', '秣', '缴', 'at', '网', '嗑', '岗', '绊', '圳', '恁', '反', '方', '癞', '煞', '雪', '尤', '鐧', '麒', '黡', '殷', '都', '则', '剃', '揄', '毐', '噱', 'fi', '氹', '泠', '樾', '迳', '嫚', '齿', '殳', '墒', '役', '晟', '咔', '芃', '睁', '柽', '戍', '屺', '虱', '韦', '涅', '姚', '鋈', 'sure', '既', '涯', '甯', '嘤', '硞', 'som@@', '惴', '狻', '堑', '屉', '愿', 'li@@', '行', '谲', '嶂', '峣', '碜', '暂', 'h@@', '鏖', '瘊', '蜈', '浈', '萦', ' 职', '蚊', '汴', 'people', '妱', '鸰', '易', '芜', '挪', '影', '竹', '洸', '烀', '鹘', '胜', '兵', '咧', '楷', '币', '妖', 'ant', '臣', '桩', '创', '囹', 'na@@', '鞑', '楂', '逡', '惆', '卿', '闱', '耀', '那', '童', '钰', '玮', '郄', '昏', '乘', '钩', '晳', '笼', '核', '芙', '小', '忋', '区', 'as@@', '颢', 'our@@', 'that', '稻', '销', '韶', '刑', '延', 'k@@', 'teen', '幄', 'pic@@', '叱', '骷', '棰', '羁', '垝', '犴', '媱', '兄', '尓', '乞', '鲦', '划', '壬', '芡', 'hotel', '佃', '氯', ' 您', '颌', '汝', '缫', '幂', '竣', '喾', '疥', 'long', '广', '镂', '酫', 'ings', 'ood', '柊', '唣', '辽', '稀', '襞', '讼', '篱', '坻', '袂', '华', '自', '歧', '昂', '摺', 'gh', '聿', '犟', '敛', '牺', '旳', '锥', '玛', '低', '鄮', '漳', '叠', ' 川', '呼', 'where', '戳', '嗮', '琦', '厓', '窠', 'cas@@', '舷', '甦', '凛', '谖', '旷', '沌', '狒', '溉', '绍', '劲', '滟', 'in', '褊', 'fam@@', '楽', '金', '磕', 'see', '斩', '佛', '壅', '境', '诂', 'around', '羑', '浆', '矜', '铈', 'provi@@', '藐', '伉', '阶', '哀', '潼', '精', '像', '凶', '琇', '秧', '涂', '豫', '镒', '蒟', '叹', '颜', '莫', '阀', '痕', '爬', '嬲', '滓', '牮', '沐', '璈', '窸', '湮', '喊', '徘', '而', '仞', '蛆', '吵', '栟', '郯', '谄', '膑', '垯', '恰', '筠', '淝', '剌', 'vie@@', '估', 'first', '渊', '鶗', '缬', '踺', '呦', '宄', '颎', '蔼', '挒', '亹', '墉', '倧', '梪', '猱', '顼', '泫', '鸳', '赠', '聋', '鬲', '隽', '胚', '驱', '丶', '邪', '鲚', '韩', '婆', 'sed', 'it', '审', '屠', '众', '翩', '铺', '磨', '醲', '瘼', '佑', '霹', '臀', '坮', '俯', '舸', '辍', '谗', '甥', '祭', 'tell', '商', 'ace', '宾', '骡', '浍', '冉', '肾', 'im@@', 'win@@', '甬', '蹚', '粕', '脖', '遽', 'next', 'expe@@', '榕', '蹂', '邹', 'stru@@', '沁', '宸', '旮', '锁', '侂', '拢', '辫', '仁', 'be', '洱', '摘', '律', '预', '徕', '鬣', '挠', '戟', '嘴', '杖', '骍', '劵', '哮', '雁', '擀', '鴐', ' 衎', '芮', '据', '霭', 'com@@', '俗', '伝', 'ory', '轭', '博', '谐', '孺', 'te@@', '锹', '瞥', '导', '糜', '堙', '乾', '搌', '鏐', '你', 'con@@', '琍', 'art', '徇', '塞', '讽', '瞄', 'rence', '溢', '卉', '逞', '阮', '阊', '婊', 'mil@@', '专', '姜', '浉', '府', 'sing@@', '嗵', '哨', '砺', '吋', '闹', '败', '居', '娓', 'ce', '囟', '楼', '元', '鲥', '嗙', 'tely', '帷', '還', '懋', '欷', 'ong', '郝', '丨', 'breakfast', '崴', '橼', '停', '沾', 'under@@', 'tion@@', '非', '堵', '仆', '铗', '难', '蛑', '狙', '找', '熠', 'over@@', '檀', '鸩', '檐', '彀', '蟋', '腚', '槟', '泄', '舅', '痼', '秤', '氆', '罄', '啼', '啡', '冽', '疎', '嵨', '吿', '航', '采', 'mb@@', '裟', '檿', '辆', '眍', '溃', 'can', '唻', '媖', '佺', '狰', '仪', 'rent', '沓', '话', '霾', '婷', '雨', 'eigh@@', '白', '瞧', '澎', '洞', '阔', 'ta@@', '侧', '躇', '莘', '骏', '宰', '縠', 'birth', '萃', 'men@@', '秉', '轮', '刹', 'fl@@', '鸮', '忾', '胖', '攫', '磁', '飧', '鲈', '邙', '阌', '皂', '危', '搹', '靺', '唔', '撴', '柝', '垮', '膈', '辋', '榷', '邘', '锂', '戚', '蔹', '粝', '翊', '攵', '悯', '涝', '媞', '俤', '镲', '梳', '蓥', ' 艾', 'guest', '顿', '譬', '兹', '囚', '倌', '遣', '朔', 'such', '篷', '囷', '宫', '戊', '嵎', '娉', '箅', '檫', '玷', 'please', '彰', '蜓', '怃', '癯', '怛', '镇', '还', '诿', '庞', '开', '节', '卒', '逵', '颔', '杰', '蘸', '楚', '颍', '吐', '堃', '澜', '弧', '流', '堍', '严', '焱', 'on', '纬', '巽', '确', '子', '紙', '沭', '戞', '屙', '胭', '劫', '珧', '信', '樵', '讴', '豺', '叽', '钎', '霁', '瀛', '糟', '噌', '豝', '湜', '洎', '菌', '悆', 'ree', '凯', '徜', '郏', 'today', '勾', '嬉', '螵', '戕', '璞', '忝', '俞', '言', '庵', '贼', '费', 'kind', '扁', '骁', '咪', '凿', '讳', '掊', 'ated', '苄', '鳝', '噍', '茧', 'govern@@', '筼', '颋', '愛', '渭', '踟', '罪', '汔', '踩', '陽', '疽', '闵', '我', '蒡', '缠', '曺', '婪', '农', ' 露', '染', 'sent', '氽', 'et', '咷', '圧', '咀', 'site', 'sti@@', '梗', 'water', '舔', '嚣', '蜉', '逖', '湄', '栅', '刳', '薢', 'ally', '诸', '藕', '钔', '伋', '莜', '硬', '窟', 'sa@@', '愚', '蟪', '秩', '雯', '褚', '鹎', '泃', 'ner', 'ast', '菜', '晦', '枨', '偲', '嚩', '遴', 'su@@', '掸', '千', '馄', '功', '胺', 'rep@@', '涡', 'ther', '孩', '液', '狲', '业', '巡', '脍', '甚', '珜', '郜', '蔑', '疔', '庚', '硌', '裉', '骘', 'sequ@@', '迎', '盖', '噪', '尺', '咒', '蜕', '店', '镐', '蝉', '宝', '卍', '弩', '学', '猁', '犊', '妄', '葭', 'every@@', '螋', '馃', 'ating', '壮', '熟', 'rela@@', '嗬', '约', '锞', '呫', '护', '磒', '疙', '羞', '绦', '铳', '掕', '宗', '荀', '玢', 'ser@@', '啦', '氪', '盯', '疸', '鬐', '绚', '锡', '鬻', '瓮', '麸', '旱', '娱', '敕', '跄', '烘', '蠕', 'te', '诽', '重', '翠', '珑', '慰', '鲍', '勣', '袱', '瑙', 'tly', '庆', 'government', '荦', '阗', '烫', '倓', '俏', '鸹', '倦', 'ound', 'co@@', '竟', '腋', '昙', '濂', '啋', '揶', '泣', '郾', '垍', '轳', '某', '酎', '板', '晤', '廑', '奶', '醴', '镑', '讣', '缤', '龅', '畿', '脁', 'ma@@', '醢', '嗟', '丗', '殿', '魅', '熨', 'wr@@', '嚷', '彤', '栎', 'americ@@', '谡', '泽', '柬', '髎', '盆', '诅', '瘁', '萘', '喁', '媒', '忸', '阍', '曡', ' 裰', '锦', 'something', '犋', '爱', '煺', '揉', '苇', '嘈', '胪', '铁', '屁', '颂', '锩', '骅', '渌', '邱', '脬', '滏', '罹', '散', '鼹', '父', '摔', '边', '申', '苛', '敝', '冈', '蕈', '郕', '耨', '闳', '逄', '拔', '将', 'fe', '鲂', '颙', '0', 'room', '胙', '澍', '媚', '廪', '量', '贩', '镕', 'nine@@', '恚', '鹾', '旃', '铵', '堺', '剑', 'ket', '支', '墙', '洒', '俳', 'ors', '诐', '黟', '珩', '跻', '浸', '孟', 'mp@@', '狷', '踌', '渎', '逝', '颁', '务', '羹', '羖', '阽', '跪', '褒', '乜', '择', '盩', '鸭', '抗', '递', '褪', '怂', 'cou@@', '蟠', 'cen@@', '傜', '砍', '鲔', '蹙', 'restaurant', '杆', '茵', '尊', '耻', '淮', '躜', '蜡', '嘱', '谩', '蝗', '堞', '姨', 'happ@@', '铿', '楝', 'park', '力', '殊', '畹', 'say', '练', '纱', '溪', '虮', '篑', '蜱', '惜', '跌', '啁', '溜', '饼', '裥', '勇', '柱', '惨', '陟', '殪', '安', '徼', '纛', '痍', '谨', '取', '犒', '鄄', '粟', '眨', '坤', '妤', '剟', '雉', '傩', '嫄', '嘧', '咽', '愀', '簸', '赏', '箾', '龈', '放', '菅', '坚', '奠', '黢', '琴', '潞', '朝', 'tually', '铞', '乇', '醪', '潍', '槊', '纯', '瀹', '诋', '慢', '奸', '嘭', '揠', '昃', '革', '司', 'ce@@', '趵', '醮', '碘', '器', '澶', '知', '&', '厢', '啷', '晔', '炜', 'ook', '斐', '盏', '妻', '娣', '燋', '窈', '法', 'also', '菸', '炽', '选', '埕', '击', '滤', '铖', '觎', '莩', 'le@@', '匏', '踅', '酸', '鹑', 'enjoy', '闰', '毡', ' 祆', '身', '郊', '笄', '乖', '甩', 'like', '否', '厖', '爹', '牦', '起', '僔', '钓', '浚', '忙', 'get', '冯', '樊', '识', '蹉', '鑫', '畚', '滕', '掮', '尾', 'war@@', '桶', '瓢', '毫', '膊', '髌', '钗', '桼', '碲', '辨', '唉', '竲', '痰', '膻', '锖', '嘡', '雩', '版', '昧', '敬', '蕨', '伴', '徍', '襀', '盘', '憷', '涑', 'different', 'after', '虽', '何', '煎', '宽', 'ori@@', '攀', '冢', '零', '樯', '哌', '瓣', '馍', '唰', '炪', '旰', '厔', '葵', '痞', '己', '靓', '凄', '服', '烤', '仮', '恨', '喳', '贯', '郞', '饪', '铄', '滠', '蠢', '薅', '齑', '褓', '黼', '涿', "n't", '磬', '匾', '沂', '镳', 'nice', '浯', '悝', '淫', '捅', '箨', '瞀', '勠', '屐', '蹁', '蹦', '槿', '类', '栉', '脘', '页', '桕', '脊', '欲', '蝽', '勃', '坷', '酶', '售', '縻', '欺', '膏', '词', '兢', '楸', '娡', '娩', '陡', 'ple', '尧', '幷', '豉', '桫', '滁', '麋', '罘', '朕', '耗', '汉', '登', '较', '逾', '蔫', '赳', '秭', '咫', '斑', '跚', '舒', '莞', '闾', '氤', '骸', '槩', '瓶', '餐', '瞪', '沉', ' 朱', 'had', '鲣', '嵪', 'nothing', '嫡', '恍', '衢', '轴', '杈', '赂', '津', 'red', '拤', '狠', '卣', '蚱', '疆', '捞', '婉', '固', '梶', '垸', '逋', '髡', '晓', '骶', '季', '炀', '喷', '垛', '蜂', 'sts', '阝', '未', '熬', '绲', '坟', '苈', 'are', '寮', '吧', '皖', '捌', '炤', 'b', 'the', '哈', '鞍', 'ir@@', '掘', 'tation', '噙', '酊', '忄', '硼', '耕', '偎', '雎', '磴', '锺', 'over', '侃', '婚', '吗', '竺', 'man', '也', '苣', '绛', '冤', '呈', '孱', '缭', '埼', '犬', '麦', '蓐', '技', '夜', 'tri@@', '杭', '佝', '莨', 'di@@', '毙', '贿', '猎', '桡', 'so', '氖', '叼', '哚', '濠', '湾', '全', '读', '盐', '钚', '鬘', '万', '鲠', '貂', '鋐', '堼', '茺', '拟', '牡', '蝮', '镢', '嚯', '束', '喱', '彿', 'col@@', '仃', '涌', '蚯', '妩', '箢', '隹', '亰', '疃', '嬗', '喇', '攒', 'am', '听', 'fe@@', 'zero', '痛', '诗', '干', '疫', '嬴', '降', '签', '丈', 'sm@@', '髑', '劬', '萏', '诀', '镵', '坭', '咤', '池', '榼', '岕', '崃', 'ined', 'chu@@', '糸', '祸', '猜', '婿', '搋', '咛', '箴', '辚', '悄', '荒', '挑', '托', 'drive', '撇', '莒', '鄜', '冒', '稹', '户', 'ould', '铤', '翃', 'try', '圊', '艿', '桐', '兒', '揿', '堋', '档', 'ink', 'email', '犍', '铍', '招', '鳇', '敲', '雳', '奌', '裝', '戏', '哗', '栱', '哆', 'ds', ' 浙', '岂', '挝', '莲', '腩', '杬', '促', '斥', '蛱', '诓', '炖', '璘', '怵', '礽', '咬', '珐', '韡', '邺', '祀', '皇', '渑', '困', '潜', '添', 'ter', 'che@@', '如', '萹', '熙', '扇', '亭', '亍', '瘗', '舍', '皌', '诌', '虢', '欣', '掰', '棘', '岞', '毽', '卵', '罴', '疋', 'ali@@', '整', 'i@@', '紫', 'mu@@', '涘', '携', '奏', 'cri@@', '馘', '翘', '抚', '筝', '玄', '霎', '铮', '澹', '嫰', '夏', '咿', '围', '従', '猾', 'pre@@', '逊', '糖', '漫', '聍', '及', '瘵', '绢', '棱', '笈', '铯', '釭', '恼', '癎', '苢', '斁', '醣', '植', '鱼', '涓', 'z', '目', 'par@@', '撬', '戎', '偓', '已', '打', 'busine@@', '桦', '庇', '坪', 'problem', '舾', 'centr@@', 'fifty', '梧', '𫖯', 'restaurants', 'beau@@', 'fac@@', '谈', '腙', '阉', '孛', '秘', ' 庶', '畴', '例', 'ang', '幛', '下', '溅', '彗', '魟', '诏', '鸢', '邽', '瘪', 'ay', '胱', '勒', '槲', '橹', '男', 'ide@@', '治', '锒', '祯', '慌', '佥', '苔', '暴', '时', '撩', '俶', '屋', '抠', '嫘', '浔', '鲻', '暌', '窿', '炒', '溇', '髯', '簿', '嶖', '峒', 'si@@', '觥', '午', '联', 'ty', '猫', '腭', '晻', '袝', '戾', '波', '啐', '戈', '蒎', '麓', '汾', '茉', '埔', '蛳', '徳', '鸶', '艇', '顶', '髻', '徒', '豨', '碳', '欻', '薰', 'hi', '呲', '舳', '劾', '形', "'", '渥', '羔', '枇', ' 模', '弁', '坏', '徊', '馗', '刨', '璀', '锤', '钤', '囔', '拦', '剐', '揸', '滈', '缚', 'ach@@', '斯', '诒', '寂', '裒', '同', 'should', '蒴', '窑', '示', '戗', '录', '恶', '培', '迦', '邮', '熜', '泊', '乌', '篌', 'center', '癸', '昔', '牠', ' 濛', '噻', '苋', 'ang@@', 'give', '姥', '祖', '搬', '悠', '瓒', '嫱', 'wit@@', '畏', '神', '湟', '扃', '桀', '醭', '谘', ' 虾', '玳', '斡', '寥', '攻', '忑', 'man@@', '僻', 'money', '飊', '国', '吠', '扦', 'j@@', '幹', '泛', '摈', '匼', '廉', '通', '袍', '楹', '搦', '缮', 'ft', '岱', 'scho@@', '丹', '蜗', '蓓', '卟', '倥', '喽', '蛾', '殭', '绣', 'chan@@', '震', '棍', '潋', '葫', '嚎', 'ed', '漉', '阻', '俐', '德', '様', '酝', '倒', '橱', 'send', '谯', '嗐', '署', '贴', '搭', '坦', '蹯', '緛', '队', '死', '閟', '岭', '倪', '诜', '偱', '醐', '痊', '社', '凡', '畯', '摒', '迭', '措', '挌', '媛', 'same', '蚓', '擤', '澈', '眈', '剪', '嫣', '鹱', '娆', '凸', '厅', '臼', '枭', '炎', '烊', '掀', '洄', '僳', 'velo@@', '觉', '逼', 'have', '皎', '酯', '怦', '叶', '统', 'thir@@', '从', '纡', 'les', '术', '径', '征', 'come', '躅', '此', '甜', '践', 'br@@', '鎸', '讲', '玲', '衣', '麾', '枥', '拈', '腮', 'reas@@', '芩', '鹖', 'more', '儆', '愕', '淏', '躄', '玕', '届', '永', '哟', '虓', 'sit@@', '宴', '夷', '梭', '紧', '瘰', 'recei@@', '匐', '号', 'from', '况', '畋', '谜', '莽', '锣', 'pas@@', '驽', ' 昱', '喟', 'v@@', '呀', "'re", '阒', '踬', '认', '订', '黝', '筻', 'seven', '们', '舱', '揭', '妾', '礼', '高', '传', '香', '谬', '篆', '逮', '玑', '眄', '驿', 'all@@', '嗅', '五', '淹', '裨', '咕', '焖', '氚', 'ari@@', '崤', '接', '沈', '寰', ' 轻', '旒', '维', 'test', '麴', '枱', '挚', '句', '驾', '筷', '坂', '须', '鼩', '炳', '推', '姿', '溴', '庠', '箻', '燹', ' 拂', '呱', '愫', '袤', '睑', '眩', '冷', '葸', 'but', '睢', '掣', '唠', '陈', '喉', '晬', '溯', '艺', '苦', '腐', '蟹', '燃', '候', '伸', '萝', '汭', '6@@', '僭', '蘖', '瘫', '书', '控', '乪', '溘', '痦', 'lo@@', '蝣', '啂', '缩', 'part@@', '只', '施', '鸨', '鎏', '悕', '诵', '孬', '獾', '玩', '汪', 'ary', '吁', '拄', '诟', '扰', '鼎', '珪', '橇', '隍', 'close', '姑', 'ty@@', '迹', '骆', '崐', 'go@@', '勺', '倜', 'ble', '盥', '剂', 'qu@@', '圻', '荧', '荣', '疵', '酪', '稗', '幔', '井', '蛭', '泔', '定', 'q', '楱', '刈', '使', '潟', '螅', '铒', '巇', '呖', '垂', '啉', '戮', '佉', '寓', '翚', '啪', 'fa@@', 'w@@', '绐', '抛', '謷', '忐', '趿', '位', '螨', 'last', '迫', '爷', '碎', '廊', '啮', '柯', '酰', '烜', '煳', '粲', '求', ' 楞', '考', '挲', '触', '荽', '荫', '疴', '遨', '仫', '瑭', '扩', '恭', '藦', '羣', '味', '缀', '享', '枣', '宠', '浊', 'back', '扪', '篥', 'sou@@', '涉', 'son', '翔', '讫', '理', '玙', '栩', '闩', '龄', '亢', '桄', 'sil', '幡', '婀', '脚', '疠', '歔', '兰', '査', '镗', '涎', '霏', '待', '忻', '警', '科', '脐', '琚', '真', 'own', '醚', '绡', 'ices', '嶷', '胶', 'u@@', '峦', '箜', '丰', '痨', '衡', '锋', '宦', '皦', '感', '矮', '爨', '亟', '装', '鹁', '房', '捡', '乐', '胫', '彧', 'kil@@', '葛', '氏', '捋', '戢', '牢', '5', 'dis@@', '园', '孓', '柃', '榄', '喺', '悛', '俢', 'c', '缢', '富', 'je@@', '竭', 'way', '拙', '牧', '阋', '究', '钘', '濮', '皙', '问', '告', '鞅', '燚', '肴', '螭', '篁', '当', '洮', '渠', '码', '辗', '慨', '崇', '诫', '意', '姤', '飓', '噬', '铐', '寡', '咨', '伺', '殓', '容', '蚁', '柏', '枧', '瑗', '挛', '绔', '月', '绾', '隘', '僖', '纇', '握', '耜', '舐', 'tom@@', '旅', '翮', 'products', '缑', '肄', '闿', '硫', '砚', '蜚', 'bu@@', '鍉', '晗', ' 阃', '衾', '蝴', '丽', '髅', '煜', '习', '窃', '芑', '櫈', 'ru@@', 'deta@@', '谌', '俄', '桨', '芦', '泵', '企', '粽', '揣', '领', 'qui@@', '鄚', '肱', '尼', '滦', '椽', '癜', '甄', 'local', 'fol@@', '偿', '丸', '涞', '铜', 'du@@', '噜', '倬', ' 珣', '圜', '墅', 'car@@', '撂', '栀', '评', '螽', '蘑', '扬', '挨', '轵', '殇', '酺', 'high', 'ach', '鲛', '啥', 'think', "don't", '表', 'cho@@', '盈', '啊', '怔', '滉', '艮', '徽', '撸', '淯', '茈', '硚', '桢', '魍', '潸', 'at@@', '仄', '鹭', ' 沅', '操', '炭', '砝', '之', '讷', '吔', '疰', '葚', '夭', '跶', '蘼', '鲩', '滴', '菘', '滨', '加', '倾', '体', '茕', '髀', '雅', 'ile', '欠', '崧', '囤', '僦', '守', '辂', '棂', 'se@@', '捶', '扛', '铠', '铩', '京', '熘', '洛', '娄', 'the@@', '痿', '箱', '驮', '礅', '郃', '谣', 'ack@@', '晴', 'res', 'lot', '纲', '诖', 'ways', '谋', '煌', '绗', '炼', '卤', '屄', '界', '拶', 'ar', '遠', '周', '驩', '肖', '跖', '莳', '级', '咣', 'hu@@', '啕', '歀', '忤', '临', 'many', '募', 'ned', 'their', '睿', '冥', '久', '巍', '堇', '洽', '郎', '董', 'this', '毒', '醉', 'ton', '诶', '旺', '蛉', '规', '琶', '騑', '捉', 'imp@@', '尥', '拗', 'e@@', '歆', '刀', '跆', 'three', '锢', '侏', '拳', '扤', '饨', '剥', '潡', '俊', '嗤', '穑', '地', '情', '憔', '惹', '奢', 'inclu@@', 'mo@@', '铭', '至', '浐', '祎', 'really', '帆', 'dri@@', '楮', '邬', '弛', '篮', '兮', 'up@@', '慧', '腆', '碴', '挂', '计', '豚', '滑', 'indi@@', 'into', '褭', '酞', 't@@', '符', '彼', '涔', 'body', 'ins', '漶', '峪', 'down', '缓', '剀', '福', '叮', 'wa@@', '失', '隅', '怄', '扳', 'great', '泰', '娌', '孙', '彬', '毁', '蛲', '萱', '泻', '舀', 'dge', '瞅', 'vo@@', '咆', '祜', '枋', '憩', '文', 'gro@@', '倮', '鹄', 'all', '槎', '忆', '絯', '裢', 'two', '兑', '明', '遥', '窍', '吃', "it's", '启', '妽', '郅', 'ies', 'centre', '吒', '亿', '雲', 'ask', '叩', '媜', '蚰', '奄', '垡', ' 椿', '篡', '聘', '窖', '垩', 'diffe@@', '购', '褥', '豸', '腠', '咦', '眦', '敏', '鼐', '昉', '顽', '琊', '砌', '僮', '乱', '瞒', '撑', '羲', '纳', '赞', '焐', '橛', '嵩', '陀', '楔', '牛', '鹊', '帽', 'may', '偬', '丫', '兜', '砰', '濡', '獬', '好', '腿', '靶', '木', '唸', '纹', '裹', 'cl@@', 'hund@@', '优', '猲', '焯', '岘', '辱', '丞', '查', '秽', '胀', '鳖', '岈', '喒', '默', '见', 'forty', '纥', '泚', '驳', '铃', '萋', '筱', '蛏', '琲', '鸵', '这', '链', 'read@@', '垵', '踦', '奭', '別', '软', '盲', '黛', '纤', 'chil@@', '扱', '狡', '贽', '贻', '鳏', '篓', 'der@@', '滢', '嵌', '妺', '臭', '谥', '夯', ' 韬', '惺', '检', '峨', 'ms', '胛', '恫', '构', '鹂', '刺', '讦', '唬', '梈', '捯', '由', '志', '达', '梢', '歉', 'al@@', ' 淳', '般', '恐', '獗', '央', '喔', '蟀', '伞', 'sh', "'s", '姮', '痈', 'ck', '俾', '髫', '蜒', '愧', '呆', 'mer', '桤', '鬏', '型', '四', '哖', '龋', '鳔', '骗', '砷', '泷', '猷', '茯', '茁', 'cha@@', 'small', '倨', '乎', '奕', '鹅', '痉', '奥', '辙', 'sting', '冲', '赅', '泅', '羰', '悲', '焘', '瞵', '紊', '卖', 'take', '鳐', '菊', '姸', '辐', '嘻', 'fast', '囿', ' 凫', '纵', '咝', '箧', 'well', '钬', 'he@@', '辕', '鹜', 'unk@@', '诔', '泐', '世', '镖', '骄', 've', '寸', 'pay', '鴂', ' 睐', '氲', '牟', '洏', '尔', 'quo@@', '椭', '芨', '险', '耳', '柁', '牾', '蚂', 'much', 'most', 'now', 'singapore', '丙', '洋', '燊', '啵', '胬', '媾', '碟', '濆', '陇', '扥', '势', '慷', '笋', '泥', '鄘', 'hote@@', '罕', '沤', '葡', '蠡', '产', '鲵', '凉', '萆', '籽', '胞', '哦', '侠', '晚', '咵', '杯', '邗', '甑', '胁', '历', '嘏', '喘', 'is', '霉', '仇', '鳀', '鉴', '猛', '纂', '决', '阏', '饯', '宣', '象', '儿', '猗', '瀑', '荚', '满', '茳', '酷', '鼬', '旁', '屡', '榇', 's@@', '呷', '驻', '薮', '鹏', '馋', '鹪', '纠', 'thank', '轶', '渺', '镁', 'cost', '汐', '谅', '甫', 'publi@@', '储', '抹', '杨', '裳', '歙', '鏊', '犼', '蹐', '愦', '断', 'able', 'ind', '狂', '雒', '妧', '巴', '嘟', '筵', '喤', '舫', '刃', 'fi@@', 'company', '郿', '瞢', '褰', '清', '巿', '苕', '内', '暾', '垠', '戌', '溟', '剋', '葶', '猡', '蔺', '岖', '邕', 'any', 'pri@@', ' 卸', '基', '馒', '出', '柞', '溍', '吨', '蜥', '炟', '徂', 'inve@@', '桂', 'frien@@', '铪', '庖', '廋', '誉', '嗡', '忘', "i'@@", '螃', '胯', 'un@@', '铨', '酹', '鷃', 'wi', '眚', '跤', '汁', '针', '钏', '雀', '限', '面', '蛐', '黃', '脩', '叛', '枷', '故', 'sk@@', '编', '菥', '勐', '汗', '胃', '洗', '洹', 'ers', '髦', '宏', '聊', '嵫', '囎', '淠', '垧', '欧', 'oms', '揵', 'before', '畈', '饬', '秦', '莓', '努', '炔', '匠', '际', '蕉', '蝇', 'po@@', '匽', 'year', '橥', 'pping', '注', '承', '琨', '又', '玉', '彖', '缸', '怒', '遏', '坍', 'sta@@', '痹', '埆', '叵', '氟', '脂', '昭', 'ments', '宕', '盅', '惩', '馁', '绕', '遢', '扫', '蔡', '崾', '阎', '葳', '殂', '潢', '每', 'ro@@', '靖', '主', 'tional', '闸', '涖', '第', 'n', '镅', '蜜', '能', '衄', '猹', '鹰', '俘', '豪', '人', '痱', '蓑', '骒', '所', '绨', '粦', '望', '减', '魔', '释', 'ars', '逑', 'clo@@', '帏', '遘', '许', '誓', '牤', '刷', '扒', '峻', '9@@', '昕', '轩', '嬷', 'them', '窗', '蹄', '骰', '湓', '堨', '巂', '婧', '骖', '螬', '酋', 'ag@@', '闷', '滞', '郧', '朴', 'supp@@', '倡', '腴', '褴', '极', '漷', '佾', '煨', '沚', '肓', 'experi@@', '亮', '堕', '对', '秋', 'every', '铅', '蛴', '擅', 'a@@', '鳃', '保', 'ze', '弓', '趄', '翦', '涛', '盗', '道', '遮', '屿', '绥', 'cont@@', '部', 'den', '付', '诮', '粘', '镔', '溏', '啬', '贫', '轼', '猩', '薯', '胳', '抻', 'than@@', 'king', '燎', '笫', '鸂', '蓊', '峋', 'ent', '搡', '磡', '昌', '妈', '玺', '叫', '済', '彩', 'lim', '孀', '囧', 'we@@', ' 绒', '驸', '架', '斛', '资', '碣', '濞', '冿', '吡', '作', 'because', '晏', '餍', '瞾', '栊', '蹿', '荩', '尖', '依', '氍', '炝', '抡', '中', '潏', '潦', 'mon@@', '域', '杩', '兼', '捎', '怩', '正', 'f', 'near', '刮', '搪', '鸿', '俺', '耪', '牯', '瞎', '鬼', '惠', 'bro@@', '沛', '浃', 'pe', '釜', 'vi@@', 'jal@@', '拴', '冇', '暎', '魉', '痘', '盔', '枚', '秾', '杲', '郛', '僚', '昞', '足', 'sen@@', 'family', '岿', '榖', '蛊', '绯', '村', '鹃', '钼', '拓', 'dly', '苫', '锄', '跎', '碰', '钝', '淆', '鳟', '浓', '柇', '桷', '愬', '裔', '邃', '恂', '楙', '沨', 'ban@@', 'ssi@@', '鳞', '褛', '钌', 'y', '搀', '骺', '坊', '娘', '鄯', '袈', '匡', '厦', '艘', '障', '舂', '蹼', '材', '胡', '痄', '凖', '憋', 'll', '睥', '窦', '砕', '雇', '個', '籀', '饥', '价', '笞', '鞴', '筊', '树', '揖', '派', '垄', '檬', '眠', '偌', '闯', '姘', '少', '破', '怿', '扼', '恽', '岔', '亲', '垕', '踱', '泳', '截', '轫', '烽', '够', '武', '差', '声', '可', '绫', '巅', '邯', '椎', '硭', '畀', '常', '沸', '榧', '忡', '妫', '皮', '讯', '铣', '轾', '衙', '蒙', '蕃', 'things', '腓', '坐', '鞘', '榻', '霈', '垆', 'ous', '粳', '枹', '薨', '胆', '龙', '绺', 'ning', 'med', '挹', '诨', '译', '但', '半', '辩', '芊', '变', '翻', '尹', '靸', '酬', '咶', '暧', '姝', '貘', '墓', '匙', 'been', '墀', '芗', 'sc@@', '妲', '狮', '岫', 'ants', '谔', '鞭', '婶', 'one', '踏', '讵', 'land', '幼', '軎', '晕', 'wi@@', '瑞', '供', '啯', '朵', '枓', '奘', '咩', '訢', '臆', '硗', '帖', 'dent', '戽', '瘿', '段', '傅', 'oun@@', '枞', '属', '野', '芫', '诣', '钐', '堀', 'uring', '试', '衍', '馏', '龇', '讧', 'stre@@', '妒', '驴', '著', '槭', '洧', 'side', '氾', '孵', '狎', '歩', '囯', '愎', '过', 'pu@@', 'right', '赋', '阳', 'mber', '谕', 'town', '盾', '救', '饱', 'loc@@', 'spa', '橡', '剅', '蒲', '姐', '厕', '铚', '套', '弗', '栴', '鹨', '糨', '騕', '雷', '裁', '拌', '鼻', 'some', '搐', '均', '蓣', 'ice', '卢', 'ad@@', '瘌', '湶', '氩', '谰', '缙', 'ard', '栾', '迨', '浣', '秏', '萨', 'always', '污', '踔', '杏', 'ded', '髽', '秀', '湉', '粪', '展', '槻', '垓', '赫', '惮', 'lar', '薷', '迈', 'p', '塾', 'ga@@', 'ey', 'ho@@', '恺', '晩', '泸', '茚', '嗰', '绳', 'lit@@', '渍', '遇', '洑', '晞', '溽', 'ying', '枸', '无', '逍', '珰', '纣', '罩', 'ons', '嶃', '瑚', '裙', '搛', '6', '咾', '唱', '瑠', '睃', '於', 'din@@', '伟', '骓', '函', '瘤', '僆', '鞣', '兕', '伄', '柚', '席', '峄', '鄱', '抔', 'an@@', '物', 'loca@@', '穸', '懊', '2', 'amer@@', '稼', '叉', 'bra@@', '拇', 'str@@', 'res@@', '沆', '饰', 'pen@@', '厘', '傈', 'ality', '廷', '式', '夔', '督', '雍', '晨', '剿', '赶', '疖', '砜', '归', 'min@@', '徙', '阚', 'tions', '女', '营', '邅', '做', '伊', 'er', '筜', '蛎', 'business', '顾', '瘸', '洪', '钕', '觯', '政', '劝', '撮', 'break@@', '氛', '漾', 'sor@@', '址', '帼', '增', 'ation', '瞋', '稊', '慊', '肯', '躁', '糗', '美', '豕', 'us', ' 炸', '饽', '挎', '挫', '枢', 'dr@@', '铉', '奖', '槌', '绮', '擂', '虬', '钽', '胍', '凌', '辟', '虹', '铂', '杂', '砗', ' 赪', '潇', '合', '鰕', '吭', '迷', '滘', '弹', 'ff', '痤', '蛄', '觱', '杳', '嘬', '偕', '赘', '烩', '孖', '钉', '龉', '钪', '睬', '撵', 'big', '笠', '俅', '锯', '呛', '欹', 'cle@@', '珈', '浑', '墕', 'hel@@', '彪', '姊', '拆', '雄', 'hi@@', '钴', '壑', '幇', '卧', '痧', '澴', '长', '船', '六', '继', '棷', '尞', '窨', '燥', '膦', '镝', '裸', '办', 'resta@@', '廆', ' 宅', '蓉', '娑', '猪', '阁', '蟆', '恢', '讥', '蜾', 'of@@', '怏', '笕', '瞽', '赤', '岁', '禺', '昀', '她', '钫', '慝', ' 滩', 'ack', '跃', '贰', '巳', '寿', '铙', 'pro@@', '朊', 'mor@@', '箦', '丧', 'ready', '止', '跺', '缉', '桅', '论', '摽', '状', '驷', '咂', '铢', '绎', '叨', '遁', '锨', '里', '笊', '缰', '嵊', '膘', '帕', '泱', '鳜', '嫒', '仅', '捕', '徵', 'each', 'vil@@', '媪', '袁', '裕', '佻', '赙', '末', '笔', '啲', '轺', '掞', '彭', '氢', 'got', '缁', '蠊', '攉', '寻', '骋', '噤', 'ar@@', '棣', '疣', '豳', '祠', '衩', '匀', '王', '觫', '庭', '亶', '篇', '铎', '咖', '禛', '艹', '哐', 'hope', '系', '娇', '窄', '髓', '阱', '懿', '右', '离', '穴', '渣', '邨', '肸', '杞', '肚', '垗', '幕', 'ct', '苊', '隼', 'wh@@', '宋', '煤', '柸', '镋', '耘', 'kes', '激', '痣', 'work', '吽', '彟', '株', '筐', '贸', '尕', 'ere', '潬', '矣', '縢', '瑄', '油', '満', '漆', 'rooms', '霞', '咏', 'e', '珮', '仡', '眶', '醛', '票', '鼓', '突', 'ia', 'hundred', '瓯', '尬', 'no@@', '西', '潴', '楪', '傍', '蜘', 'yes', '蛀', 'sig@@', 'gs', '筮', '因', '谳', 'bur@@', '跹', '鲑', '惑', '肋', 'dy', '螟', '霓', '眼', '郇', '宙', '畲', '刬', '翟', '诧', '跫', 'or', '颅', '跷', '淅', '臑', '楦', '屯', '走', '羸', 'you', '咘', 'als', ' 斤', '畔', '晧', '忠', '肥', '实', '凋', '褶', 'ba@@', '摆', '熊', '篝', '乃', '绑', '沟', '瞿', '哜', '舶', '甘', '事', ' 瑛', '踧', '黯', '鲞', 'cal@@', '璩', 'enty', '剧', '洿', '孳', '祓', '天', '岑', '拾', 'nineteen', 'cause', '愇', '翁', ' 瞟', '昆', '弢', '摛', '义', '咸', '勮', '伪', '瑱', '橄', '丘', '铧', '怖', '终', '膨', '5@@', '扮', 'to', '猕', '琬', '群', '叁', '狄', '眵', '烹', '蚝', '儒', '援', 'ber@@', '1', '掾', '裂', '藓', 'r@@', 'squ@@', '鲁', '潺', '晷', '消', '鸸', '觊', '诡', '却', 'as', '弼', '脰', '铷', '册', '懵', '啜', '涣', '扈', '垚', '森', '赵', '垫', '镪', '怼', '峁', '惚', '缄', 'ass', '诼', 'tive', 'never', '昇', 'ence', '耍', '异', '朐', '枕', '襻', '契', '祈', '骜', '羯', '豁', '瘕', '贲', '禥', '哔', '祁', '螫', '锆', '孕', 'st', '忏', '掏', '昵', '哪', '鹞', '疍', '遄', '蠲', '恹', '狩', '憾', '悫', '弦', '脾', 'soci@@', '爸', 'home', '奔', '懂', '爿', '帙', '襦', '除', 'ters', 'ted', '景', '昚', 'eas@@', 'ven@@', '蹩', 'ac@@', '歇', 'fas@@', '抄', '牌', '浦', '榭', 'lan@@', '巺', '刁', 'secon@@', '顒', '个', '鲒', '嗨', 'out', '凤', '褫', '瑮', '珏', ' 惦', '斝', '监', '土', 'did', 'date', '陋', '眉', '湴', '谎', '撖', '礞', '抑', '旻', '倩', '蟑', 'produ@@', '忍', 'free', '瓜', '康', '晾', '偏', '麇', '缱', '唏', '杪', '戥', '坳', '皲', '拘', '簋', '怆', '负', '陬', '埽', '皞', '素', '锐', '苘', '菀', '嚜', '免', '嫌', '靡', 'd', '尪', '颧', '苷', 'de@@', 'inc@@', '策', '浠', '窾', '么', '壁', 'pi@@', '荟', '愣', '砫', '巻', 'ke', '旌', '咉', '矢', '婺', '绝', '线', '奂', '袋', '喻', '贡', '鍒', '逛', 'days', '疟', '筇', '髃', '锛', '俱', '缅', '锊', '骟', 'belie@@', '跏', '岜', '睚', '蜊', '铝', '闭', '麈', '趺', '锜', '彻', '缟', '甭', '苞', '钥', 'ph@@', 'ef@@', '灼', '銮', '塍', '淙', '鸷', '撼', '浒', 'though', '或', 'av@@', '荇', '要', 'road', 'wn', '纺', '师', '裇', ' 颚', 'end', '汜', '驰', '8', '椹', '牂', 'ties', '侪', '樉', '扎', '吞', '蕤', '垦', '茜', '泌', 'wor@@', '艋', '次', '珙', '建', '切', '盱', '簌', 'bus', 'ten@@', 'check', '硅', '概', '袒', '厚', '馐', '橐', '涵', '抢', '兖', '颠', '扑', '渐', '底', '獒', '瘴', '喝', '期', '踪', '镭', '糕', '翡', '养', '苺', 'gi@@', '顷', '蚜', '琛', '县', '槽', '滹', '钙', 'ple@@', '叙', '囵', '煮', '杵', '铛', '丢', '任', '咹', '莎', '请', '喂', '旬', '枫', '颊', '圄', '霍', '耆', '阅', '嵴', '鱿', ' 倭', 'me', '咎', 'sing', '眸', '羽', '二', '橦', '菉', '巩', '柑', '赧', '暮', 'd@@', '损', 'world', '隻', '鑹', '捧', '轨', '按', '啄', '聒', '炅', '湣', '詹', '饧', '柒', '恃', '掬', '爽', '料', '窎', '歹', '刿', '完', '餮', '鵀', '是', '榈', '透', '粗', '俛', '音', '份', '置', '砊', '蓬', '蘘', '缛', 'going', '遯', 'your', 'ever@@', '制', '淼', '冏', '妆', '钣', '怡', 'vern@@', 'z@@', '弇', '砥', '蜴', '饸', '冮', '衫', '韘', '寖', '虐', '玹', '3@@', 'tho@@', '闼', '旋', '煿', '嶝', '涟', '龌', 'ko@@', 'ge@@', '遛', '郢', '褡', '薪', '邓', '嫁', '叔', '嫤', '漠', 'gue@@', '藿', '臾', '屣', 'ily', '早', ' 痴', '埴', '囊', '俩', 'ours', '蓦', '渗', '锟', '朦', '附', '赴', 'make', '灾', '汛', '埸', '吾', '栲', '翰', '磔', '汰', '椋', '锸', '搴', '蒐', '韧', '撒', '纽', '笙', 'order', '坼', '伷', 'val@@', '烬', '梡', '瑶', '陷', 'go', '绤', '琉', '湖', '硷', '熹', '觇', '侥', '往', '牲', '租', '矾', '祉', '跂', 'age', '刘', '缃', '复', '阂', '刊', 'am@@', '苻', '步', 'cess', '蓖', 'he', '喙', '畜', '鄠', '让', '雱', '噫', 'sel@@', '钜', '胥', '蹬', 'help', '捻', '芽', '肷', '掳', '旧', '謦', '甲', 'ort', '搽', '哧', '骢', '吼', '口', '倚', '肽', '憨', '竦', 'product', '棚', '黍', '砾', '愈', '昊', '阑', '魂', ' 持', '献', 'pa@@', '凳', '厩', '佚', '鲱', '粮', '鹋', '姬', '抉', '籇', 'part', 'found', '嗍', '提', '蛇', '籓', '盼', '裱', '暄', '妞', '喑', '惶', 'wha@@', '窥', 'ks', '橘', '绞', '菁', '饭', 'ste@@', '猞', '芘', '搧', 'tain', 'street', '呔', '懆', '袼', '肼', 'on@@', '亨', '谑', '椅', '莹', '校', '躔', '颈', '圬', '匕', 'mail', '税', '年', '脆', '郦', '拿', 'fo@@', '鑱', '改', 'la', '轸', '掩', '芒', "can't", '彺', '觑', '邴', '埵', '莴', '伢', '腕', '阴', 'tre@@', '眇', '责', '青', '钠', 'point', '僵', '庹', '具', '迠', '穗', '逃', '源', '咻', '今', '绻', '发', '嗝', '逆', '侈', '抓', '羌', '款', '耧', '锰', '罢', '跗', '汞', '赚', '累', '掷', '啫', '膳', '捭', '禀', '崂', '栝', '掴', '讶', 'y@@', '瑾', '账', '挈', '鹬', ' 儁', '鳎', '骛', '釉', '矬', '潮', 'ort@@', '病', '日', 'o', '憍', '镍', '蜛', '壤', '葑', '聚', '灭', '立', '怙', 'm@@', '诙', '躲', '泡', 'would', '缺', '横', '违', '蛛', 'ls', '鹮', '鲐', '茌', '罚', 'ques@@', '畦', '厍', '缳', '劭', '拐', '萍', '债', '偈', '镛', '踽', '趋', '扭', '椒', '痔', '史', '称', 'thing', 'q@@', 'fre@@', '剁', 'beach', '舢', '黠', '执', ' 饵', '欤', '骥', '蓄', '邈', '疚', "'@@", '吓', '撄', '苌', '蜢', 'mar@@', '桓', 'made', '煦', '襁', '垃', '姓', '送', '柜', '埪', '怠', '蚍', 'little', '殴', '炕', '蛙', '伱', '禁', '睺', '捍', '颀', '猬', '蟊', '岢', 'o@@', '纩', '妨', '阡', ' 率', '蒋', '寐', '孚', '想', '洁', '沮', '铦', '镟', '潲', '杮', 'tal', '镯', '虞', '鶒', '椐', '簦', '锈', '顗', '仑', '额', '掼', '耿', '荑', '撙', '谍', '沃', '惊', '碉', '皓', '峯', '炮', '厶', '拼', '得', '帅', '蜎', '莠', '近', '诩', '喓', 'sequence', '愉', 'ure', '枯', 'thous@@', '槅', '移', '菡', '菪', 'ten', '癍', '珀', '湎', '瓿', '名', '妇', '再', '傺', ' 蚬', '荡', '孽', '匜', '堠', '荜', '绉', '刽', 'des', '1@@', '留', '裎', '赐', 'ving', '光', 'ly', '睟', '遂', '琮', '俟', '岛', '蛹', '蒿', 'nor@@', '钦', '麻', '纫', 'coun@@', '杷', '宿', '吣', '蝎', '湃', '惬', 'enjo@@', '卯', '牸', '诠', '谷', '摞', '螈', '然', '盂', 'mple', '萁', '灰', '畊', 'when', '烈', '俸', '衿', '龟', '谭', '唳', '楀', '痪', '曳', '羿', '罂', '脱', '廿', 'sur@@', '拝', '洵', '湿', '鸽', '冂', '跛', 'school', '尡', '挡', '疭', '议', 'self', '拍', '溺', '垅', '瞳', '笑', 'thousand', '乓', '噼', 'lie@@', '沿', '俣', '灏', '赈', '砭', '造', '标', 'loo@@', '罅', '鳣', '翥', '铬', '诞', '酌', '佟', '歘', '瑀', '眬', '璁', '拣', 'u', '颤', '笏', 'ur@@', '蝌', '蛟', 'number', '馕', 'place', '伐', '热', 'ver@@', '淜', '嚼', '鼢', '謩', '趟', '恬', '搅', '伫', '杌', '琐', '崽', '苗', '其', '熄', '在', '泾', '毓', 'very', '琥', '孤', '扞', '恪', "t's", '埙', 'do', '攮', '昴', '靥', 'feel', '酗', '蕫', 'der', '焙', 'gen@@', '辛', '觞', '平', '椤', '耦', '黜', '寇', '醍', '耶', '撰', '蔌', '曜', '谂', '烛', '旦', 'kno@@', '爪', '絺', '欢', '圈', '臂', 'ch', '谆', '荏', '巫', '擐', '嵋', '霸', 'ames', 'wel@@', '叆', '傧', 'tra@@', '隳', '唆', '澧', '焦', '窭', '剜', '菇', '锇', '锑', '枳', '娈', ' 侗', '橙', '愊', '泯', '茼', '荙', '遭', '蟥', '捱', '挤', '蹊', '暹', 'ity', '珠', '驼', '嗷', '贪', '扔', '硒', '沣', '鸡', '蕖', '仓', 'looking', '冦', '邳', '赝', '嵛', 'chi@@', '典', '桧', 'about', '茏', 'es@@', '缒', '蕻', '闽', '独', '骨', '尰', '鲼', 'und@@', '详', '姩', '脉', '腥', '拯', '婕', '杜', '俑', '经', 'they', '距', '僧', '辅', '骂', '睛', '夤', 'door', '捊', '靴', '庅', '劓', '舣', '饿', '幻', 'ry', '悱', '畎', '先', '畠', 'ye@@', 'those', '菲', '榔', '邑', 'city', '泖', '夕', 'better', '吏', '酡', '蕾', '犯', '磲', '锝', '鹌', '鹫', 'you@@', '耐', '梯', '鸬', '榛', '笛', '睡', '歪', '遍', '飒', '笃', '蓝', '于', '欸', '为', '琎', '宁', 'ki@@', 'tle', '直', '漕', 'years', '烙', '捏', '馅', 're', '诺', 'tur@@', 'deci@@', '励', '佘', 'bl@@', 'of', '噩', '阪', '梿', '孃', '杻', '覃', '瞆', '瑷', '观', '循', '泪', 'six@@', '集', 'cust@@', '瘠', '尽', '鱀', '虎', '来', 'other', '衮', '芷', '卓', '擒', '锶', '玻', 'lu@@', '缝', '瀬', '缯', '嫖', '偭', '运', '厣', '顺', '璝', '锬', '悼', 'off', '苑', '恤', '囡', '锻', '舁', '粼', '丑', '字', '酽', '鲜', '豹', '厝', '琤', '谞', '簖', '癣', '蚌', '退', '戡', '云', '颞', 'cur@@', '腻', '呗', '沼', '骤', '崭', '扯', '怍', 'app@@', '存', '害', '色', '踣', '淘', '嗦', '獠', '怗', '砸', '镜', '毹', 'table', '@', '鲀', '寤', '桥', '嘞', '髁', '哙', '织', '嗒', '麂', 'let', '偁', '苎', '燧', 'rou@@', '哃', '鄂', '翌', '酚', '咄', '嗥', '蒸', '虚', '囗', '条', '鹕', '挖', 'per@@', '矿', '珦', '囨', '冻', '佰', '洲', '漓', '菹', '镱', '尝', 'low', '苜', '剞', 'could', '埒', '铼', '结', '侵', '星', 'then', '埠', '峤', '晒', '睦', '搓', '時', '铫', '偃', 'might', '氦', '呻', '淦', 'ge', '缧', '脑', '膴', '英', 'pla@@', '案', '生', '顔', '记', '艰', 'tic@@', '砖', '药', '宪', '涩', '鞠', '潆', '郭', '宵', '砀', '蟛', '岀', '赃', '蛤', '粢', '驶', '膂', '垤', 'nee@@', '逐', '芳', '螂', '超', 'change', 'j', '锔', '箓', '笤', '弱', '畸', '分', '攸', '蛸', '烷', '脒', '昨', '咴', '羚', '芍', '卲', '噔', '飖', '蜍', 'some@@', '崦', '嬛', '仍', '柳', '鹗', '増', '乙', '嫂', '骚', '巾', '憬', '刭', '顸', '磙', '伃', '孜', '碌', '堐', '祗', '鳉', '汊', 'ab@@', '袄', '诚', '剡', '锷', '锕', '益', '箕', '瘾', 'restaur@@', '鹣', '瘩', '矸', 'his', '灸', '勖', '括', '呢', '漯', '罟', '嚓', '焗', 'tru@@', '赛', '伶', '垢', '擘', '镠', '惰', '贬', '惙', '捺', 'sto@@', '犁', 'were', '艉', '佞', '配', '耙', '猢', '遗', '豊', '急', '蓟', '泓', '屑', '焊', '筲', '缈', '柔', '蜀', '珺', '怕', '淡', '.', '靛', '载', '堎', '潥', '懒', '邸', '惇', '粥', '斫', '漼', "re's", '簇', '修', '梏', 'tr@@', '肟', '葙', '鲸', '呐', '游', '廖', '收', '鳅', '覆', '琳', '熵', '翱', '民', '飨', '螳', '媳', '辖', '旯', '榴', '瑰', '嗯', '渤', 'ic', '熔', '暝', '柘', '鬟', '斋', '夸', '晢', '氰', '炫', "dn't", '押', '撺', '烁', '偶', '貅', '冀', '胎', '糊', '润', '咭', '各', '摐', '錞', '垌', '插', '舯', '嶙', '筋', '住', '酆', '镰', '工', '嗲', '铱', '暖', 'off@@', '茔', '髂', '挺', '瞑', '伤', '忌', '僬', '萜', '普', 'that@@', '爵', '稍', '瓠', '崛', 'um', '磐', '投', '畅', '缕', '楯', '悒', '判', '筢', '馇', '芰', '遒', '黩', '迪', '蝾', '腈', '才', '峙', '悟', '禩', '惋', 'five', '局', '砘', 'ver', '恋', '斓', '蹀', '犹', '绶', 'ep', '喛', '亩', '嵘', '汆', '狸', '迸', '盉', 'line', '妣', '躐', '琢', '赣', 'le', '疬', '穿', '焓', '跳', '匪', 'feat@@', '鹦', '璠', '蔸', '百', '愁', '芋', '吙', '谚', '熳', '哩', 'ad', '轧', '祇', '泬', '瘳', '列', '簃', '谦', 'lea@@', '垴', '啤', '滇', '饮', '瞰', '獭', '倏', '肿', '宜', '涮', '礓', '数', '屌', '濒', '蟜', '上', '詝', '褐', '眭', '菟', '瘅', '蝙', '眜', '羡', '硖', '背', '芣', '歃', '红', '血', '混', '橞', '樽', '纶', '圭', 'stay', '冗', '竑', '玥', '它', '怎', '薜', '殆', '看', '剖', '棠', '瘙', 'be@@', '裤', '篙', '带', '助', '骀', '锍', '廓', '库', '绅', '劂', '蝻', '锘', '濩', '等', '魑', '酿', '阆', '弑', '悖', '姹', '傀', 'which', '股', '湲', '贝', '相', '茑', '荔', '钲', '致', '沦', '另', '桑', '佳', '班', '昼', '埂', '剕', '箐', '岵', '涨', '柩', '两', '勰', '氮', '稽', '匝', 'ting', '瘀', '缔', '鹉', '雕', '蛮', '镞', '魈', '謇', 'me@@', 'set', '豇', 'h', '秫', '荬', '赢', '沩', '向', '石', '踖', '迥', '受', '抅', '镘', '嚭', '双', '锚', 'ex@@', '辉', '谠', '綉', 'ws', '吟', '湔', 'll@@', '市', '娥', '堢', '跽', '处', '厂', '朓', '笸', '悭', '孥', '嘀', '恻', '皱', '塄', 'who@@', '妙', '栽', '瓷', '稔', '碧', '哽', 'pol@@', '徨', 's', '邋', '息', '樨', '鼾', '拱', '谟', '墡', '匆', 'speci@@', 'ha@@', '庐', '鹿', 'ak@@', '镣', 'm', '唯', '州', '筒', '性', '髋', '哄', '镡', '贇', 'and', '菩', '姻', '聪', 'tion', '催', '醌', '唢', '雾', '堂', '芝', '擢', '匈', '悸', '钢', 'gy', '徉', '渡', '誊', 'ti@@', '军', '铸', '和', '沄', '兽', '短', '育', '韪', '邀', '谴', '耋', '浩', 'public', '胝', '猴', 'six', '黧', 'being', '弊', '捂', 'ical', ' 癖', '堆', '伧', '割', '引', 'la@@', '鼙', '逻', '堪', 'cap@@', '首', '谝', '瘦', '硐', '郑', '凑', 'fu@@', 'b@@', '睨', ' 骠', '佼', '篦', '贵', '铌', 'act', '呑', '竖', '莅', '陕', '飙', '黏', '镆', '屦', '仨', '敫', '马', '埜', '垇', '阇', '耒', '笩', '蓂', '妃', '讨', 'see@@', '吊', '踊', '弥', '梅', '坑', '渫', 'au@@', '赡', '坛', '苹', 'don@@', '察', '俚', 'fir@@', '濑', '块', '浏', '唵', '婴', '恙', '隈', '麽', 'ca@@', '莼', '御', '鞋', '禧', '衰', '潩', '茬', 'ven', '瞌', '鸥', '凭', '咯', '饺', '耄', 'uni@@', '瑟', '煟', '舟', '尅', '帚', '鼋', '嫔', '婄', '赟', '棼', '沪', '秃', '粧', 'roo@@', '蚳', '亏', '备', '压', '悍', '嚅', 'fin@@', '交', '癔', '跞', '莪', '闻', '罾', '炁', '钻', '鹆', '友', '傣', '糁', 'service', '邾', '窝', '扉', '鲡', '盛', '玖', '卫', 'ne@@', '眛', '鸟', '党', '杼', '碓', '篚', '裆', '钟', '大', '介', '揩', '砻', '卅', '弟', '颛', '嫪', '代', '会', '悻', '者', '箍', '番', '疯', 'sp@@', '聃', '瀚', '挥', '拷', '氡', '璎', '肪', 'tan', 'ul@@', '蠃', '龊', '家', '挽', 'must', '蹭', '樟', '令', '饶', 'buy', '忖', '娠', '手', 'house', '贷', '嘚', '莺', '茛', '宥', '刼', '傥', 'rest', '厄', '洙', '牝', '林', '铆', '觳', '嬢', '芥', '充', '絮', '陲', 'zer@@', '藤', 'air@@', '贤', ' 趾', '孝', '袖', '氇', '章', '柷', '恓', '麹', '胧', 'came', '休', '舲', '籼', '惕', '泼', '馨', '町', 'wer', '鹧', '咚', '崔', '嵂', '蝈', 'ils', '湛', '报', '蜷', '削', '根', '恣', '呙', '郫', '把', '窕', '癌', 'xt', '榘', '傫', '用', '撕', '蚩', '菰', '崖', '酐', '襄', 'cu@@', '呤', '帯', '炊', '揲', '啸', '鲶', 've@@', '艟', '嗜', 'ili@@', '凹', '球', '鬓', '荼', '给', '亘', '醯', '燔', '鳕', '侄', '簪', '洳', '嘛', '稠', "'t", '氘', '暍', '指', '泆', '总', 'offer', '糅', '膀', '毋', '蹲', '煅', '桔', '什', '浜', '八', '涧', '跬', '祷', '湫', '脸', '描', 'pr@@', '狱', '赉', '砼', '袜', '馑', '枪', '碁', '娜', '辇', '颐', '捩', '峥', '聆', '善', '询', '疝', '坜', '碚', '缥', '鼯', '钮', '腘', '溥', '脔', '殁', '肮', '泮', '捐', '槛', '吆', '河', '溆', '度', '殒', '骕', '饹', 'cour@@', '臜', '空', '2@@', '允', '枉', '蒯', 'or@@', '趔', '菠', '栌', 'view', 'possi@@', '仗', '鳄', '摊', '寝', '躯', '湝', '后', '删', '揪', '徐', '寒', '坨', '挣', '举', '氅', '臻', '饔', '丁', '彦', '泉', '曷', '衅', 'ei@@', '呸', '狍', '甓', '禽', '薄', 'shipping', 'ght', '慵', '幽', '匣', '炬', '访', 'four', '秒', '樗', '陶', '脓', '枰', '磷', '哕', '君', '脲', '灯', '回', '索', '璺', '邡', '愍', '薏', '辄', '珽', '離', '鬃', ' 赊', '枲', '邰', '佷', '颏', '寺', '抵', '鲅', '刎', '給', '戴', '溲', '袢', 'less', '逯', '封', 'food', '匮', '关', '尘', '皤', '鳍', '搏', '幌', '连', '杠', '蹋', '彷', '虺', '怀', '雏', '钿', '澄', '扶', '错', '樋', '黉', '胄', '薇', '阿', '墁', '觜', '利', '汵', '羟', '杀', '淋', '蓁', '腧', '哞', 'was', '钊', '胸', '艽', '撞', '虑', '煋', '飗', 'throu@@', '门', '玦', '琪', '煸', '酵', '婌', 'serv@@', '硿', '净', '夐', '撅', '頠', 'want', '犄', '厉', 'p@@', 'en', '斗', '层', '玫', 'life', '潘', '骐', '臃', '谮', '殃', '厌', '摄', '磾', 'v', 'dress', '络', '囱', '眯', '忽', '壸', '咐', '搁', '肤', '魇', '芬', '窡', '拉', '纮', '楣', '蔬', '寄', '妯', '教', '俪', '颡', '碛', '互', '奈', '憎', '炉', '蹰', '聂', '员', '呶', '瞬', 'il@@', '恿', '阙', '卡', 'mi@@', '禾', '椴', 'yo@@', '帀', '醵', '帝', '隔', '忒', '哑', '効', '楗', '鼱', '塽', '苴', '蜞', '健', '醅', 'ju@@', '新', '程', '茗', '琰', '几', '揍', '匍', '砣', '禳', '罗', '勿', '擗', '畛', '框', '泒', '析', ' 沢', '偷', '繁', '嗣', '呵', '念', 'so@@', '溷', '曩', 'spon@@', '狼', '倔', '威', '潭', '踯', '晁', '吩', '袅', '喀', '洌', '炯', '纸', '抽', '簧', 'c@@', '买', '吖', '俬', '梓', '叡', '祼', '烃', '荃', '眀', '<unk>'], split_with_space=True, init=None, input_size=560, ctc_conf={'dropout_rate': 0.0, 'ctc_type': 'builtin', 'reduce': True, 'ignore_nan_grad': True}, joint_net_conf=None, use_preprocessor=True, token_type='char', bpemodel=None, non_linguistic_symbols=None, cleaner=None, g2p=None, speech_volume_normalize=None, rir_scp=None, rir_apply_prob=1.0, noise_scp=None, noise_apply_prob=1.0, noise_db_range='13_15', frontend=None, frontend_conf={}, specaug='specaug_lfr', specaug_conf={'apply_time_warp': False, 'time_warp_window': 5, 'time_warp_mode': 'bicubic', 'apply_freq_mask': True, 'freq_mask_width_range': [0, 30], 'lfr_rate': 6, 'num_freq_mask': 1, 'apply_time_mask': True, 'time_mask_width_range': [0, 12], 'num_time_mask': 1}, normalize=None, normalize_conf={}, model='paraformer', model_conf={'ctc_weight': 0.0, 'lsm_weight': 0.1, 'length_normalized_loss': True, 'predictor_weight': 1.0, 'predictor_bias': 1, 'sampling_ratio': 0.75}, preencoder=None, preencoder_conf={}, encoder='sanm', encoder_conf={'output_size': 512, 'attention_heads': 4, 'linear_units': 2048, 'num_blocks': 50, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'attention_dropout_rate': 0.1, 'input_layer': 'pe', 'pos_enc_class': 'SinusoidalPositionEncoder', 'normalize_before': True, 'kernel_size': 11, 'sanm_shfit': 0, 'selfattention_layer_type': 'sanm'}, postencoder=None, postencoder_conf={}, decoder='paraformer_decoder_sanm', decoder_conf={'attention_heads': 4, 'linear_units': 2048, 'num_blocks': 16, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'self_attention_dropout_rate': 0.1, 'src_attention_dropout_rate': 0.1, 'att_layer_num': 16, 'kernel_size': 11, 'sanm_shfit': 0}, predictor='cif_predictor_v2', predictor_conf={'idim': 512, 'threshold': 1.0, 'l_order': 1, 'r_order': 1, 'tail_threshold': 0.45}, gpu_id=1, required=['output_dir', 'token_list'], distributed=False, version='202211')
# writer None
# _bs 1  batch_size=1
# keys ['asr_example_zh']
# batch {'speech': tensor([[ 9.1553e-05,  9.1553e-05,  9.1553e-05,  ..., -1.5259e-04, -9.1553e-05, -6.1035e-05]]), 'speech_lengths': tensor([88747])} batch是一个字典，里面是要解码的wav特征向量
# results [('欢迎大家来体验达摩院推出的语音识别模型</s>', ['欢', '迎', '大', '家', '来', '体', '验', '达', '摩', '院', '推', '出', '的', '语', '音', '识', '别', '模', '型', '</s>'], [7023, 2998, 7950, 7977, 7188, 4536, 892, 4844, 519, 1111, 4158, 5065, 1373, 523, 6431, 3359, 352, 3892, 4869, 2], Hypothesis(yseq=tensor([   1, 7023, 2998, 7950, 7977, 7188, 4536,  892, 4844,  519, 1111, 4158, 5065, 1373,  523, 6431, 3359,  352, 3892, 4869,    2,    2], device='cuda:0'), score=tensor(-1.6451, device='cuda:0'), scores={}, states={}), 93, 1108)]
# result [('欢迎大家来体验达摩院推出的语音识别模型</s>', ['欢', '迎', '大', '家', '来', '体', '验', '达', '摩', '院', '推', '出', '的', '语', '音', '识', '别', '模', '型', '</s>'], [7023, 2998, 7950, 7977, 7188, 4536, 892, 4844, 519, 1111, 4158, 5065, 1373, 523, 6431, 3359, 352, 3892, 4869, 2], Hypothesis(yseq=tensor([   1, 7023, 2998, 7950, 7977, 7188, 4536,  892, 4844,  519, 1111, 4158, 5065, 1373,  523, 6431, 3359,  352, 3892, 4869,    2,    2], device='cuda:0'), score=tensor(-1.6451, device='cuda:0'), scores={}, states={}))]
# text is not None
# item {'key': 'asr_example_zh', 'value': '欢迎大家来体验达摩院推出的语音识别模型'}