#coding=utf-8
import os,sys,time,codecs
from funasr.bin.asr_inference_paraformer import inference
model_path="C:\\Users\\deep\\.cache\\modelscope\\hub\\damo\\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\\"
audio_in= [{'key': 'wav_%d'%i, 'file': './asr_example_zh.wav'} for i in range(10)]
cmd={'model_type': 'pytorch', 'ngpu': 1, 'log_level': 'ERROR', 'audio_in':audio_in, 'name_and_type': ['speech', 'sound', model_path+'am.mvn'], 'asr_model_file': model_path+'model.pb', 'idx_text': '', 'sampled_ids': 'seq2seq/sampled_ids', 'sampled_lengths': 'seq2seq/sampled_lengths', 'lang': 'zh-cn', 'code_base': 'funasr', 'mode': 'paraformer', 'fs': {'audio_fs': 16000, 'model_fs': 16000}, 'beam_size': 1, 'penalty': 0.0, 'maxlenratio': 0.0, 'minlenratio': 0.0, 'ctc_weight': 0.0, 'lm_weight': 0.0, 'asr_train_config': model_path+'config.yaml', 'lm_file': model_path+'lm.pb', 'lm_train_config': model_path+'config_lm.yaml', 'batch_size': 1, 'frontend_conf': {'fs': 16000, 'win_length': 400, 'hop_length': 160, 'window': 'hamming', 'n_mels': 80, 'lfr_m': 7, 'lfr_n': 6}, 'token_num_relax': None, 'decoding_ind': None, 'decoding_mode': None, 'num_workers': 0}
global global_asr_language,global_sample_rate
global_asr_language = cmd['lang']
global_sample_rate = cmd['fs']
if cmd['ngpu'] > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
asr_result = inference(
    mode=cmd['mode'],
    batch_size=cmd['batch_size'],
    maxlenratio=cmd['maxlenratio'],
    minlenratio=cmd['minlenratio'],
    beam_size=cmd['beam_size'],
    ngpu=cmd['ngpu'],
    num_workers=cmd['num_workers'],
    ctc_weight=cmd['ctc_weight'],
    lm_weight=cmd['lm_weight'],
    penalty=cmd['penalty'],
    log_level=cmd['log_level'],
    data_path_and_name_and_type=cmd['name_and_type'],
    audio_lists=cmd['audio_in'],
    asr_train_config=cmd['asr_train_config'],
    asr_model_file=cmd['asr_model_file'],
    lm_file=cmd['lm_file'],
    lm_train_config=cmd['lm_train_config'],
    frontend_conf=cmd['frontend_conf'],
    token_num_relax=cmd['token_num_relax'],
    decoding_ind=cmd['decoding_ind'],
    decoding_mode=cmd['decoding_mode'])
print(asr_result)