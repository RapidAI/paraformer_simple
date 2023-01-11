#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0,1" # set gpus, e.g., CUDA_VISIBLE_DEVICES="0,1"
gpu_num=2
count=1
gpu_inference=true # Whether to perform gpu decoding, set false for cpu decoding
njob=1 # the number of jobs for each gpu
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# general configuration
feats_dir="../DATA" #feature output dictionary, for large data
exp_dir="."
lang=zh
dumpdir=dump/fbank
feats_type=fbank
token_type=char
scp=feats.scp
type=kaldi_ark
stage=1
stop_stage=4

# feature configuration
feats_dim=560
sample_frequency=16000
nj=32
speed_perturb="1.0"
lfr=True
lfr_m=7
lfr_n=6

init_model_name=speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch  # pre-trained model, download from modelscope during fine-tuning
model_revision="v1.0.4"     # please do not modify the model revision
cmvn_file=init_model/${init_model_name}/am.mvn
seg_file=init_model/${init_model_name}/seg_dict
vocab=init_model/${init_model_name}/tokens.txt

# data
dataset=  # dataset (include train/wav.scp, train/text, dev/wav.scp, dev/text, optional test/wav.scp test/text)

# exp tag
tag=""

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr_paraformer_sanm_50e_16d_2048_512_lfr6.yaml
init_param="init_model/${init_model_name}/model.pb"

inference_config=conf/decode_asr_transformer_noctc_1best.yaml
inference_asr_model=valid.acc.ave_10best.pth

. utils/parse_options.sh || exit 1;

# download model from modelscope
python modelscope_utils/download_model.py --model_name ${init_model_name} --model_revision ${model_revision}

if [ ! -d ${HOME}/.cache/modelscope/hub/damo/${init_model_name} ]; then
    echo "${HOME}/.cache/modelscope/hub/damo/${init_model_name} must exist"
    exit 1
else
    if [ -d init_model/${init_model_name} ]; then
        echo "init_model/${init_model_name} is already exists. if you want to decode again, please delete init_model/${init_model_name} first."
    else
        mkdir -p init_model/${init_model_name}
        cp -r ${HOME}/.cache/modelscope/hub/damo/${init_model_name}/* init_model/${init_model_name}
    fi
fi

model_dir="baseline_$(basename "${asr_config}" .yaml)_${feats_type}_${lang}_${token_type}_${tag}"

# you can set gpu num for decoding here
gpuid_list=$CUDA_VISIBLE_DEVICES  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

if ${gpu_inference}; then
    inference_nj=$njob
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi

[ ! -d ${dataset} ] && echo "$0: Training data is required" && exit 1;
[ ! -f ${dataset}/train/wav.scp ] && [ ! -f ${dataset}/train/text ] && echo "$0: Training data wav.scp or text is not found" && exit 1;

if [ ! -d "${dataset}/dev" ]; then
    utils/fix_data.sh ${dataset}/train
    utils/subset_data_dir_tr_cv.sh --dev-num-utt 1000 ${dataset}/train ${dataset}
fi
if [ ! -d "${dataset}/test" ]; then
   test_sets="dev" 
fi

feat_train_dir=${feats_dir}/${dumpdir}/train; mkdir -p ${feat_train_dir}
feat_dev_dir=${feats_dir}/${dumpdir}/dev; mkdir -p ${feat_dev_dir}
feat_test_dir=${feats_dir}/${dumpdir}/test; mkdir -p ${feat_test_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    # compute fbank features
    fbankdir=${feats_dir}/fbank
    utils/compute_fbank.sh --cmd "$train_cmd" --nj $nj --sample_frequency ${sample_frequency} --speed_perturb ${speed_perturb} \
        ${dataset}/train ${exp_dir}/exp/make_fbank/train ${fbankdir}/train
    utils/fix_data_feat.sh ${fbankdir}/train
    utils/compute_fbank.sh --cmd "$train_cmd" --nj $nj --sample_frequency ${sample_frequency} \
        ${dataset}/dev ${exp_dir}/exp/make_fbank/dev ${fbankdir}/dev
    utils/fix_data_feat.sh ${fbankdir}/dev
    if [ -d "${dataset}/test" ]; then
        utils/compute_fbank.sh --cmd "$train_cmd" --nj $nj --sample_frequency ${sample_frequency} \
            ${dataset}/test ${exp_dir}/exp/make_fbank/test ${fbankdir}/test
        utils/fix_data_feat.sh ${fbankdir}/test
    fi

    echo "apply low_frame_rate and cmvn"
    [ ! -f ${cmvn_file} ] && echo "$0: cmvn file is required" && exit 1;
    utils/apply_lfr_and_cmvn.sh --cmd "$train_cmd" --nj $nj \
        --lfr $lfr --lfr-m $lfr_m --lfr-n $lfr_n \
        ${fbankdir}/train ${cmvn_file} ${exp_dir}/exp/make_fbank/train ${feat_train_dir}
    utils/apply_lfr_and_cmvn.sh --cmd "$train_cmd" --nj $nj \
        --lfr $lfr --lfr-m $lfr_m --lfr-n $lfr_n \
        ${fbankdir}/dev ${cmvn_file} ${exp_dir}/exp/make_fbank/dev ${feat_dev_dir}
    if [ -d "${dataset}/test" ]; then
        utils/apply_lfr_and_cmvn.sh --cmd "$train_cmd" --nj $nj \
            --lfr $lfr --lfr-m $lfr_m --lfr-n $lfr_n \
            ${fbankdir}/test ${cmvn_file} ${exp_dir}/exp/make_fbank/test ${feat_test_dir}
    fi

    echo "Text Tokenize"
    # 我爱reading->我 爱 read@@ ing
    utils/text_tokenize.sh --cmd "$train_cmd" --nj $nj ${fbankdir}/train ${seg_file} ${feat_train_dir}/log ${feat_train_dir}
    utils/fix_data_feat.sh ${feat_train_dir}
    utils/text_tokenize.sh --cmd "$train_cmd" --nj $nj ${fbankdir}/dev ${seg_file} ${feat_dev_dir}/log ${feat_dev_dir}
    utils/fix_data_feat.sh ${feat_dev_dir}
    if [ -d "${dataset}/test" ]; then
        cp ${fbankdir}/test/text ${feat_test_dir}
    fi
fi

token_list=${feats_dir}/data/${lang}_token_list/char/tokens.txt
echo "dictionary: ${token_list}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary Preparation"
    mkdir -p ${feats_dir}/data/${lang}_token_list/char/
    cp $vocab ${token_list}

    vocab_size=$(wc -l <${token_list})
    awk -v v=,${vocab_size} '{print $0v}' ${feat_train_dir}/text_shape > ${feat_train_dir}/text_shape.char
    awk -v v=,${vocab_size} '{print $0v}' ${feat_dev_dir}/text_shape > ${feat_dev_dir}/text_shape.char
    mkdir -p ${feats_dir}/asr_stats_fbank_zh_char/train
    mkdir -p ${feats_dir}/asr_stats_fbank_zh_char/dev
    cp ${feat_train_dir}/speech_shape ${feat_train_dir}/text_shape ${feat_train_dir}/text_shape.char ${feats_dir}/asr_stats_fbank_zh_char/train
    cp ${feat_dev_dir}/speech_shape ${feat_dev_dir}/text_shape ${feat_dev_dir}/text_shape.char ${feats_dir}/asr_stats_fbank_zh_char/dev
fi

# Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training"
    # update asr train config.yaml
    python modelscope_utils/update_config.py --modelscope_config init_model/${init_model_name}/finetune.yaml --finetune_config ${asr_config} --output_config init_model/${init_model_name}/asr_finetune_config.yaml
    finetune_config=init_model/${init_model_name}/asr_finetune_config.yaml

    mkdir -p ${exp_dir}/exp/${model_dir}
    mkdir -p ${exp_dir}/exp/${model_dir}/log
    INIT_FILE=${exp_dir}/exp/${model_dir}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
            asr_train_paraformer.py \
                --gpu_id $gpu_id \
                --use_preprocessor true \
                --token_type $token_type \
                --token_list $token_list \
                --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/${scp},speech,${type} \
                --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/text,text,text \
                --train_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${train_set}/speech_shape \
                --train_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${train_set}/text_shape.char \
                --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/${scp},speech,${type} \
                --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/text,text,text \
                --valid_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${valid_set}/speech_shape \
                --valid_shape_file ${feats_dir}/asr_stats_fbank_zh_char/${valid_set}/text_shape.char  \
                --resume true \
                --output_dir ${exp_dir}/exp/${model_dir} \
                --init_param $init_param \
                --config $finetune_config \
                --input_size $feats_dim \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --multiprocessing_distributed true \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
fi

# Testing Stage
# Testing Stage
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Inference"
    for dset in ${test_sets}; do
        asr_exp=${exp_dir}/exp/${model_dir}
        inference_tag="$(basename "${inference_config}" .yaml)"
        _dir="${asr_exp}/${inference_tag}/${inference_asr_model}/${dset}"
        _logdir="${_dir}/logdir"
        if [ -d ${_dir} ]; then
            echo "${_dir} is already exists. if you want to decode again, please delete this dir first."
            exit 0
        fi
        mkdir -p "${_logdir}"
        _data="${feats_dir}/${dumpdir}/${dset}"
        key_file=${_data}/${scp}
        num_scp_file="$(<${key_file} wc -l)"
        _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
        split_scps=
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}
        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
            python -m funasr.bin.asr_inference_launch \
                --batch_size 64 \
                --ngpu "${_ngpu}" \
                --njob ${njob} \
                --gpuid_list ${gpuid_list:0:1} \
                --data_path_and_name_and_type "${_data}/${scp},speech,${type}" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --asr_train_config "${asr_exp}"/config.yaml \
                --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
                --output_dir "${_logdir}"/output.JOB \
                --mode paraformer \
                ${_opts}

        for f in token token_int score text; do
            if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | sort -k1 >"${_dir}/${f}"
            fi
        done
        python utils/proce_text.py ${_dir}/text ${_dir}/text.proc
        python utils/proce_text.py ${_data}/text ${_data}/text.proc
        python utils/compute_wer.py ${_data}/text.proc ${_dir}/text.proc ${_dir}/text.cer
        tail -n 3 ${_dir}/text.cer > ${_dir}/text.cer.txt
        cat ${_dir}/text.cer.txt
    done
fi
