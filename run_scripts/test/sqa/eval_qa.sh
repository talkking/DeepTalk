#!/bin/bash
WORK_DIR=$(pwd)

# ME=$(basename "$0")
# ME=${ME%.*}
# TIMESTAMP=$(date '+%m%d%y-%H%M%S')

CACHE_DIR=/mnt/data/hetinggao/models

SAVE_AUDIO=True #False #True
TEXT_ONLY=False

# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v3p3p3_zh/checkpoint-7600
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/backups/vita_qwen2-7b-instruct_s3v3p3_zh/checkpoint-15600
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v3p3_zh/checkpoint-44400
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v3p3p3p2_zh/checkpoint-3600
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v3p3p3p2_zh/checkpoint-12400
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/backups/vita_qwen2-7b-instruct_s3v3p3p3_zh/checkpoint-5600
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v3p3p4_zh/checkpoint-2400
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v3p3p1_zh/checkpoint-5600
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v10p1_zh/checkpoint-48000
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v10p1_zh/checkpoint-62000
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v10p2_zh/checkpoint-4000
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-addExperts_s2p2/checkpoint-1060
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-addExperts_s3/checkpoint-12800
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-addExperts_s3/checkpoint-22400
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-chat_s3/checkpoint-7200
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/bak0320/lucy_deepseek-addExperts_s3/checkpoint-12800
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/bak0403/lucy_deepseek-chat_s3/checkpoint-17012
#MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/vita_qwen2moe-chat_s3
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-chat_s3/checkpoint-10400
#MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/bak0407/lucy_deepseek-chat_s3/checkpoint-14800
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-addExperts_s3/checkpoint-45200
MODEL_NAME_OR_PATH=outputs/lucy_deepseek_adaptive_s2p2/checkpoint-2000
MODEL_NAME_OR_PATH=outputs/lucy_deepseek_adaptive_s3/checkpoint-13200
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek_adaptive_s3/checkpoint-22800
#MODEL_NAME_OR_PATH=outputs/lucy_deepseek_adaptive_s3_audio20/checkpoint-11200
#/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-chat_s3/checkpoint-6000
#/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-addExperts_s2p2/checkpoint-200
#/mnt/data/alanhshao/vita-e2e/outputs/vita_qwen2moe-chat_s3/checkpoint-3600
#/mnt/data/alanhshao/vita-e2e/outputs/vita_deepseek-chat_s3/checkpoint-5600
#/mnt/data/alanhshao/vita-e2e/outputs/lucy_qwen2moe-chat_s3/checkpoint-58000
#/mnt/data/alanhshao/vita-e2e/outputs/lucy_qwen2moe-chat_s3/checkpoint-39200
#/mnt/data/alanhshao/vita-e2e/outputs/vita_qwen2moe-chat_s3/checkpoint-79200
#/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v10p2p1_zh/checkpoint-14000
AUDIO_ENCODER=/mnt/data/hetinggao/models/whisper-medium
#/mnt/data/hetinggao/models/audio-encoder-Qwen2-7B-instruct-weight-base-11wh-tunning

# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v8p1_zh/checkpoint-49200
# AUDIO_ENCODER=/mnt/data/hetinggao/models/audio-encoder-Qwen2.5-7B-instruct-weight-base-11wh-tunning

# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v9p1_zh/checkpoint-34400
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v9p1_zh/checkpoint-25600
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v9p1_zh/checkpoint-54400
# AUDIO_ENCODER=/mnt/data/hetinggao/models/whisper-medium

# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v5p1p1_zh/checkpoint-4800
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v6p1_zh/checkpoint-21200
# AUDIO_ENCODER=/mnt/data/hetinggao/models/audio-encoder-Qwen2.5-7B-1219-weight-base-11wh-tunning

EXPNAME=$(basename `dirname $MODEL_NAME_OR_PATH`)
CKPTNAME=$(basename $MODEL_NAME_OR_PATH)

SPLIT=$1
# SPLIT=triva_qa/triva_qa.00
# SPLIT=LLAMA1-test-set.question
# SPLIT=triva_qa
# SPLIT=web_questions
DATA_DIR=/mnt/data/alanhshao/vita-e2e/datasets/SpokenQA/speech_qa_set #/mnt/data/hetinggao/manifest/speech_qa_set
QUESTIONS=$DATA_DIR/$SPLIT
SUFFIX=`basename $DATA_DIR`_$SPLIT
use_audio_input=$2 #True or False

OUTPUT_PATH=$WORK_DIR/generated/eval_qa-$EXPNAME-$CKPTNAME-topk1/$SUFFIX.txt
echo $OUTPUT_PATH
# OUTPUT_PATH=$WORK_DIR/generated/$CKPTNAME-best-$SUFFIX

mkdir -p `dirname $OUTPUT_PATH`

export PYTHONPATH=$WORK_DIR

set -x
python src/scripts/eval_qa.py \
    --audio_feature_rate 50 \
    --sample_rate 16000 \
    --model_type "lucy-deepseek_v2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder $AUDIO_ENCODER \
    --model_hidden_size 2048 \
    --freeze_backbone True \
    --freeze_audio_encoder True \
    --audio_encoder_hidden_size 1024 \
    --audio_projector_hidden_size 7168 \
    --audio_num_codebook 7 \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --text_vocab_size 102400 \
    --audio_special_tokens 64 \
    --cache_dir ${CACHE_DIR} \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" "F10" "M29" "ER" \
	--max_code_length 1000 \
    --max_keep_sample_size $((30*16000)) \
    --output_path ${OUTPUT_PATH} \
	--save_audio ${SAVE_AUDIO} \
	--output_text_only ${TEXT_ONLY} \
	--use_audio_input ${use_audio_input} \
	--questions $QUESTIONS
set +x
unused="""
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
	
"""
