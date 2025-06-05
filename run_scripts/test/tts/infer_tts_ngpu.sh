#!/bin/bash
WORK_DIR=$(pwd)

# ME=$(basename "$0")
# ME=${ME%.*}
# TIMESTAMP=$(date '+%m%d%y-%H%M%S')

CACHE_DIR=/home/tione/notebook/alanhshao/pretrained_models

SAVE_AUDIO=True
TEXT_ONLY=False

# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s4_zh-110224-020715/checkpoint-34000
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3v2p2_zh/checkpoint-34500
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3v2p2p2_zh/checkpoint-32500
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/backups/vita_qwen2_s3v2p2p2_zh/checkpoint-32500
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3v2p2p2_zh/checkpoint-136000
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3v2p2p2_zh/checkpoint-51500
MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s3v2p2p2_zh
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/vita_qwen2moe-chat_s3/checkpoint-79600
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-addExperts_s3/checkpoint-23200
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-chat_s3/checkpoint-8000
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/vita_deepseek-chat_s3/checkpoint-5600
MODEL_NAME_OR_PATH=outputs/lucy_deepseek-chat_s3/checkpoint-10400
#MODEL_NAME_OR_PATH=outputs/bak0407/lucy_deepseek-chat_s3/checkpoint-14800/
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-addExperts_s3/checkpoint-45200
MODEL_NAME_OR_PATH=outputs/lucy_deepseek_adaptive_s3/checkpoint-13200
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek_adaptive_s3/checkpoint-22800
#MODEL_NAME_OR_PATH=outputs/lucy_deepseek_adaptive_rl
#MODEL_NAME_OR_PATH=outputs/lucy_deepseek_adaptive_s3_audio20/checkpoint-8000
#MODEL_NAME_OR_PATH=outputs/bak0413/lucy_deepseek-chat_s3/checkpoint-16400
#MODEL_NAME_OR_PATH=/home/tione/notebook/alanhshao/LUCY/outputs/bak0401/vita_deepseek-chat_s3/checkpoint-33200
#MODEL_NAME_OR_PATH=/home/tione/notebook/alanhshao/LUCY/outputs/vita_deepseek-chat_s3
#/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2_s3v2p5_zh/checkpoint-26500
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/vita-e2e/outputs/vita_qwen2_s4_zh_v4/checkpoint-43400
AUDIO_ENCODER="/mnt/data/hetinggao/models/whisper-medium"

gpu=$1
gpu_num=$2
lang=en
#testname=librispeech_asr
#testname=AIshell
testname=seed_tts
#testname=train.100.clean
#testname=seed_tts_hardcase
EXPNAME=$(basename `dirname $MODEL_NAME_OR_PATH`)
CKPTNAME=$(basename $MODEL_NAME_OR_PATH)
SUFFIX=test
OUTPUT_PATH=$WORK_DIR/generated/$EXPNAME-$CKPTNAME-$SUFFIX/${lang}/${testname}/split${gpu_num}/$gpu
# OUTPUT_PATH=$WORK_DIR/generated/$CKPTNAME-best-$SUFFIX

# seed_tts
INPUT_FILE=/mnt/data/alanhshao/vita-e2e/datasets/seedtts_testset/${lang}/split${gpu_num}/text.json.$gpu
#INPUT_FILE=datasets/seedtts_testset/${lang}/split${gpu_num}/hardcase.json.$gpu
# libri test other
#INPUT_FILE=datasets/seedtts_testset/${lang}/${testname}/split${gpu_num}/test.other.json.$gpu
# libri test clean
#INPUT_FILE=datasets/seedtts_testset/${lang}/${testname}/split${gpu_num}/test.clean.json.$gpu
# aishell
#INPUT_FILE=datasets/seedtts_testset/${lang}/${testname}/split${gpu_num}/test.json.$gpu
# libri train clean 100
#INPUT_FILE=datasets/seedtts_testset/${lang}/librispeech_asr/$testname/split${gpu_num}/train.100.clean.json.$gpu
use_audio_input=False


mkdir -p $OUTPUT_PATH

export PYTHONPATH=$WORK_DIR
python3 src/scripts/infer_tts_ngpu.py \
    --audio_feature_rate 50 \
    --sample_rate 16000 \
    --model_type "deepseek_v2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder $AUDIO_ENCODER \
    --model_hidden_size 2048 \
    --freeze_backbone True \
    --freeze_audio_encoder True \
    --audio_encoder_hidden_size 1024 \
    --audio_projector_hidden_size 7168 \
    --audio_num_codebook 7 \
    --text_vocab_size 102400 \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --audio_special_tokens 64 \
    --cache_dir ${CACHE_DIR} \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" \
	--max_code_length 1000 \
    --max_keep_sample_size $((25*16000)) \
    --input_file ${INPUT_FILE} \
    --output_path ${OUTPUT_PATH} \
	--save_audio ${SAVE_AUDIO} \
	--output_text_only ${TEXT_ONLY} \
    --use_audio_input ${use_audio_input} \
    --gpu ${gpu}

unused="""
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
	
"""
