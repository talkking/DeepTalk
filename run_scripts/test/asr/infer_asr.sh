#!/bin/bash
WORK_DIR=$(pwd)

# ME=$(basename "$0")
# ME=${ME%.*}
# TIMESTAMP=$(date '+%m%d%y-%H%M%S')

CACHE_DIR=/mnt/data/hetinggao/models

MANIFEST_DIR=/mnt/data/alanhshao/vita-e2e/datasets/ASR/metadata/AIShell/test
#MANIFEST_DIR=/mnt/data/alanhshao/vita-e2e/datasets/ASR/metadata/WenetSpeech
#/mnt/data/hetinggao/manifest/WenetSpeech
#MANIFEST_DIR=/mnt/data/hetinggao/manifest/SER/metadata/sub10

#MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_qwen2moe-chat_s3/checkpoint-39200#
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/vita_deepseek-chat_s1/checkpoint-101600
#/mnt/data/alanhshao/vita-e2e/outputs/vita_qwen2moe-chat_s1/checkpoint-107600

#/mnt/data/alanhshao/vita-e2e/outputs/vita_qwen2moe-chat_s1/checkpoint-33200
#/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2_s1v3_zh/checkpoint-55000
CKPT_PATH=$MODEL_NAME_OR_PATH
EXPNAME=$(basename `dirname $MODEL_NAME_OR_PATH`)
CKPTNAME=$(basename $MODEL_NAME_OR_PATH)
SUFFIX=test
OUTPUT_PATH=$WORK_DIR/generated/$EXPNAME-$CKPTNAME-$SUFFIX
# OUTPUT_PATH=$OUTPUT_DIR/hyp.txt
mkdir -p $OUTPUT_PATH
TASKS="ASRE"
TASKS="ASRRAW"
SAVE_AUDIO=False
TEXT_ONLY=True
# MODEL_NAME_OR_PATH=/mnt/data/hetinggao/models/Qwen2-1.5B-Instruct

testset=$1
AUDIO_IN="${MANIFEST_DIR}/${testset}.tsv"
TEXT_IN="${MANIFEST_DIR}/${testset}.wrd"
TEXT_OUT="${MANIFEST_DIR}/${testset}.wrd"
#CODEC_OUT="${MANIFEST_DIR}/${testset}.snac"

# AUDIO_IN="${MANIFEST_DIR}/test_meeting.tsv"
# TEXT_IN="${MANIFEST_DIR}/test_meeting.wrd"
# TEXT_OUT="${MANIFEST_DIR}/test_meeting.wrd"
# CODEC_OUT="${MANIFEST_DIR}/test_meeting.snac"

# AUDIO_IN="${MANIFEST_DIR}/test_net.tsv"
# TEXT_IN="${MANIFEST_DIR}/test_net.wrd"
# TEXT_OUT="${MANIFEST_DIR}/test_net.wrd"
# CODEC_OUT="${MANIFEST_DIR}/test_net.snac"

export PYTHONPATH=$WORK_DIR
python src/scripts/infer_asr.py \
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --audio_feature_rate 50 \
    --sample_rate 16000 \
    --tasks ${TASKS} \
    --model_type "vita-deepseek_v2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder /mnt/data/hetinggao/models/whisper-medium \
    --model_hidden_size 2048 \
    --freeze_backbone True \
    --freeze_audio_encoder True \
	--bf16 True \
	--fp16 False \
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
	--max_code_length 100 \
    --max_keep_sample_size $((30*16000)) \
    --ckpt_path ${CKPT_PATH} \
    --output_path ${OUTPUT_PATH} \
	--save_audio ${SAVE_AUDIO} \
	--output_text_only ${TEXT_ONLY} \
	--testset ${testset}  

unused="""
    --shuffle False \
"""
