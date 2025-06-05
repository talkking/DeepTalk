#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUTPUT_DIR=${WORK_DIR}/outputs/${ME}

CACHE_DIR=/mnt/data/hetinggao/models
# MODEL_NAME_OR_PATH="/mnt/data/hetinggao/models/Qwen2.5-7B-1219"
# TOKENIZER_NAME_OR_PATH="/mnt/data/hetinggao/Projects/vita-e2e/backups/vita_qwen2-7b-instruct_s3v3p3_zh/checkpoint-15600/modified_tokenizer"
#MODEL_NAME_OR_PATH="/mnt/data/hetinggao/Projects/vita-e2e/outputs/vita_qwen2-7b-instruct_s3v8p1_zh"
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/vita_deepseek-chat_s1/checkpoint-101600
#"/mnt/data/alanhshao/vita-e2e/outputs/vita_qwen2moe-chat_s2/checkpoint-24400"
AUDIO_ENCODER="/mnt/data/hetinggao/models/whisper-medium"
#"/mnt/data/hetinggao/models/audio-encoder-Qwen2.5-7B-instruct-weight-base-11wh-tunning"

WENETEM_DIR="/mnt/data/hetinggao/manifest/SER/stage1/threshold0.95"

UNION60W_DIR="/mnt/data/hetinggao/manifest/text/6w"
COMMON_DIR="/mnt/data/hetinggao/manifest/AudioQA/jsons/AudioQA-1450K-filtered-v4"
FC_DIR="/mnt/data/hetinggao/manifest/vita2tts_v4/jsons_v2/fc"
FC2_DIR="/mnt/data/hetinggao/manifest/vita2tts_v4/jsons_v2/fc_chaofan"
NATURAL_DIR="/mnt/data/hetinggao/manifest/vita2tts_v4/jsons_v2/audioagent"
EMO1_DIR="/mnt/data/hetinggao/manifest/Emotion_Control/stage3_v3/QNeutral_v2"
EMO2_DIR="/mnt/data/hetinggao/manifest/Emotion_Control/stage3_v3/QEmo_v2"
SQA_DIR="/mnt/data/hetinggao/manifest/SQA/jsons"
# NUM_DIR="/mnt/data/hetinggao/manifest/num_tts/jsons/num_tts"
# ID_DIR="/mnt/data/hetinggao/manifest/Identity/jsons"
NEGA_DIR="/mnt/data/hetinggao/manifest/vita_negative"
NOISE_DIR="/mnt/data/hetinggao/manifest/noise_negative"
SWQA_DIR="/mnt/data/hetinggao/manifest/spoken-web-questions/jsons/spoken-web-questions"

DATA_JSONS="$COMMON_DIR/train.json $COMMON_DIR/train.json "`
`"$EMO1_DIR/train.json $EMO1_DIR/train.json $EMO1_DIR/train.json $EMO1_DIR/train.json "` 
`"$EMO2_DIR/train.json $EMO2_DIR/train.json $EMO2_DIR/train.json $EMO2_DIR/train.json "`
`"$FC_DIR/train.json $FC_DIR/train.json $FC_DIR/train.json $FC_DIR/train.json "`
`"$FC2_DIR/train.json $FC2_DIR/train.json $FC2_DIR/train.json $FC2_DIR/train.json "`
`"$NATURAL_DIR/train.json $NATURAL_DIR/train.json $NATURAL_DIR/train.json $NATURAL_DIR/train.json "`
`"$SQA_DIR/train.json $SQA_DIR/train.json $SQA_DIR/train.json $SQA_DIR/train.json "`
`"$SWQA_DIR/train.json"
DATA_JSONS="$COMMON_DIR/train.json $COMMON_DIR/train.json $COMMON_DIR/train.json $UNION60W_DIR/train.json"


EVAL_DATA_JSONS="$COMMON_DIR/eval.json $COMMON_DIR/eval.json "`
`"$EMO1_DIR/eval.json $EMO1_DIR/eval.json $EMO1_DIR/eval.json $EMO1_DIR/eval.json "`
`"$EMO2_DIR/eval.json $EMO2_DIR/eval.json $EMO2_DIR/eval.json $EMO2_DIR/eval.json "`
`"$FC_DIR/eval.json $FC_DIR/eval.json $FC_DIR/eval.json $FC_DIR/eval.json "`
`"$FC2_DIR/eval.json $FC2_DIR/eval.json $FC2_DIR/eval.json $FC2_DIR/eval.json "` 
`"$NATURAL_DIR/eval.json $NATURAL_DIR/eval.json $NATURAL_DIR/eval.json $NATURAL_DIR/eval.json "`
`"$SQA_DIR/eval.json $SQA_DIR/eval.json $SQA_DIR/eval.json $SQA_DIR/eval.json "`
`"$SWQA_DIR/eval.json"

EVAL_DATA_JSONS="$COMMON_DIR/eval.json $COMMON_DIR/eval.json $COMMON_DIR/eval.json $UNION60W_DIR/eval.json"

TASKS="RQACONV RQACONVA "`
`"AQACONVA_EMO RQACONV_EMO RQACONVA_EMO RQACONV_EMO "`
`"AQACONVA_EMO AQACONV_EMO AQACONVA_EMO AQACONV_EMO "`
`"AQACONVA RQACONV RQACONVA RQACONV "`
`"AQACONVA RQACONV RQACONVA RQACONV "`
`"AQACONVA_NTRL RQACONV_NTRL RQACONVA_NTRL RQACONV_NTRL "`
`"AQACONVA RQACONV RQACONVA RQACONV "`
`"RQACONV"

TASKS="AQACONVA RQACONVA RQACONV RQACONV"


DATASET_DIRS=(${WENETEM_DIR})
AUDIO_IN_EXT=(tsv)
TEXT_IN_EXT=(wrdemo)
TEXT_OUT_EXT=(wrdemo)
CODEC_OUT_EXT=("<NONE>")

. $WORK_DIR/run_scripts/parse_data_dir.sh

NEGATIVE_TSVS="$NEGA_DIR/train_sub44k.tsv $NOISE_DIR/train.tsv"
NEGATIVE_RATIO=0.70

# unset CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$WORK_DIR
TRAINING_SCRPT=src/scripts/train_puremoe.py
NODE_NUM=1
INDEX=0
GPU_NUM_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29501
DISTRIBUTED_ARGS="
    --nnodes=$NODE_NUM \
    --node_rank=$INDEX \
    --nproc_per_node $GPU_NUM_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"
if [[ -z $DISTRIBUTED_ARGS ]]; then
	LAUNCH_CMD="deepspeed --include localhost:0,1,2,3,4,5,6,7 $TRAINING_SCRPT"
	#LAUNCH_CMD="deepspeed --include localhost:0 $TRAINING_SCRPT"
else
	LAUNCH_CMD="torchrun $DISTRIBUTED_ARGS $TRAINING_SCRPT"
fi
$LAUNCH_CMD \
	--deepspeed config/ds_z2_offload_config.json \
    --model_type "vita-deepseek_v2" \
	--initialize_additional_modules False \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder $AUDIO_ENCODER \
	--audio_projector_type "linear" \
    --freeze_backbone False \
    --freeze_audio_encoder_adapter True \
    --freeze_audio_encoder True \
    --freeze_tts_adapter False \
    --freeze_embed_tokens False \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --add_codec_target True \
	--num_train_epochs 1 \
    --load_best_model_at_end True \
    --save_steps 400 \
    --save_total_limit 3 \
    --eval_strategy "steps" \
    --eval_steps 400 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 25 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 True \
    --model_hidden_size 2048 \
    --audio_encoder_hidden_size 1024 \
    --audio_projector_hidden_size 7168 \
    --audio_num_codebook 7 \
    --text_vocab_size 102400 \
    --text_special_tokens 64 \
    --audio_vocab_size 4096 \
    --audio_special_tokens 64 \
    --data_jsons $DATA_JSONS \
    --eval_data_jsons $EVAL_DATA_JSONS \
	--negative_tsvs $NEGATIVE_TSVS \
	--negative_ratio $NEGATIVE_RATIO \
    --text_additional "EOT" "PAD_T" "BOT" "ANS_T" "TTS" "TQA" "TQAA" \
    --audio_additional "EOA" "PAD_A" "BOA" "ANS_A" "ASR" "AQA" "AQAA" "M29" "F10" "ER" \
    --asr_template /mnt/data/hetinggao/manifest/asr_prompts/asr_template.json \
    --tasks ${TASKS} \
    --output_dir ${OUTPUT_DIR} \
    --sample_rate 16000 \
    --audio_feature_rate 50 \
    --dataloader_num_workers 2 \
    --remove_unused_columns False \
    --max_keep_sample_size $((25*16000)) \
	--tune_text_embed True \
	--tie_word_embeddings True \
	--loss_reduction mean \
	--max_input_length 1000 \
	--use_last_turn_if_codec True \
	--emotion_token_as_text False \

    
unused="""
	--data_ratio $DATA_RATIO \
    --audio_in ${AUDIO_IN} \
    --text_in ${TEXT_IN} \
    --text_out ${TEXT_OUT} \
    --codec_out ${CODEC_OUT} \
    --eval_audio_in ${EVAL_AUDIO_IN} \
    --eval_text_in ${EVAL_TEXT_IN} \
    --eval_text_out ${EVAL_TEXT_OUT} \
    --eval_codec_out ${EVAL_CODEC_OUT} \
	--tokenizer_name_or_path $TOKENIZER_NAME_OR_PATH \
"""
