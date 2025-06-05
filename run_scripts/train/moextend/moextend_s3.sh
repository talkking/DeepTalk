#!/bin/bash
WORK_DIR=$(pwd)

ME=$(basename "$0")
ME=${ME%.*}
TIMESTAMP=$(date '+%m%d%y-%H%M%S')

OUTPUT_DIR=${WORK_DIR}/outputs/${ME}

CACHE_DIR=/mnt/data/hetinggao/models
MODEL_NAME_OR_PATH=/mnt/data/alanhshao/vita-e2e/outputs/lucy_deepseek-addExperts_s2p2
#/mnt/data/alanhshao/vita-e2e/outputs/lucy_qwen2moe-chat_s2p2
#"/mnt/data/alanhshao/vita-e2e/outputs/vita_qwen2moe-chat_s1/checkpoint-107600"
#"/mnt/data/alanhshao/vita-e2e/outputs/vita_qwen2moe-chat_s1/checkpoint-33200"
AUDIO_ENCODER="/mnt/data/hetinggao/models/whisper-medium"
#AUDIO_ENCODER=/mnt/data/hetinggao/models/audio-encoder-Qwen2-7B-instruct-weight-base-11wh-tunning

UNION60W_DIR="/mnt/data/hetinggao/manifest/text/6w"
COMMON_DIR="/mnt/data/hetinggao/manifest/AudioQA/jsons/AudioQA-1450K-filtered-v4"
FC_DIR="/mnt/data/hetinggao/manifest/vita2tts_v4/jsons/fc"
FC1_DIR="/mnt/data/hetinggao/manifest/vita2tts_v4/jsons/fcneg241206"
FC2_DIR="/mnt/data/hetinggao/manifest/vita2tts_v4/jsons/fcneg241218"
FC3_DIR="/mnt/data/hetinggao/manifest/vita2tts_v4/jsons/fc_241212"
FC4_DIR="/mnt/data/hetinggao/manifest/vita2tts_v4/jsons/fc_241218"
NATURAL1_DIR="/mnt/data/hetinggao/manifest/vita2tts_v4/jsons/audioagent3"
NATURAL2_DIR="/mnt/data/hetinggao/manifest/vita2tts_v4/jsons/audioagent4"
EMO1_DIR="/mnt/data/hetinggao/manifest/Emotion_Control/stage3/instruction"
EMO2_DIR="/mnt/data/hetinggao/manifest/Emotion_Control/stage3/dialog"
EMO3_DIR="/mnt/data/hetinggao/manifest/Emotion_Control/stage3/normal1227"
EMO4_DIR="/mnt/data/hetinggao/manifest/Emotion_Control/stage3/abnormal1227"
NUM_DIR="/mnt/data/hetinggao/manifest/num_tts/jsons/num_tts"
ID_DIR="/mnt/data/hetinggao/manifest/Identity/jsons_v2"
NEGA_DIR="/mnt/data/hetinggao/manifest/vita_negative"
NOISE_DIR="/mnt/data/hetinggao/manifest/noise_negative"

# DATA_JSONS="$NATURAL1_DIR/train.json $NATURAL1_DIR/train.json $NATURAL2_DIR/train.json $NATURAL2_DIR/train.json $FC_DIR/train.json $FC_DIR/train.json $FC1_DIR/train.json $FC1_DIR/train.json $FC2_DIR/train.json $FC2_DIR/train.json $FC3_DIR/train.json $FC3_DIR/train.json $FC4_DIR/train.json $FC4_DIR/train.json $COMMON_DIR/train.json $COMMON_DIR/train.json $UNION60W_DIR/train.json $EMO1_DIR/train.json $EMO1_DIR/train.json $EMO2_DIR/train.json $EMO2_DIR/train.json $EMO3_DIR/train.json $EMO3_DIR/train.json $EMO4_DIR/train.json $EMO4_DIR/train.json $ID_DIR/train.json $ID_DIR/train.json"
# EVAL_DATA_JSONS="$NATURAL1_DIR/eval.json $NATURAL1_DIR/eval.json $NATURAL2_DIR/eval.json $NATURAL2_DIR/eval.json $FC_DIR/eval.json $FC_DIR/eval.json $FC1_DIR/eval.json $FC1_DIR/eval.json $FC2_DIR/eval.json $FC2_DIR/eval.json $FC3_DIR/eval.json $FC3_DIR/eval.json $FC4_DIR/eval.json $FC4_DIR/eval.json $COMMON_DIR/eval.json $COMMON_DIR/eval.json $UNION60W_DIR/eval.json $EMO1_DIR/eval.json $EMO1_DIR/eval.json $EMO2_DIR/eval.json $EMO2_DIR/eval.json $EMO3_DIR/eval.json $EMO3_DIR/eval.json $EMO4_DIR/eval.json $EMO4_DIR/eval.json $ID_DIR/eval.json $ID_DIR/eval.json"
# TASKS="RQACONVA_NTRL RQACONV_NTRL RQACONVA_NTRL RQACONV_NTRL RQACONVA RQACONV RQACONVA RQACONV RQACONVA RQACONV RQACONVA RQACONV RQACONVA RQACONV RQACONVA RQACONV RQACONV RQACONV_EMO RQACONVA_EMO RQACONV_EMO RQACONVA_EMO AQACONV_EMO AQACONVA_EMO AQACONV_EMO AQACONVA_EMO RQACONV RQACONVA"

DATA_JSONS="$COMMON_DIR/train.json $COMMON_DIR/train.json $COMMON_DIR/train.json $UNION60W_DIR/train.json $ID_DIR/train.json $ID_DIR/train.json $NUM_DIR/train.json $NUM_DIR/train.json"
#DATA_JSONS="$COMMON_DIR/train.json $UNION60W_DIR/train.json $ID_DIR/train.json $NUM_DIR/train.json"
EVAL_DATA_JSONS="$COMMON_DIR/eval.json $COMMON_DIR/eval.json $COMMON_DIR/eval.json $UNION60W_DIR/eval.json $ID_DIR/eval.json $ID_DIR/eval.json $NUM_DIR/eval.json $NUM_DIR/eval.json"
#EVAL_DATA_JSONS="$COMMON_DIR/eval.json $UNION60W_DIR/eval.json $ID_DIR/eval.json $NUM_DIR/eval.json"

DATA_JSONS="$COMMON_DIR/train.json $COMMON_DIR/train.json $COMMON_DIR/train.json $UNION60W_DIR/train.json"
EVAL_DATA_JSONS="$COMMON_DIR/eval.json $COMMON_DIR/eval.json $COMMON_DIR/eval.json $UNION60W_DIR/eval.json"

#TASKS="AQACONVA RQACONVA RQACONV RQACONV RQACONV RQACONVA RQACONV RQACONVA"
TASKS="AQACONVA RQACONVA RQACONV RQACONV"
. $WORK_DIR/run_scripts/parse_data_dir.sh

NEGATIVE_TSVS="$NEGA_DIR/train.tsv $NOISE_DIR/train.tsv"
NEGATIVE_RATIO=0.25

unset CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$WORK_DIR
TRAINING_SCRPT=src/scripts/train_moextend.py
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
#DISTRIBUTED_ARGS=
if [[ -z $DISTRIBUTED_ARGS ]]; then
	LAUNCH_CMD="deepspeed --include localhost:0,1,2,3,4,5,6,7 $TRAINING_SCRPT"
	#LAUNCH_CMD="deepspeed --include localhost:0 $TRAINING_SCRPT"
else
	LAUNCH_CMD="torchrun $DISTRIBUTED_ARGS $TRAINING_SCRPT"
fi
$LAUNCH_CMD \
	--deepspeed config/ds_z2_offload_config.json\
    --model_type "lucy-deepseek_v2" \
	--initialize_additional_modules False \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_encoder $AUDIO_ENCODER \
	--audio_projector_type "linear" \
    --freeze_backbone False \
    --freeze_audio_encoder_adapter True \
    --freeze_audio_encoder True \
    --freeze_tts_adapter False \
    --freeze_embed_tokens False \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --add_codec_target True \
	--num_train_epochs 2 \
    --load_best_model_at_end True \
    --save_steps 400 \
    --save_total_limit 3 \
    --eval_strategy "steps" \
    --eval_steps 400 \
    --learning_rate 5e-5 \
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
"""
