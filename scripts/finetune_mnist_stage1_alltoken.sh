#!/bin/bash

NUM_GPUS=$(nvidia-smi -L | wc -l)
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

MODEL_ID=llava-onevision-0.5b-ov
MODEL_LOCAL_PATH=llava-hf/llava-onevision-qwen2-0.5b-ov-hf
TRAIN_DATA_PATH=/home/okada/vlm/lmms-finetune/jsons/mnist/mnist_train.jsonl
IMAGE_FOLDER=/dataset/mnist

# ハイパーパラメータのリスト
BATCH_SIZES=(1)
LEARNING_RATES=(1e-6 5e-6 1e-5 )
EPOCHS=(30)

# 固定の設定
TRAIN_VISION_ENCODER=False
USE_VISION_LORA=False
TRAIN_VISION_PROJECTOR=True
DS_STAGE=zero3
GRAD_ACCUM=32
MODEL_MAX_LEN=1024

for BS in "${BATCH_SIZES[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    for EP in "${EPOCHS[@]}"; do

      DATE=$(date '+%Y-%m-%dT%H_%M_%S')
      GLOBAL_BS=$((BS * NUM_GPUS * GRAD_ACCUM))
      RUN_ID="gbs${GLOBAL_BS}_lr${LR}_ep${EP}_alltoken_mnisit_$DATE"

      torchrun $DISTRIBUTED_ARGS train_add_head_alltoken.py \
        --model_id $MODEL_ID \
        --model_local_path $MODEL_LOCAL_PATH \
        --data_path $TRAIN_DATA_PATH \
        --image_folder $IMAGE_FOLDER \
        --output_dir ./checkpoints/mnist/alltoken/${RUN_ID} \
        --run_name $RUN_ID \
        --deepspeed ./ds_configs/${DS_STAGE}.json \
        --bf16 False \
        --fp16 True \
        --num_train_epochs $EP \
        --per_device_train_batch_size $BS \
        --per_device_eval_batch_size $BS \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --save_strategy "epoch" \
        --save_total_limit 3 \
        --learning_rate ${LR} \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length $MODEL_MAX_LEN \
        --gradient_checkpointing False \
        --dataloader_num_workers 4 \
        --train_vision_encoder $TRAIN_VISION_ENCODER \
        --use_vision_lora $USE_VISION_LORA \
        --train_vision_projector $TRAIN_VISION_PROJECTOR \
        --stage1 True

    done
  done
done
