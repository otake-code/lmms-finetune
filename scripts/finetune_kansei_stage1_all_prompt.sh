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

# ランダム系列のデータパスを設定
TRAIN_DATA_PATHS=(
  jsons/train0_yesno_random_wo_yn.jsonl
  jsons/train0_yesno_random_add_reason.jsonl
)

# 出力先ベースディレクトリ
CHECKPOINT_BASE=./checkpoints/kansei/yesno_fire_ve

# その他のハイパーパラメータ
BATCH_SIZES=(1)
LEARNING_RATES=(7e-6 5e-6 3e-6)
EPOCHS=(100)
GRAD_ACCUM=2
SAVE_EPOCHS=10000  # 保存間隔＝10エポック分のステップ数

# 固定設定
IMAGE_FOLDER=/home/okada/llama3_feature/grain_dataset
TRAIN_VISION_ENCODER=True
USE_VISION_LORA=False
TRAIN_VISION_PROJECTOR=True
DS_STAGE=zero3
MODEL_MAX_LEN=1024

for TRAIN_DATA_PATH in "${TRAIN_DATA_PATHS[@]}"; do
  # 全サンプル数を Python でカウント
  TOTAL_EXAMPLES=$(python3 - <<EOF
print(sum(1 for _ in open("${TRAIN_DATA_PATH}", "r")))
EOF
)
  # サブフォルダ名を抽出
  # 例: train0_yesno_random_reason.jsonl → reason
  filename=$(basename "$TRAIN_DATA_PATH")
  # 「random_」以降を取り出す → reason.jsonl
  suffix_with_ext=${filename#*random_}
  # 「.jsonl」を削除 → reason
  suffix=${suffix_with_ext%.jsonl}


  for BS in "${BATCH_SIZES[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
      for EP in "${EPOCHS[@]}"; do
        # ステップ数計算
        STEPS_PER_EPOCH=$(python3 - <<EOF
import math
total = ${TOTAL_EXAMPLES}
bs = ${BS}
gpus = ${NUM_GPUS}
acc = ${GRAD_ACCUM}
print(math.ceil(total / (bs * gpus * acc)))
EOF
)
        SAVE_STEPS=$(python3 - <<EOF
print(${STEPS_PER_EPOCH} * ${SAVE_EPOCHS})
EOF
)

        DATE=$(date '+%Y-%m-%dT%H_%M_%S')
        GLOBAL_BS=$((BS * NUM_GPUS * GRAD_ACCUM))
        # RUN_ID に "random" タグを含める
        RUN_ID="gbs${GLOBAL_BS}_lr${LR}_ep${EP}_random_ve_kansei_${DATE}"

        torchrun $DISTRIBUTED_ARGS train_add_head_alltoken.py \
          --model_id $MODEL_ID \
          --model_local_path $MODEL_LOCAL_PATH \
          --data_path $TRAIN_DATA_PATH \
          --image_folder $IMAGE_FOLDER \
          --output_dir ${CHECKPOINT_BASE}/${suffix}/${RUN_ID} \
          --run_name $RUN_ID \
          --deepspeed ./ds_configs/${DS_STAGE}.json \
          --bf16 False \
          --fp16 True \
          --num_train_epochs $EP \
          --per_device_train_batch_size $BS \
          --per_device_eval_batch_size $BS \
          --gradient_accumulation_steps $GRAD_ACCUM \
          --save_strategy steps \
          --save_steps $SAVE_STEPS \
          --save_total_limit 10 \
          --learning_rate ${LR} \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 1 \
          --tf32 True \
          --model_max_length $MODEL_MAX_LEN \
          --gradient_checkpointing True \
          --dataloader_num_workers 4 \
          --train_vision_encoder $TRAIN_VISION_ENCODER \
          --use_vision_lora $USE_VISION_LORA \
          --train_vision_projector $TRAIN_VISION_PROJECTOR \
          --stage1 True

      done
    done
  done
done
