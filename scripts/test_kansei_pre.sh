GPU_INDEX=0

MODEL_ID=llava-onevision-0.5b-ov
MODEL_LOCAL_PATH=/home/okada/vlm/lmms-finetune/checkpoints/kansei/ft
DATA_PATH="/home/okada/vlm/lmms-finetune/jsons/test0_yesno_word.json"
IMAGE_FOLDER=/home/okada/iad/LLaVA-NeXT/images
# 出力先親ディレクトリ
OUTPUT_BASE="results/snack/yesno"

PER_DEVICE_BATCH_SIZE=4
MODEL_MAX_LEN=32768

CUDA_VISIBLE_DEVICES=$GPU_INDEX \

for RUN_DIR in "${MODEL_BASE}"/*/; do
  RUN_NAME=$(basename "${RUN_DIR}")
  # RUN_NAME 例: bs1_lr1e-5_ep30_2025-07-08T16_35_58
  BS_PART="$(echo ${RUN_NAME} | cut -d'_' -f1)"   # bs1
  LR_PART="$(echo ${RUN_NAME} | cut -d'_' -f2)"   # lr1e-5

  for MODEL_DIR in "${RUN_DIR}"checkpoint-*; do
    CKPT_NAME=$(basename "${MODEL_DIR}")  # checkpoint-1, checkpoint-3, …

    # 出力先を bs1/lr1e-5/checkpoint-1/ 以下に置く
    OUTPUT_DIR="${OUTPUT_BASE}/${BS_PART}/${LR_PART}/${CKPT_NAME}"
    mkdir -p "${OUTPUT_DIR}"

    echo ">>> 推論: ${BS_PART} / ${LR_PART} / ${CKPT_NAME}"
    python infer.py \
        --model_id $MODEL_ID \
        --model_local_path $MODEL_LOCAL_PATH \
        --data_path $DATA_PATH \
        --image_folder $IMAGE_FOLDER \
        --bf16 \
        --per_device_infer_batch_size $PER_DEVICE_BATCH_SIZE \
        --model_max_length $MODEL_MAX_LEN \
        --dataloader_num_workers 4 \
        --infer_stop_steps 2 \
        --infer_checkpoint \
        --output_dir "${OUTPUT_DIR}"

    echo "   → 保存先: ${OUTPUT_DIR}/anomaly_scores.csv"
    echo
  done
done
echo "全ての推論が完了しました。"
