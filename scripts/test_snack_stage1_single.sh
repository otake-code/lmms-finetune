#!/bin/bash
set -e

# 共通設定
BASE_MODEL="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
TEST_JSON="/home/okada/vlm/lmms-finetune/jsons/snack/test_snack_yesno.jsonl"
IMAGE_FOLDER="/data01/snacks/test"
OUTPUT_BASE="results/snack/yesno"
BATCH_SIZE=1
MAX_LENGTH=2048
DEVICE="cuda"

# ここだけを指定
SPECIFIC_RUN_DIR="/home/okada/vlm/lmms-finetune/checkpoints/snack/yesno/gbs128_lr5e-5_ep100_snack2025-07-27T21_31_00"
SPECIFIC_CKPT="checkpoint-3200"

MODEL_DIR="${SPECIFIC_RUN_DIR}/${SPECIFIC_CKPT}"
OUTPUT_DIR="${OUTPUT_BASE}/bs1/lr1e-5/${SPECIFIC_CKPT}"
mkdir -p "${OUTPUT_DIR}"

echo ">>> 推論: ${MODEL_DIR}"
python test_stage1_inference_alltoken.py \
  --base_model   "${BASE_MODEL}" \
  --model_dir    "${MODEL_DIR}" \
  --jsonl_path   "${TEST_JSON}" \
  --image_folder "${IMAGE_FOLDER}" \
  --batch_size   "${BATCH_SIZE}" \
  --max_length   "${MAX_LENGTH}" \
  --device       "${DEVICE}" \
  --output_csv   "${OUTPUT_DIR}/anomaly_scores.csv"

echo "→ 保存先: ${OUTPUT_DIR}/anomaly_scores.csv"
echo "完了しました。"
