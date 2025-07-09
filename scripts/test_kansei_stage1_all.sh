#!/bin/bash
set -e

# モデルチェックポイントの親ディレクトリ
MODEL_BASE="/home/okada/vlm/lmms-finetune/checkpoints/kansei/yesno"

# 推論用 JSONL
TEST_JSON="/home/okada/vlm/lmms-finetune/jsons/test0_finetune_data_onevision.jsonl"

# 画像フォルダ
IMAGE_FOLDER="/home/okada/iad/LLaVA-NeXT/images"

# ベースモデル ID
BASE_MODEL="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

# 出力先親ディレクトリ
OUTPUT_BASE="results/kansei/stage1"

# 推論パラメータ
BATCH_SIZE=1
MAX_LENGTH=512
DEVICE="cuda"

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
    python test_stage1_inference.py \
      --base_model   "${BASE_MODEL}" \
      --model_dir    "${MODEL_DIR}" \
      --jsonl_path   "${TEST_JSON}" \
      --image_folder "${IMAGE_FOLDER}" \
      --batch_size   "${BATCH_SIZE}" \
      --max_length   "${MAX_LENGTH}" \
      --device       "${DEVICE}" \
      --output_csv   "${OUTPUT_DIR}/anomaly_scores.csv"

    echo "   → 保存先: ${OUTPUT_DIR}/anomaly_scores.csv"
    echo
  done
done
echo "全ての推論が完了しました。"
