#!/bin/bash
set -e

# モデルチェックポイントの親ディレクトリ
MODEL_BASES=(
  "/home/okada/vlm/lmms-finetune/checkpoints/kansei/yesno_fire_ve/add_reason"
  "/home/okada/vlm/lmms-finetune/checkpoints/kansei/yesno_fire_ve/wo_yn"
)

# 推論用 JSONL
TEST_JSON="/home/okada/vlm/lmms-finetune/jsons/test0_finetune_data_onevision.jsonl"

# 画像フォルダ
IMAGE_FOLDER="/home/okada/iad/LLaVA-NeXT/images"

# ベースモデル ID
BASE_MODEL="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

# 出力先親ディレクトリ
OUTPUT_BASE="results/kansei/stage1_fire_ve"

# 推論パラメータ
BATCH_SIZE=1
MAX_LENGTH=1024
DEVICE="cuda"

for MODEL_BASE in "${MODEL_BASES[@]}"; do
  for RUN_DIR in "${MODEL_BASE}"/*/; do
    for MODEL_DIR in "${RUN_DIR}"checkpoint-*; do
      # checkpoints/kansei/yesno_fire_ve/ 以降を相対パスとして取得
      REL_PATH="${MODEL_DIR#/home/okada/vlm/lmms-finetune/checkpoints/kansei/yesno_fire_ve/}"
      OUTPUT_DIR="${OUTPUT_BASE}/${REL_PATH}"
      mkdir -p "${OUTPUT_DIR}"

      echo ">>> 推論: ${REL_PATH}"
      python test_stage1_inference_alltoken.py \
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
done
echo "全ての推論が完了しました。"
