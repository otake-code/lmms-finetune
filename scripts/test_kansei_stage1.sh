#!/bin/bash

export PYTHONPATH=.

function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

# --- 設定セクション ---
# k-fold list
KF_LIST=(0)

# Stage1 で保存されたモデルディレクトリ（fold ごとにサブフォルダを分けておく）
MODEL_BASE="/home/okada/vlm/lmms-finetune/checkpoints/kansei/yesno/llava-ov_ft_kansei"
# ※ 例: ${MODEL_BASE}/kf0/checkpoint-5 という構成を想定

# テスト用 JSONL のプレフィックス
TEST_JSON_BASE="/home/okada/vlm/lmms-finetune/jsons"
# ※ 例: ${TEST_JSON_BASE}/test${kf}_finetune_data_oneqa.json

# 画像フォルダ
IMAGE_FOLDER="/home/okada/iad/LLaVA-NeXT/images"

# 推論パラメータ
BATCH_SIZE=1
MAX_LENGTH=512
DEVICE="cuda"

# 出力先ベース
OUTPUT_BASE="results/kansei/stage1"

# --- 推論ループ ---
for kf in "${KF_LIST[@]}"; do
  # fold ごとのパス
  # MODEL_DIR="${MODEL_BASE}/kf${kf}/checkpoint-5"
  MODEL_DIR="/home/okada/vlm/lmms-finetune/checkpoints/kansei/yesno/llava-ov_ft_kansei_/checkpoint-5/"
  TEST_JSON="${TEST_JSON_BASE}/test${kf}_finetune_data_onevision.jsonl"
  OUTPUT_DIR="${OUTPUT_BASE}_kf${kf}"
  mkdir -p "${OUTPUT_DIR}"

  echo "=== [kf=${kf}] 推論開始 ==="

  python test_stage1_inference.py \
    --model_dir     "${MODEL_DIR}" \
    --jsonl_path    "${TEST_JSON}" \
    --image_folder  "${IMAGE_FOLDER}" \
    --max_length    "${MAX_LENGTH}" \
    --batch_size    "${BATCH_SIZE}" \
    --device        "${DEVICE}" \
    --output_csv    "${OUTPUT_DIR}/anomaly_scores_kf${kf}.csv"

  echo "=== 結果 → ${OUTPUT_DIR}/anomaly_scores_kf${kf}.csv ==="
  echo
done
