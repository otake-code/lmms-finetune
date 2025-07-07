#!/bin/bash

# LLaVA-OV の元の性能を確認するための推論スクリプト呼び出し例

export PYTHONPATH=.

function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

# k-fold（ここでは 0 のみ）
num_kf=("0")

# LLaVA-OV のモデル名 or ローカルパス
# 例: Hugging Face 上のモデルなら "llava-hf/llava-onevision-qwen2-7b-ov-hf"
#     ローカルにダウンロード済みなら "/home/okada/llava_models/llava-onevision-qwen2-7b-ov-hf"
LLAVA_MODEL="llava-hf/llava-onevision-qwen2-7b-ov-hf"
# LLAVA_MODEL="/home/okada/vlm/lmms-finetune/checkpoints/kansei/llava-ov_ft_kansei_2025-06-25T12_49_49/checkpoint-1"

for kf in ${num_kf[@]}
do
  # --- データパスの設定 ---
  # 事前に各画像あたり 1QA + 相対パス化済みの JSON を用意しておく
  TEST_DATA="/home/okada/iad/LLaVA-NeXT/jsons/kansei/test/test${kf}_finetune_data_oneqa.json"

  # --- 出力ディレクトリの設定 ---
  OUTPUT_BASE="results/kansei/baseline_kf${kf}"
  mkdir -p "${OUTPUT_BASE}"

  echo "=== LLaVA-OV 推論開始: kf=${kf} ==="
  echo "テストデータ: ${TEST_DATA}"
  echo "モデル   : ${LLAVA_MODEL}"
  echo "出力先   : ${OUTPUT_BASE}"

  torchrun --nproc_per_node=1 predict_kansei.py \
    --test_data_path    "${TEST_DATA}" \
    --model_name_or_path "${LLAVA_MODEL}" \
    --output_dir         "${OUTPUT_BASE}" \
    --device             "cuda"

  echo "=== 結果が ${OUTPUT_BASE} に保存されました ==="
  echo ""
done
