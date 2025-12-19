# lmms-finetune (two-stage fine-tuning workflow)

## 1. Project Overview

本リポジトリは、Vision-Language Model（VLM）のファインチューニングを行うための軽量なコードベースです。
モデル読み込み・データコレーション等の構成要素を抽象化し、Hugging Face の公式実装に沿った形で学習・推論を進められることを目的としています。

基盤モデルとして **LLaVA-OneVision** を使用しており、`train.py`（stage1）に加えて、`train_add_head*.py` など **2段階学習（stage1 → stage2）** を想定したスクリプト群が含まれています。

## 2. Key Features

- LLaVA-OneVision ベースの VLM ファインチューニングフレームワーク
- VLM の学習を進めるための最小構成（モデル・データ・学習処理の分離）
- Hugging Face 実装ベースのため、学習後も HF 流儀で推論/運用がしやすい
- DeepSpeed 設定（`ds_configs/`）や補助モジュール（`collators/`, `loaders/`, `models/` 等）を同梱
- 2段階学習用の学習スクリプト（`train.py` / `train_add_head*.py`）が存在
- 推論・ユーティリティ（`infer.py`, `merge_lora_weights.py`, `webui.py` 等）

## 3. Directory Structure
```
.
├── collators/                 # データコレーター
├── loaders/                   # データ読み込み関連
├── models/                    # モデル関連
├── ds_configs/                # DeepSpeed 設定
├── scripts/                   # スクリプト類
├── example_scripts/           # 実行例スクリプト
├── example_data/              # サンプルデータ
├── jsons/                     # JSON 設定/テンプレート
├── docs/                      # ドキュメント
├── results/                   # 実験結果（フォルダが存在）
├── train.py                   # 学習（stage1）
├── train_add_head.py          # 学習（stage2 系）
├── train_add_head_alltoken.py # 学習（stage2 亜種）
├── infer.py                   # 推論
├── webui.py                   # Web UI
├── merge_lora_weights.py      # LoRA 重みマージ
├── requirements.txt
└── supported_models.py        # 対応モデル確認
```

## 4. Installation

一般的なセットアップ例です（環境に合わせて調整してください）。
```bash
git clone https://github.com/otake-code/lmms-finetune
cd lmms-finetune

conda create -n lmms-finetune python=3.10 -y
conda activate lmms-finetune

python -m pip install -r requirements.txt

# 任意: flash-attn（環境により導入方法が異なるため、必要に応じて）
python -m pip install --no-cache-dir --no-build-isolation flash-attn
```

**動作環境:**
- Python 3.10
- CUDA 対応 GPU（推奨）
- PyTorch 2.0 以上

## 5. Usage

### 5.1 Supported models

対応モデル一覧は `supported_models.py` から確認できます。
```bash
python supported_models.py
```

本リポジトリは LLaVA-OneVision を基盤としていますが、他の VLM にも対応可能な設計となっています。

### 5.2 Two-stage training

本フォークでは、概ね以下の流れを想定します。

* **Stage 1:** 基本の学習（`train.py`）- LLaVA-OneVision の基本的なファインチューニング
* **Stage 2:** 追加ヘッド/追加トークン等を含む学習（`train_add_head*.py`）

#### Stage 1
```bash
python train.py \
  --config <YOUR_CONFIG_OR_ARGS>
```

#### Stage 2
```bash
python train_add_head.py \
  --config <YOUR_CONFIG_OR_ARGS>
```

（必要に応じて `train_add_head_alltoken.py` 等のバリエーションを利用）

### 5.3 Inference
```bash
python infer.py \
  --config <YOUR_CONFIG_OR_ARGS>
```

### 5.4 Web UI (optional)
```bash
python webui.py
```

### 5.5 Merge LoRA weights (optional)
```bash
python merge_lora_weights.py \
  --base_model <BASE_MODEL_PATH> \
  --lora_path <LORA_PATH> \
  --output_path <OUTPUT_MODEL_PATH>
```

## 6. Model Architecture

本手法は LLaVA-OneVision を基盤として以下の構成要素から成ります:

1. **Vision Encoder**: 画像特徴の抽出
2. **Language Model**: テキスト理解と推論
3. **Vision-Language Projector**: マルチモーダル情報の統合
4. **Custom Head (Stage 2)**: タスク特化の追加ヘッド

## 7. Notes

* `example_scripts/` や `example_data/` が同梱されているため、最初はそれらを起点に引数やデータ形式を合わせる形が分かりやすいです。
* DeepSpeed を使う場合、`ds_configs/` を参照して実行環境に合わせた設定にできます。
* 本リポジトリは複数モデル/学習方式に対応する設計のため、実験設定（データ形式・学習引数・チェックポイント出力）を固定して再現性を確保する運用が向いています。
* LLaVA-OneVision の事前学習済みモデルは Hugging Face Hub から取得できます。

## 8. Acknowledgments

本リポジトリは以下のプロジェクトに基づいています:

- [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## 9. License

本プロジェクトのライセンスについては、`LICENSE` ファイルを参照してください。使用する基盤モデル（LLaVA-OneVision等）のライセンスにも従ってください。
