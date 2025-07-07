# test_llavaov_dynamic_with_txt.py

import os
import json
import torch
import argparse
import logging
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
import transformers

# Hugging Face の警告を抑制
transformers.logging.set_verbosity_error()

########################
# tqdm と共存するログハンドラ
########################
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger():
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

########################
# モデルとプロセッサのロード関数（単一GPU対応）
########################
def load_model(model_name_or_path, device):
    """
    LlavaOnevisionForConditionalGeneration と AutoProcessor をロードし、
    単一の GPU/CPU に配置して返す。
    """
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map=None
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    return model, processor

########################
# 画像読み込みユーティリティ
########################
def load_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.warning(f"Failed to load image: {image_path} -> {e}")
        return None

########################
# Dynamic Chat 用プロンプト構築
########################
def build_prompt(question: str):
    """
    Dynamic Chat テンプレート用に、ユーザーからの質問文だけを組み立てる。
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"}
            ]
        }
    ]
    return conversation

########################
# 推論実行関数
########################
def run_inference(model, processor, image, question: str, max_tokens: int = 64):
    """
    Dynamic Chat テンプレートで推論を行い、「Yes/No + 根拠」を返す。
    """
    conversation = build_prompt(question)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, padding=True, return_tensors="pt").to(model.device, torch.float16)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    gen_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    answer = processor.decode(gen_tokens, skip_special_tokens=True).strip()
    return answer

########################
# メイン処理
########################
def main():
    parser = argparse.ArgumentParser(description='LLaVA-OV 推論スクリプト（テキスト出力対応版）')
    parser.add_argument(
        '--test_data_path',
        type=str,
        required=True,
        help='テストデータ (JSON) のパス'
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help='Hugging Face 形式の LLaVA-OV モデル名またはローカルディレクトリパス'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='推論結果の出力先ディレクトリパス'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=64,
        help='生成時の最大トークン数'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='推論に使用するデバイス (例: cuda or cpu)'
    )
    args = parser.parse_args()

    setup_logger()
    logging.info("=== LLaVA-OV 推論開始（テキスト出力対応版） ===")

    # 出力ディレクトリを絶対パス化して作成
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # テスト JSON ファイルを読み込み
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # モデルとプロセッサをロード
    model, processor = load_model(args.model_name_or_path, args.device)

    correct_count = 0
    total_count = len(test_data)

    # 出力ファイルパス
    output_txt = os.path.join(output_dir, "output.txt")
    output_acc = os.path.join(output_dir, "accuracy.txt")

    with open(output_txt, "w", encoding="utf-8") as fout_txt:
        for example in tqdm(test_data, desc="推論中", unit="case"):
            img_path = example["image"]
            image = load_image(img_path)
            if image is None:
                continue

            # human の質問文と gpt の正解を取得
            question = example["conversations"][0]["value"].strip()
            correct_answer = example["conversations"][1]["value"].strip()

            # 推論を実行
            pred_answer = run_inference(
                model=model,
                processor=processor,
                image=image,
                question=question,
                max_tokens=args.max_tokens
            )

            # 出力テキスト内の改行をすべて空白に置き換えて一行にまとめる
            pred_answer_single_line = pred_answer.replace("\n", " ").strip()

            # Yes/No 部分で正誤判定
            pred_prefix = pred_answer_single_line.split(",")[0].strip().lower()
            correct_prefix = correct_answer.split(",")[0].strip().lower()
            if pred_prefix == correct_prefix:
                correct_count += 1

            # output.txt に書き込む
            fout_txt.write(f"id:      {example.get('id', '')}\n")
            fout_txt.write(f"image:   {img_path}\n")
            fout_txt.write(f"prompt:  {question}\n")
            fout_txt.write(f"correct: {correct_answer}\n")
            fout_txt.write(f"output:  {pred_answer_single_line}\n")
            fout_txt.write("-" * 90 + "\n")

            # コンソールにも logging で出力
            logging.info(f"id:      {example.get('id', '')}")
            logging.info(f"image:   {img_path}")
            logging.info(f"prompt:  {question}")
            logging.info(f"correct: {correct_answer}")
            logging.info(f"output:  {pred_answer_single_line}")
            logging.info("-" * 60)

    # accuracy.txt に精度を書き込む
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    with open(output_acc, "w", encoding="utf-8") as fout_acc:
        fout_acc.write(f"Total cases: {total_count}\n")
        fout_acc.write(f"Correct   : {correct_count}\n")
        fout_acc.write(f"Accuracy  : {accuracy:.4f}\n")

    logging.info(f"=== 推論終了: output.txt と accuracy.txt を {output_dir} に保存しました ===")

if __name__ == "__main__":
    main()
