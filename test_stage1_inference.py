# test_stage1_inference.py

import os
import argparse
import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig
from torch.nn.functional import sigmoid
from importlib.metadata import version, PackageNotFoundError

from models.custom_llava_onevision import LlavaOnevisionForYesNo
from train_add_head import GrainYesNoDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",    type=str, required=True)
    p.add_argument("--jsonl_path",   type=str, required=True)
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--max_length",   type=int, default=512)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--output_csv",   type=str, default="anomaly_scores.csv")
    return p.parse_args()


def get_bnb_config():
    """
    bitsandbytes がインストールされていれば 4bit quantization、
    なければ load_in_4bit=False（FP16ロード）を返す。
    """
    try:
        _ = version("bitsandbytes")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    except PackageNotFoundError:
        print("⚠ bitsandbytes が見つからなかったため、4bit を無効化します。")
        return BitsAndBytesConfig(load_in_4bit=False)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 量子化設定を取得
    bnb_config = get_bnb_config()
    use_4bit = bnb_config.load_in_4bit

    # ── モデルロード ──
    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "ignore_mismatched_sizes": True,
    }
    if use_4bit:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = LlavaOnevisionForYesNo.from_pretrained(
        args.model_dir,
        **load_kwargs
    ).to(device).eval()

    # ── プロセッサロード ──
    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        local_files_only=True,
    )

    # ── データセット＋DataLoader ──
    test_ds = GrainYesNoDataset(
        jsonl_path=args.jsonl_path,
        image_folder=args.image_folder,
        processor=processor,
        max_length=args.max_length,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ── 推論ループ ──
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            # pixel_values はモデル dtype に合わせてキャスト
            pix = batch["pixel_values"].to(device)
            if not use_4bit:
                pix = pix.half()
            inp = batch["input_ids"].to(device)
            img_sz = batch["image_sizes"].to(device)

            outs = model(
                pixel_values=pix,
                input_ids=inp,
                image_sizes=img_sz,
            )
            probs = sigmoid(outs.logits).cpu().tolist()

            for lab, score in zip(batch["labels"], probs):
                results.append({
                    "label": int(lab.item()),
                    "anomaly_score": score,
                })

    # ── CSV出力 ──
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "anomaly_score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved anomaly scores to {args.output_csv}")


if __name__ == "__main__":
    main()
