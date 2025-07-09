# test_stage1_inference.py

import os
import argparse
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from safetensors.torch import load_file as safe_load_file
from models.custom_llava_onevision import LlavaOnevisionForYesNo
from test_add_head import GrainYesNoDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",   type=str, required=True,
                   help="HuggingFace 上のベースモデル ID")
    p.add_argument("--model_dir",    type=str, required=True,
                   help="Stage1 の出力ディレクトリ (.safetensors あり)")
    p.add_argument("--jsonl_path",   type=str, required=True,
                   help="推論用 JSONL ファイルのパス")
    p.add_argument("--image_folder", type=str, required=True,
                   help="画像ルートフォルダのパス")
    p.add_argument("--max_length",   type=int, default=512,
                   help="テキスト最大長")
    p.add_argument("--batch_size",   type=int, default=8,
                   help="バッチサイズ")
    p.add_argument("--device",       type=str, default="cuda",
                   help="使用デバイス")
    p.add_argument("--output_csv",   type=str, default="anomaly_scores.csv",
                   help="出力 CSV パス")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # モデルロード
    base = LlavaOnevisionForYesNo.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16,
    ).to(device)
    base.eval()

    # ヘッド部ロード
    weight_file = next((f for f in os.listdir(args.model_dir) if f.endswith(".safetensors")), None)
    if weight_file is None:
        raise FileNotFoundError(f".safetensors ファイルが見つかりません: {args.model_dir}")
    ckpt_path = os.path.join(args.model_dir, weight_file)
    ckpt = safe_load_file(ckpt_path, device="cpu")
    head_state = {k: v for k, v in ckpt.items()
                  if k.startswith("classifier.") or k.startswith("multi_modal_projector.")}
    base.load_state_dict(head_state, strict=False)

    # プロセッサロード
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # DataLoader 設定
    test_ds = GrainYesNoDataset(
        jsonl_path=args.jsonl_path,
        image_folder=args.image_folder,
        processor=processor,
        max_length=args.max_length,
    )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 推論 & CSV 出力
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "image_path", "label", "anomaly_score"])
        writer.writeheader()

        for batch in tqdm(loader, desc="Inference"):
            # 入力準備
            pix = batch["pixel_values"].to(device)
            if base.dtype == torch.float16:
                pix = pix.half()
            inp = batch["input_ids"].to(device)
            sz  = batch.get("image_sizes")
            if sz is not None:
                sz = sz.to(device)

            # 推論
            out = base(pixel_values=pix, input_ids=inp, image_sizes=sz)
            probs = out["yesno_probs"].squeeze(-1).cpu()

            # メタデータ取得
            sources = batch.get("source", [""] * len(probs))
            imgps   = batch.get("images", batch.get("image_path", [""] * len(probs)))
            labels_tensor = batch.get("labels")
            if labels_tensor is not None:
                labels = [float(x) for x in labels_tensor.cpu().tolist()]
            else:
                labels = [None] * len(probs)

            # 書き込み
            for src, imgp, lab, p in zip(sources, imgps, labels, probs):
                score = p.item()
                writer.writerow({
                    "source":       src,
                    "image_path":   imgp,
                    "label":        lab,
                    "anomaly_score": f"{score:.6f}"
                })
                print(f"anomaly_score: {score:.6f}")

if __name__ == "__main__":
    main()
