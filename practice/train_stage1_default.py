# test_stage1.py
#サンプルデータでお試しの学習

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from models.custom_llava_onevision import LlavaOnevisionForYesNo
from PIL import Image
import numpy as np

class DummyStage1Dataset(Dataset):
    """ランダムな画像とラベルを返すダミーデータセット"""
    def __init__(self, processor, num_samples=100):
        self.processor = processor
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # ランダムなRGB画像を生成
        arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr)

        # text＋images を一括トークナイズし、image_sizes も取得
        enc = self.processor(
            text=[""],            # ダミーの空テキスト
            images=[img],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=8          # テキスト長は短めでOK
        )
        pixel_values = enc["pixel_values"][0]   # (3, H, W)
        input_ids     = enc["input_ids"][0]      # (seq_len,)
        image_sizes   = enc["image_sizes"][0]    # (2,) height, width

        # ダミーラベル (0/1 を交互)
        label = torch.tensor(idx % 2, dtype=torch.float)

        return {
            "pixel_values": pixel_values,
            "input_ids":    input_ids,
            "image_sizes":  image_sizes,
            "label":        label
        }

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])      # (B,3,H,W)
    input_ids    = torch.stack([b["input_ids"]    for b in batch])      # (B,seq_len)
    image_sizes  = torch.stack([b["image_sizes"]  for b in batch])      # (B,2)
    labels       = torch.stack([b["label"]        for b in batch])      # (B,)

    return {
        "pixel_values": pixel_values,
        "input_ids":    input_ids,
        "image_sizes":  image_sizes,
        "labels":       labels
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # モデル＆プロセッサのロード
    model = LlavaOnevisionForYesNo.from_pretrained(
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    )
    processor = AutoProcessor.from_pretrained(
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    )
    model.to(device)

    # 分類ヘッド以外を凍結
    for n, p in model.named_parameters():
        if "classifier" not in n:
            p.requires_grad = False

    # ダミーデータ用 DataLoader
    dataset = DummyStage1Dataset(processor, num_samples=200)
    loader  = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    # オプティマイザ
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    # トレーニングループ
    model.train()
    for epoch in range(3):
        total_loss = 0.0
        for batch in loader:
            pv, ids, isz, lbl = (
                batch["pixel_values"].to(device),
                batch["input_ids"].to(device),
                batch["image_sizes"].to(device),
                batch["labels"].to(device),
            )

            optimizer.zero_grad()
            out = model(
                pixel_values=pv,
                input_ids=ids,
                image_sizes=isz           # ← これが必須
            )
            probs = out["yesno_probs"].squeeze(-1)  # (B,)
            loss  = F.binary_cross_entropy(probs, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    main()
