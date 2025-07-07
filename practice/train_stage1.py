# train_stage1_jsonl.py
#木目画像でお試しの学習
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor
from models.custom_llava_onevision import LlavaOnevisionForYesNo

class GrainYesNoDataset(Dataset):
    """JSONL から wood-grain の Yes/No データを読み込む Dataset"""
    def __init__(self, jsonl_path: str, image_folder: str, processor: AutoProcessor):
        self.processor = processor
        self.image_folder = image_folder
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.records = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.image_folder, rec["images"][0])
        img = Image.open(img_path).convert("RGB")

        enc = self.processor(
            text=[rec["conversations"][0]],
            images=[img],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=16,
        )
        return {
            "pixel_values":   enc["pixel_values"][0],   # (3,H,W)
            "input_ids":      enc["input_ids"][0],      # (seq_len,)
            "attention_mask": enc["attention_mask"][0], # (seq_len,)
            "image_sizes":    enc["image_sizes"][0],    # (2,)
            "label":          torch.tensor(
                                  1.0 if rec["conversations"][1].strip().lower().startswith("yes") else 0.0,
                                  dtype=torch.float
                              ),
        }

def collate_fn(batch):
    return {
        "pixel_values":   torch.stack([b["pixel_values"]   for b in batch]),  # (B,3,H,W)
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),  # (B,seq_len)
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),  # (B,seq_len)
        "image_sizes":    torch.stack([b["image_sizes"]    for b in batch]),  # (B,2)
        "labels":         torch.stack([b["label"]          for b in batch]),  # (B,)
    }

def main():
    JSONL_PATH   = "jsons/train0_finetune_data_onevision.jsonl"
    IMAGE_FOLDER = "/home/okada/llama3_feature/grain_dataset"
    MODEL_NAME   = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    BATCH_SIZE   = 8
    LR           = 1e-4
    EPOCHS       = 3
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

    # モデル＆プロセッサのロード
    model     = LlavaOnevisionForYesNo.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    # 分類ヘッド＋ビジョンプロジェクター以外を凍結
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "classifier" in name or "vision_projector" in name:
            param.requires_grad = True

    # データセット＆データローダー
    dataset = GrainYesNoDataset(JSONL_PATH, IMAGE_FOLDER, processor)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # オプティマイザ
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch in loader:
            pv    = batch["pixel_values"].to(DEVICE)
            ids   = batch["input_ids"].to(DEVICE)
            amask = batch["attention_mask"].to(DEVICE)
            isz   = batch["image_sizes"].to(DEVICE)
            lbls  = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(
                pixel_values   = pv,
                input_ids      = ids,
                attention_mask = amask,
                image_sizes    = isz,

            )
            probs = outputs["yesno_probs"].squeeze(-1)  # (B,)
            loss  = F.mse_loss(probs, lbls)             # MSE のまま

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch}/{EPOCHS}] Avg Loss: {avg_loss:.4f}")

    # 保存
    model.save_pretrained("stage1_yesno_model")
    processor.save_pretrained("stage1_yesno_model")

if __name__ == "__main__":
    main()
