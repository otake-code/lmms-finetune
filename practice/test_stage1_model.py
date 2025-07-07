# evaluate_stage1_no_sklearn.py

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor
from models.custom_llava_onevision import LlavaOnevisionForYesNo
import matplotlib.pyplot as plt

class TestYesNoDataset(Dataset):
    def __init__(self, jsonl_path, image_folder, processor):
        self.processor = processor
        self.image_folder = image_folder
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.records = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(os.path.join(self.image_folder, rec["images"][0])).convert("RGB")
        enc = self.processor(
            text=[rec["conversations"][0]],
            images=[img],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=16,
        )
        label = 1.0 if rec["conversations"][1].strip().lower().startswith("yes") else 0.0
        return {
            "pixel_values": enc["pixel_values"][0],
            "input_ids":     enc["input_ids"][0],
            "image_sizes":   enc["image_sizes"][0],
            "label":         label
        }

def collate_fn(batch):
    pv = torch.stack([x["pixel_values"] for x in batch])
    ids= torch.stack([x["input_ids"]    for x in batch])
    sz= torch.stack([x["image_sizes"]  for x in batch])
    lbl = torch.tensor([x["label"] for x in batch], dtype=torch.float)
    return {"pixel_values": pv, "input_ids": ids, "image_sizes": sz, "labels": lbl}

def compute_roc(y_true, y_score):
    # ソートしてTP/FPを積み上げ
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    P = sum(y_true)
    N = len(y_true) - P
    tp = fp = 0
    tpr = [0.0]
    fpr = [0.0]
    prev_score = None
    for score, label in pairs:
        if label == 1.0:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P if P>0 else 0.0)
        fpr.append(fp / N if N>0 else 0.0)
    # AUC（台形法）
    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return fpr, tpr, auc

def main():
    JSONL_PATH   = "jsons/test0_finetune_data_onevision.jsonl"
    IMAGE_FOLDER = "/home/okada/llama3_feature/grain_dataset"
    MODEL_DIR    = "stage1_yesno_model"
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

    # モデル＆プロセッサロード
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model     = LlavaOnevisionForYesNo.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    # データローダー
    ds     = TestYesNoDataset(JSONL_PATH, IMAGE_FOLDER, processor)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    y_true = []
    y_score = []
    with torch.no_grad():
        for batch in loader:
            pv  = batch["pixel_values"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            sz  = batch["image_sizes"].to(DEVICE)
            out = model(pixel_values=pv, input_ids=ids, image_sizes=sz)
            probs = out["yesno_probs"].squeeze(-1).cpu().tolist()
            y_score.extend(probs)
            y_true.extend(batch["labels"].tolist())

    fpr, tpr, auc = compute_roc(y_true, y_score)

    # プロット
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Stage1 ROC Curve (no sklearn)")
    plt.legend(loc="lower right")
    out_path = "roc_curve.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Computed AUROC: {auc:.3f}")
    print(f"ROC curve saved to {out_path}")

if __name__ == "__main__":
    main()
