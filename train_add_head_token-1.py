import os
os.environ["WANDB_PROJECT"] = "lmms-ft"

from dataclasses import asdict
import math
from pathlib import Path
import yaml

import torch
import transformers
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, deepspeed, AutoProcessor, default_data_collator
from torch.utils.data import Dataset

from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators import COLLATORS
from datasets import LazySupervisedDataset
from loaders import LOADERS
from supported_models import MODULE_KEYWORDS
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)

# ──────────────── Stage1 用 Dataset ────────────────
class GrainYesNoDataset(Dataset):
    """Stage1: JSONL + 画像 1 枚 + Yes/No ラベル"""
    def __init__(self, jsonl_path, image_folder, processor, max_length):
        self.processor = processor
        self.image_folder = image_folder
        self.max_length = max_length
        import json
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.records = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        from PIL import Image
        img = Image.open(os.path.join(self.image_folder, rec["images"][0])).convert("RGB")
        enc = self.processor(
            text=[rec["conversations"][0]],
            images=[img],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        ans = rec["conversations"][1].strip().lower()
        label = 1.0 if (ans.startswith("yes") or ans == "1") else 0.0
        return {
            "pixel_values": enc["pixel_values"][0],
            "input_ids":     enc["input_ids"][0],
            "image_sizes":   enc["image_sizes"][0],
            "labels":        torch.tensor(label, dtype=torch.float),
        }

# ──────────────────────────────────────────────────────

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # 引数ダンプ
    output_dir = training_args.output_dir
    assert output_dir, "output_dir is required"
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))

    # デバイスタイプ設定など（元からのロジック）
    compute_dtype = torch.float16 if training_args.fp16 else (
        torch.bfloat16 if training_args.bf16 else torch.float32
    )
    if training_args.deepspeed and lora_args.q_lora:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} \
            if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None
        if training_args.fsdp or deepspeed.is_deepspeed_zero3_enabled():
            raise ValueError("FSDP or ZeRO3 are not compatible with QLoRA.")

    bnb_config = None
    if lora_args.use_lora and lora_args.q_lora:
        from transformers import BitsAndBytesConfig
        rank0_print("Quantization for LLM enabled...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )

    # ──────────────── Stage1 分岐 ────────────────
    if training_args.stage1:
        rank0_print("=== Stage1: Yes/No + VisionProjector Training ===")
        # モデルとプロセッサロード
        from models.custom_llava_onevision_token_1 import LlavaOnevisionForYesNo
        model = LlavaOnevisionForYesNo.from_pretrained(
            model_args.model_local_path,
        )
        processor = AutoProcessor.from_pretrained(model_args.model_local_path)
        model.to(training_args.device)

        # Freeze all except classifier & vision_projector
        vision_proj_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
        for name, param in model.named_parameters():
            if ("classifier" in name) or any(name.startswith(k) for k in vision_proj_keys):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # データセットロード
        rank0_print("Loading Stage1 dataset...")
        train_dataset = GrainYesNoDataset(
            jsonl_path=data_args.data_path,
            image_folder=data_args.image_folder,
            processor=processor,
            max_length=training_args.model_max_length,
        )
        eval_dataset = None
        if data_args.eval_data_path:
            eval_dataset = GrainYesNoDataset(
                jsonl_path=data_args.eval_data_path,
                image_folder=data_args.image_folder,
                processor=processor,
                max_length=training_args.model_max_length,
            )

        # Trainer のセットアップ
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
        )
        trainer.train()
        trainer.save_model(output_dir)
        return
    # ──────────────── End Stage1 分岐 ────────────────

    # 以下、元のフルファインチューニングロジック
    rank0_print("Loading model, tokenizer, processor for full finetuning...")
    loader = LOADERS[model_args.model_family_id](
        model_hf_path=model_args.model_hf_path,
        model_local_path=model_args.model_local_path,
        compute_dtype=compute_dtype,
        bnb_config=bnb_config,
        use_flash_attn=training_args.use_flash_attn,
        device_map=device_map,
    )
    model, tokenizer, processor, config = loader.load()
    tokenizer.model_max_length = training_args.model_max_length
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # freeze vision encoder if needed
    vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]
    if not training_args.train_vision_encoder:
        rank0_print("Vision encoder is frozen.")
        for key in vision_encoder_keys:
            eval(f"model.{key}").requires_grad_(False)

    # freeze vision projector if needed
    vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
    if not training_args.train_vision_projector:
        rank0_print("Vision projector is frozen.")
        for key in vision_projector_keys:
            eval(f"model.{key}").requires_grad_(False)

    # freeze other components
    if "others" in MODULE_KEYWORDS[model_args.model_family_id]:
        rank0_print("Other multimodal components are frozen.")
        for key in MODULE_KEYWORDS[model_args.model_family_id]["others"]:
            eval(f"model.{key}").requires_grad_(False)

    # LoRA 部分（変更なし）
    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    # ... （以降、元の LoRA から TrainerWithCustomSampler まで同じ） ...

    # load data via LazySupervisedDataset
    rank0_print("Loading data for full finetuning...")
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        video_folder=data_args.video_folder,
        num_frames=data_args.num_frames,
        model_family_id=model_args.model_family_id,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key,
    )
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = LazySupervisedDataset(
            data_path=data_args.eval_data_path,
            image_folder=data_args.image_folder,
            video_folder=data_args.video_folder,
            num_frames=data_args.num_frames,
            model_family_id=model_args.model_family_id,
            user_key=data_args.user_key,
            assistant_key=data_args.assistant_key,
        )
    else:
        training_args.eval_strategy = "no"

    data_collator = COLLATORS[model_args.model_family_id](
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        mask_question_tokens=training_args.mask_question_tokens,
    )

    trainer = TrainerWithCustomSampler(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)


if __name__ == "__main__":
    train()
