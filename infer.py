import os
import json
import datetime
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
from transformers import set_seed
import tqdm

from arguments import ModelArguments, DataArguments, InferArguments, LoraArguments, GenerationArguments
from datasets import LazySupervisedDataset
from collators import COLLATORS
from loaders import LOADERS
from utils import rank0_print


def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def train():
    # 引数パーサ
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, InferArguments, LoraArguments, GenerationArguments)
    )
    model_args, data_args, infer_args, lora_args, gen_args = parser.parse_args_into_dataclasses()

    # シード固定
    if infer_args.seed is not None:
        rank0_print(f"Setting seed to {infer_args.seed}...")
        set_seed(infer_args.seed)

    # 演算精度の決定
    if infer_args.fp16:
        compute_dtype = torch.float16
    elif infer_args.bf16:
        rank0_print("Warning: bf16 not supported for this model, falling back to float32")
        compute_dtype = torch.float32
    else:
        compute_dtype = torch.float32

    # データセット読み込み
    rank0_print("Loading data...")
    eval_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        model_family_id=model_args.model_family_id,
        image_folder=data_args.image_folder,
        video_folder=data_args.video_folder,
        num_frames=data_args.num_frames,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key,
    )

    # 生の JSON リストを保持
    raw_samples = eval_dataset.list_data_dict
    total = len(raw_samples)

    # チェックポイント（LoRA 未使用）
    checkpoint_dir = model_args.model_local_path
    rank0_print(f"Inference on checkpoint: {checkpoint_dir}")

    # # 出力ディレクトリ準備
    # ts = datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')
    # out_dir = os.path.join(
    #     checkpoint_dir,
    #     "eval_output",
    #     Path(data_args.data_path).stem,
    #     ts
    # )
    # os.makedirs(out_dir, exist_ok=True)
    # rank0_print(f"Output directory: {out_dir}")
        # 出力ディレクトリ準備（--output_dir で上書き可）
    if infer_args.output_dir:
        out_dir = infer_args.output_dir
    else:
        ts = datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')
        out_dir = os.path.join(
            checkpoint_dir,
            "eval_output",
            Path(data_args.data_path).stem,
            ts
        )
    os.makedirs(out_dir, exist_ok=True)
    rank0_print(f"Output directory: {out_dir}")


    # モデルロード
    loader = LOADERS[model_args.model_family_id](
        model_hf_path=model_args.model_hf_path,
        model_local_path=checkpoint_dir,
        compute_dtype=compute_dtype,
        use_flash_attn=infer_args.use_flash_attn,
        device_map=None,
    )
    model, tokenizer, processor, config = loader.load()

    # モデル初期設定
    tokenizer.model_max_length = infer_args.model_max_length
    model.eval().to("cuda")
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # # モデル設定の保存
    # config.save_pretrained(out_dir)
    # model.generation_config.to_json_file(
    #     os.path.join(out_dir, "generation_config.json")
    # )

    # コラレータ（推論モード）
    data_collator = COLLATORS[model_args.model_family_id](
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        mask_question_tokens=infer_args.mask_question_tokens,
    )

    # 推論ループ（インデックスで回す）
    rank0_print("Starting evaluation...")
    gen_kwargs = asdict(gen_args)
    all_outputs = []

    pbar = tqdm.tqdm(range(total), desc="Inference", position=0, leave=True)
    for idx in pbar:
        # 生データからメタ情報取得
        raw = raw_samples[idx]
        img = raw.get("image", "N/A")
        convs = raw.get("conversations", [])
        lbl = convs[1]["value"] if len(convs) > 1 and "value" in convs[1] else "N/A"

        # 前処理済みサンプルを取得して collate_fn に渡す
        proc = eval_dataset[idx]
        batch = data_collator([proc])

        # Tensor 部分だけ GPU に移動
        inputs = {k: v.to("cuda") for k, v in batch.items() if torch.is_tensor(v)}

        # 推論
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=compute_dtype):
            gen_ids = model.generate(**inputs, **gen_kwargs)

        # テキスト化 & 最終行抽出
        trimmed = [
            out_ids[len(inp):]
            for inp, out_ids in zip(inputs["input_ids"], gen_ids)
        ]
        raw_text = processor.batch_decode(trimmed, skip_special_tokens=True)
        text = raw_text[0] if raw_text else ""
        pred = text.strip().splitlines()[-1]

        # ログ表示
        tqdm.tqdm.write(f"image: {img}  |  label: {lbl}  |  pred: {pred}")

        # 結果格納
        all_outputs.append({
            "image": img,
            "label": lbl,
            "prediction": pred
        })

        # # 早期終了
        # if infer_args.infer_stop_steps and (idx + 1) >= infer_args.infer_stop_steps:
        #     break

    pbar.close()

    # 結果保存
    save_json(os.path.join(out_dir, "eval_output.json"), all_outputs)


if __name__ == "__main__":
    train()
