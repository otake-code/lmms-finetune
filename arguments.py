from typing import Dict, Optional, List
from dataclasses import dataclass, field

import transformers

from supported_models import MODEL_HF_PATH, MODEL_FAMILIES


@dataclass
class ModelArguments:
    model_id: str = field(default="llava-1.5-7b")
    model_local_path: Optional[str] = field(default=None)

    def __post_init__(self):
        assert self.model_id in MODEL_HF_PATH, f"Unknown model_id: {self.model_id}"
        self.model_hf_path: str = MODEL_HF_PATH[self.model_id]
        assert self.model_id in MODEL_FAMILIES, f"Unknown model_id: {self.model_id}"
        self.model_family_id: str = MODEL_FAMILIES[self.model_id]

        if not self.model_local_path:
            self.model_local_path = self.model_hf_path


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data json file."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data json file."}
    )
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    num_frames: Optional[int] = field(default=8)
    user_key: Optional[str] = field(default="human")
    assistant_key: Optional[str] = field(default="gpt")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_flash_attn: bool = field(default=False)
    train_vision_encoder: bool = field(default=False)
    train_vision_projector: bool = field(default=False)
    mask_question_tokens: bool = field(default=True)

    # Stage1 モード：Yes/No 判定ヘッド＋Vision Projector のみ学習
    stage1: bool = field(
        default=False,
        metadata={"help": "If True, run only Stage1: train classifier  vision_projector, freeze everything else."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False

        # Stage1 モードなら自動でプロジェクターも有効にする
        if self.stage1:
            # Vision encoder は常に Freeze
            self.train_vision_encoder = False
            # Vision projector は Stage1 で学習対象
            self.train_vision_projector = True


@dataclass
class LoraArguments:
    use_lora: bool = field(default=True)
    use_vision_lora: bool = field(default=True)
    q_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = ""
    lora_bias: str = "none"

@dataclass
class InferModelArguments:
    model_id: str = field(default="llava-1.5-7b")
    model_local_path: Optional[str] = field(default=None)
    lora_path: Optional[str] = field(default=None)

    def __post_init__(self):
        assert self.model_id in MODEL_HF_PATH, f"Unknown model_id: {self.model_id}"
        self.model_hf_path: str = MODEL_HF_PATH[self.model_id]
        assert self.model_id in MODEL_FAMILIES, f"Unknown model_id: {self.model_id}"
        self.model_family_id: str = MODEL_FAMILIES[self.model_id]

        if not self.model_local_path:
            self.model_local_path = self.model_hf_path

@dataclass
class InferArguments:
    output_dir: Optional[str] = field(default=None, metadata={"help": "カスタムの出力先ディレクトリ"})
    seed: int = field(default=42, metadata={"help": "Random seed for initialization."})
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 (mixed precision) instead of fp16."},
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed precision) instead of bf16."},
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        },
    )
    per_device_infer_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size (per device) for the evaluation dataloader."
        },
    )
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_flash_attn: bool = field(default=False)
    mask_question_tokens: bool = field(default=True)
    infer_stop_steps: int = field(
        default=None,
        metadata={
            "help": "Number of steps to run inference for. If set to None, will run inference until the end of the dataset."
        },
    )
    infer_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Whether to evaluate the model checkpoint. If set to True, will inference all model checkpoint."
        },
    )


@dataclass
class InferMultiArguments:
    per_device_infer_batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size (per device) for the evaluation dataloader."
        },
    )

    infer_stop_steps: int = field(
        default=None,
        metadata={
            "help": "Number of steps to run inference for. If set to None, will run inference until the end of the dataset."
        },
    )
    infer_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Whether to evaluate the model checkpoint. If set to True, will inference all model checkpoint."
        },
    )

@dataclass
class GenerationArguments:
    max_new_tokens: int = field(
        default=512,
        metadata={
            "help": "The maximum number of new tokens to generate in the output sequence."
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "The value used to module the next token probabilities."
        },
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "The cumulative probability for nucleus sampling."
        },
    )
    top_k: int = field(
        default=50,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."
        },
    )
    num_beams: int = field(
        default=1,
        metadata={
            "help": "Number of beams for beam search. 1 means no beam search."
        },
    )
    do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether to use sampling; use greedy decoding otherwise."
        },
    )

