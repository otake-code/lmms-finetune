from typing import Tuple

from transformers import AutoProcessor, PreTrainedTokenizer, AutoConfig
from models.custom_llava_onevision import LlavaOnevisionForYesNo

from . import register_loader
from .base import BaseModelLoader

@register_loader("llava-onevision")
class LLaVAOnevisionModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[LlavaOnevisionForYesNo, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        if load_model:
            # Stage1 用のカスタムモデルをロード
            model = LlavaOnevisionForYesNo.from_pretrained(
                self.model_local_path,
                **self.loading_kwargs,
            )
            # DeepSpeed 等で必要となる場合に hidden_size を同期
            model.config.hidden_size = model.language_model.config.hidden_size
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_hf_path)
        tokenizer = processor.tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        return model, tokenizer, processor, config
