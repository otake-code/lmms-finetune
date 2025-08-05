# models/custom_llava_onevision.py

import torch
import torch.nn as nn
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionConfig

class LlavaOnevisionForYesNo(LlavaOnevisionForConditionalGeneration):
    def __init__(self, config: LlavaOnevisionConfig):
        config.return_dict = True
        config.hidden_size = config.text_config.hidden_size
        config.output_hidden_states = True
        config.return_dict_in_generate = True
        super().__init__(config)
        hidden_size = config.text_config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            **kwargs
        )
        last_hidden = outputs.hidden_states[-1]
        pooled = last_hidden[:, -1, :]
        logits = self.classifier(pooled)
        probs  = self.sigmoid(logits)

        loss = None
        if labels is not None:
            labels = labels.float().view_as(probs)
            loss = self.mse_loss(probs.half(), labels.half())

        outputs["yesno_logits"] = logits
        outputs["yesno_probs"]  = probs
        if loss is not None:
            outputs["loss"] = loss
        return outputs
