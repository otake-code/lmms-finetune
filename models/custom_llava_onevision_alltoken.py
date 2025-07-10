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
        # トークンごとにスコアを出す線形層
        self.classifier = nn.Linear(hidden_size, 1)
        self.sigmoid    = nn.Sigmoid()
        self.mse_loss   = nn.MSELoss()

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # ベースモデルの出力を取得（hidden_states[-1] が必要）
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

        # 全トークンに同じ線形層を適用 ⇒ [batch, seq_len, 1]
        token_logits = self.classifier(last_hidden)
        token_logits = token_logits.squeeze(-1)   # [batch, seq_len]

        # トークンごとの確率
        token_probs  = self.sigmoid(token_logits) # [batch, seq_len]

        # シーケンス全体の yes 確率（平均プーリング）
        seq_prob     = token_probs.mean(dim=1)    # [batch]

        loss = None
        if labels is not None:
            # labels: [batch] の 0/1
            labels = labels.float().view_as(seq_prob)
            # MSELoss で seq_prob と比較
            loss = self.mse_loss(seq_prob.half(), labels.half())

        # 出力に追加
        outputs["yesno_token_logits"] = token_logits
        outputs["yesno_token_probs"]  = token_probs
        outputs["yesno_seq_prob"]     = seq_prob
        if loss is not None:
            outputs["loss"] = loss

        return outputs
