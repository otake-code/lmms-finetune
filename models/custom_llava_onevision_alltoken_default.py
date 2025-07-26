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

        # 最終層の隠れ状態 [batch, seq_len, hidden_size]

        last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

        # ① 全トークンを分類器に通す → [batch, seq_len, 1]
        token_logits = self.classifier(last_hidden)
        token_probs  = self.sigmoid(token_logits)

        # ② パッドを除外してトークン軸で平均プーリング
        if attention_mask is not None:
            # [batch, seq_len, 1]
            mask = attention_mask.unsqueeze(-1).to(token_probs.dtype)
            masked = token_probs * mask
            sum_probs = masked.sum(dim=1)               # [batch, 1]
            valid_cnt = mask.sum(dim=1).clamp(min=1)    # 0除算防止
            final_prob = sum_probs / valid_cnt          # [batch, 1]
        else:
            final_prob = token_probs.mean(dim=1)        # attention_mask がない場合

        # ③ MSELoss の計算（half precision で高速化）
        loss = None
        if labels is not None:
            labels = labels.float().view_as(final_prob)
            # self.mse_loss = nn.MSELoss() として定義しておく
            loss = self.mse_loss(final_prob.half(), labels.half())

        # ④ 出力に付与
        outputs["yesno_logits"] = token_logits    # すべてのトークン分のロジット
        outputs["yesno_probs"]  = final_prob      # 平均化された確率
        if loss is not None:
            outputs["loss"] = loss

        return outputs
