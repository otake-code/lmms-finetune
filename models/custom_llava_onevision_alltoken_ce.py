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
        # トークンごとにスコアを出す線形層（ロジット出力用）
        self.classifier   = nn.Linear(hidden_size, 1)
        # 学習用の損失関数（sigmoid を内部で呼ぶ BCEWithLogits）
        self.bce_loss     = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # 1) ベースモデルから hidden_states を取得
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )

        # 2) 最終層の隠れ状態 [batch, seq_len, hidden_size]
        last_hidden = outputs.hidden_states[-1]

        # 3) 全トークンを分類器に通してロジット取得 → [batch, seq_len, 1]
        token_logits  = self.classifier(last_hidden)

        # import pdb; pdb.set_trace()
        # 4) パッドを除外してトークン軸で平均ロジットにプーリング
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(token_logits.dtype)  # [batch, seq_len, 1]
            masked_logits = token_logits * mask
            sum_logits    = masked_logits.sum(dim=1)                    # [batch, 1]
            valid_cnt     = mask.sum(dim=1).clamp(min=1)                # 0除算防止
            pooled_logits = sum_logits / valid_cnt                      # [batch, 1]
        else:
            pooled_logits = token_logits.mean(dim=1)                    # マスクなし時

        # 5) 損失計算はロジットに対して直接 BCEWithLogits
        loss = None
        if labels is not None:
            # labels は 0.0 または 1.0 の float Tensor
            loss = self.bce_loss(
                pooled_logits.view(-1),      # [batch]
                labels.float().view(-1)      # [batch]
            )

        # 6) 推論時の確率化
        final_prob = torch.sigmoid(pooled_logits)  # [batch, 1]

        # 7) 出力に詰め替え
        outputs["yesno_logits"]        = token_logits       # トークンごとのロジット
        outputs["yesno_pooled_logits"] = pooled_logits      # プーリング後ロジット
        outputs["yesno_probs"]         = final_prob         # プーリング後確率
        if loss is not None:
            outputs["loss"]            = loss

        return outputs
