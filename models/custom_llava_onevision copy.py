#lm head追加
#OOMのため、使えず、別のファイルでheadではなく最終層一個前を採用
import torch
import torch.nn as nn
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionConfig

class LlavaOnevisionForYesNo(LlavaOnevisionForConditionalGeneration):
    """
    Stage1 用に Yes/No 判定ヘッドを追加したカスタムモデル

    カスタムヘッドは以下の処理を行います：
      1. Transformer の最終 hidden_states から BOS トークン位置をプーリング
      2. nn.Linear で logits を計算
      3. Sigmoid で確率に変換
    """
    def __init__(self, config: LlavaOnevisionConfig):
        # 基底クラスの初期化
        config.return_dict = True
        # DeepSpeed の "auto" バケットサイズを解決するために hidden_size を top‐level に登録
        config.hidden_size = config.text_config.hidden_size
        config.output_hidden_states = True
        config.return_dict_in_generate = True
        super().__init__(config)
        # LLM の隠れ層サイズ
        hidden_size = config.text_config.hidden_size
        # バイナリ分類用ヘッド
        self.classifier = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        output_hidden_states=True,
        yesno_labels=None,
        **kwargs
    ):
        # 元の生成モデルとして動作
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )
        # 最終層の出力 (batch_size, seq_len, hidden_size)
        last_hidden = outputs.hidden_states[-1]
        # BOS トークン (<bos>) の隠れ状態を取得
        pooled = last_hidden[:, 0, :]
        # 分類ヘッド
        logits = self.classifier(pooled)       # (batch_size, 1)
        probs  = self.sigmoid(logits)          # (batch_size, 1)

        # MSE Loss を計算
        loss = None
        if yesno_labels is not None:
            # yesno_labels は 0/1 の Tensor を想定
            labels = yesno_labels.float().view_as(probs)
            loss   = self.mse_loss(probs, labels)

        # outputs は ModelOutput(dict subclass) なので属性を追加
        outputs["yesno_logits"] = logits
        outputs["yesno_probs"] = probs
        if loss is not None:
            outputs["loss"] = loss
        return outputs
