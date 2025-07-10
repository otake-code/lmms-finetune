from transformers import AutoTokenizer

# ① モデル ID を正しく指定（ここでは公開モデルの例）
model_id = "lmms-lab/llava-onevision-qwen2-0.5b-ov"

# ② 認証が必要な場合は use_auth_token=True を付ける
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)

# ③ トークン "[Y/N]:" の ID を取得
token_id_with_colon = tokenizer.convert_tokens_to_ids("[Y/N]:")

# ④ 参考までに、コロンなしもチェック
token_id_without_colon = tokenizer.convert_tokens_to_ids("[Y/N]")

print(f'"[Y/N]:" のトークン ID = {token_id_with_colon}')
print(f'"[Y/N]" のトークン ID = {token_id_without_colon}')
