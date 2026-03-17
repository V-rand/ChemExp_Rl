import pandas as pd
from transformers import AutoTokenizer

df = pd.read_parquet("/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/processed/train.parquet")

# 加载对应 tokenizer（Qwen2.5-3B）
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)

def count_tokens(prompt_list):
    """将 message list 转换为文本后计算 token 数"""
    # 使用 chat_template 或直接拼接
    text = tokenizer.apply_chat_template(prompt_list, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text))

df['token_length'] = df['prompt'].apply(count_tokens)
print(df['token_length'].describe())

# 查看超过 1024 的样本比例
overlimit = (df['token_length'] > 1024).sum()
print(f"\n超过 1024 tokens 的样本数: {overlimit} / {len(df)} ({overlimit/len(df)*100:.1f}%)")