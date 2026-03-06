import pandas as pd

df = pd.read_parquet('data/processed/train.parquet')
print(df.head())
print(df.columns)
for i in range(0, 5):
    print(df.iloc[i]['reward_model'])
    print(df.iloc[i]['prompt'])