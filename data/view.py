import pandas as pd

df = pd.read_parquet('data/processed/train.parquet')
print(df.head())
print(df.columns)
# print(df.iloc[0:5]['prompt'])
for i in range(5000, 5005):
    print(df.iloc[i]['reward_model'])