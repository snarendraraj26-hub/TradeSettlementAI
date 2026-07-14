import pandas as pd

# Load dataset

df = pd.read_csv("data/trades.csv")

print("========== DATA ==========")
print(df)

print("\n========== INFO ==========")
print(df.info())

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== SHAPE ==========")
print(df.shape)