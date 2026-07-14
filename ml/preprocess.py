import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ======================================
# Project Paths
# ======================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "trades.csv"

MODEL_PATH = BASE_DIR / "models"

MODEL_PATH.mkdir(exist_ok=True)

# ======================================
# Load Dataset
# ======================================

print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv(DATA_PATH)

print("\nFirst 5 Records")
print(df.head())

# ======================================
# Dataset Information
# ======================================

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

# ======================================
# Missing Values
# ======================================

print("\nMissing Values")

print(df.isnull().sum())

# ======================================
# Duplicate Records
# ======================================

print("\nDuplicate Records")

print(df.duplicated().sum())

# ======================================
# Owner Distribution
# ======================================

print("\nOwner Distribution")

print(df["Owner"].value_counts())

# ======================================
# Encoding
# ======================================

encoders = {}

categorical_columns = [

    "Currency",

    "Product",

    "Country",

    "SettlementType",

    "Priority",

    "Owner"

]

print("\nEncoding Columns...")

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    encoders[column] = encoder

print("\nEncoding Completed")

# ======================================
# Save Encoders
# ======================================

joblib.dump(encoders, MODEL_PATH / "encoders.pkl")

print("\nEncoders Saved Successfully")


# ======================================
# Save Encoded Dataset
# ======================================

ENCODED_PATH = BASE_DIR / "data" / "trades_encoded.csv"

df.to_csv(ENCODED_PATH, index=False)

print("\nEncoded Dataset Saved Successfully")
print("Location :", ENCODED_PATH)

print("\nFirst 5 Encoded Records")
print(df.head())

print("\nPreprocessing Completed Successfully")

