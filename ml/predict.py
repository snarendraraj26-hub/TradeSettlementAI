import pandas as pd
import joblib
from pathlib import Path

# ======================================
# Project Paths
# ======================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models"

# ======================================
# Load Model & Encoders
# ======================================

model = joblib.load(MODEL_PATH / "model.pkl")

encoders = joblib.load(MODEL_PATH / "encoders.pkl")

print("=" * 60)
print("Trade Settlement Owner Prediction")
print("=" * 60)

# ======================================
# User Input
# ======================================

amount = int(input("Amount : "))

currency = input("Currency (USD/EUR/GBP/INR): ")

product = input("Product (Equity/Bond/ETF/Derivative): ")

country = input("Country (India/Germany/France/UK/USA): ")

settlement = input("Settlement Type (T+1/T+2/T+3): ")

priority = input("Priority (High/Medium/Low): ")

# ======================================
# Encode Inputs
# ======================================

try:
    currency = encoders["Currency"].transform([currency])[0]
    product = encoders["Product"].transform([product])[0]
    country = encoders["Country"].transform([country])[0]
    settlement = encoders["SettlementType"].transform([settlement])[0]
    priority = encoders["Priority"].transform([priority])[0]

except ValueError as e:
    print("\n❌ Invalid input!")
    print(e)

    print("\nValid values are:")

    print("Currency :", list(encoders["Currency"].classes_))
    print("Product :", list(encoders["Product"].classes_))
    print("Country :", list(encoders["Country"].classes_))
    print("Settlement Type :", list(encoders["SettlementType"].classes_))
    print("Priority :", list(encoders["Priority"].classes_))

    exit()

# ======================================
# Create DataFrame
# ======================================

input_df = pd.DataFrame({

    "Amount": [amount],

    "Currency": [currency],

    "Product": [product],

    "Country": [country],

    "SettlementType": [settlement],

    "Priority": [priority]

})

# ======================================
# Predict
# ======================================

prediction = model.predict(input_df)[0]

owner = encoders["Owner"].inverse_transform([prediction])[0]

# ======================================
# Display Result
# ======================================

print("\n" + "=" * 60)

print("Predicted Settlement Owner :", owner)

print("=" * 60)