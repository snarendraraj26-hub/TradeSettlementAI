import random
import pandas as pd
from pathlib import Path

# -----------------------------
# Project Path
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"
DATA_FOLDER.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_FOLDER / "trades.csv"

# -----------------------------
# Master Data
# -----------------------------

currencies = ["USD", "EUR", "GBP", "INR"]

products = [
    "Equity",
    "Bond",
    "ETF",
    "Derivative"
]

countries = [
    "India",
    "Germany",
    "France",
    "UK",
    "USA"
]

settlement_types = [
    "T+1",
    "T+2",
    "T+3"
]

priorities = [
    "High",
    "Medium",
    "Low"
]

# -----------------------------
# Business Rules
# -----------------------------

def assign_owner(country, product):

    rules = {

        "India": {
            "Equity": "Naren",
            "ETF": "Rahul",
            "Bond": "Sneha",
            "Derivative": "Akhil"
        },

        "Germany": {
            "Equity": "Bharath",
            "ETF": "Priya",
            "Bond": "Revathi",
            "Derivative": "Karthik"
        },

        "France": {
            "Equity": "Naveen",
            "ETF": "Keerthana",
            "Bond": "Divya",
            "Derivative": "Gayathri"
        },

        "UK": {
            "Equity": "Suren",
            "ETF": "Vignesh",
            "Bond": "Manoj",
            "Derivative": "Kavya"
        },

        "USA": {
            "Equity": "Pradeep",
            "ETF": "Harish",
            "Bond": "Arun",
            "Derivative": "Deepak"
        }

    }

    return rules[country][product]
# -----------------------------
# Generate Records
# -----------------------------

records = []

for trade_id in range(1001, 11001):

    country = random.choice(countries)

    product = random.choice(products)

    owner = assign_owner(country, product)

    record = {

        "TradeID": trade_id,

        "Amount": random.randint(5000, 50000),

        "Currency": random.choice(currencies),

        "Product": product,

        "Country": country,

        "SettlementType": random.choice(settlement_types),

        "Priority": random.choice(priorities),

        "Owner": owner

    }

    records.append(record)

# -----------------------------
# Save CSV
# -----------------------------

df = pd.DataFrame(records)

df.to_csv(OUTPUT_FILE, index=False)

print("=" * 50)
print("Dataset Generated Successfully")
print("=" * 50)

print(df.head())

print()

print("Total Records :", len(df))

print("Saved To :", OUTPUT_FILE)