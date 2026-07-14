import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===========================================
# Project Paths
# ===========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "trades_encoded.csv"
MODEL_PATH = BASE_DIR / "models"

# ===========================================
# Load Dataset
# ===========================================

print("=" * 60)
print("Loading Encoded Dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# ===========================================
# Features & Target
# ===========================================

X = df.drop(columns=["TradeID", "Owner"])

y = df["Owner"]

print("\nFeatures")
print(X.columns.tolist())

print("\nTarget")
print("Owner")

# ===========================================
# Train Test Split
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ===========================================
# Build Model
# ===========================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42

)

model.fit(X_train, y_train)

print("Training Completed")

# ===========================================
# Prediction
# ===========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

# ===========================================
# Save Model
# ===========================================

joblib.dump(model, MODEL_PATH / "model.pkl")

print("\nModel Saved Successfully")

print(MODEL_PATH / "model.pkl")