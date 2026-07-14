import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

# ======================================
# Project Paths
# ======================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "trades_encoded.csv"

MODEL_PATH = BASE_DIR / "models"

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv(DATA_PATH)

# ======================================
# Features & Target
# ======================================

X = df.drop(columns=["TradeID", "Owner"])

y = df["Owner"]

# ======================================
# Train-Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ======================================
# Load Trained Model
# ======================================

model = joblib.load(MODEL_PATH / "model.pkl")

# ======================================
# Prediction
# ======================================

predictions = model.predict(X_test)

# ======================================
# Accuracy
# ======================================

accuracy = accuracy_score(y_test, predictions)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy : {accuracy * 100:.2f}%")

# ======================================
# Classification Report
# ======================================

print("\nClassification Report\n")

print(classification_report(y_test, predictions))

# ======================================
# Confusion Matrix
# ======================================

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, predictions))

# ======================================
# Feature Importance
# ======================================

print("\nFeature Importance\n")

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)