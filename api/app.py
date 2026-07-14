from pathlib import Path
from pydantic import BaseModel
from api.chat import ask_llm
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from api.schemas import TradeRequest, PredictionResponse

# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models"

# -------------------------------------------------

model = joblib.load(MODEL_PATH / "model.pkl")

encoders = joblib.load(MODEL_PATH / "encoders.pkl")

# -------------------------------------------------

app = FastAPI(
    title="Trade Settlement AI",
    version="1.0"
)
class ChatRequest(BaseModel):
    question: str

# -------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Trade Settlement AI API Running"
    }

# -------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)

def predict(request: TradeRequest):

    try:

        currency = encoders["Currency"].transform(
            [request.currency]
        )[0]

        product = encoders["Product"].transform(
            [request.product]
        )[0]

        country = encoders["Country"].transform(
            [request.country]
        )[0]

        settlement = encoders["SettlementType"].transform(
            [request.settlement_type]
        )[0]

        priority = encoders["Priority"].transform(
            [request.priority]
        )[0]

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    df = pd.DataFrame({

        "Amount": [request.amount],

        "Currency": [currency],

        "Product": [product],

        "Country": [country],

        "SettlementType": [settlement],

        "Priority": [priority]

    })

    prediction = model.predict(df)[0]

    owner = encoders["Owner"].inverse_transform(
        [prediction]
    )[0]

    return PredictionResponse(
        predicted_owner=owner
    )
@app.post("/chat")
def chat(request: ChatRequest):

    answer = ask_llm(request.question)

    return {
        "answer": answer
    }