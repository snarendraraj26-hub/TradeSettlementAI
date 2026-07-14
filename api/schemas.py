from pydantic import BaseModel


class TradeRequest(BaseModel):

    amount: int

    currency: str

    product: str

    country: str

    settlement_type: str

    priority: str


class PredictionResponse(BaseModel):

    predicted_owner: str