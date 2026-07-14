from pathlib import Path
import joblib
import pandas as pd

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models"

CHROMA_PATH = BASE_DIR / "chroma_db"

# ----------------------------

model = joblib.load(MODEL_PATH / "model.pkl")
encoders = joblib.load(MODEL_PATH / "encoders.pkl")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding
)

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

print("="*60)
print("Trade Settlement AI Assistant")
print("="*60)

while True:

    question = input("\nYou : ")

    if question.lower()=="exit":
        break

    # ---------------------------
    # Prediction Mode
    # ---------------------------

    if "predict" in question.lower():

        print("\nPrediction Mode\n")

        amount = int(input("Amount : "))
        currency = input("Currency : ")
        product = input("Product : ")
        country = input("Country : ")
        settlement = input("Settlement Type : ")
        priority = input("Priority : ")

        row = pd.DataFrame({

            "Amount":[amount],

            "Currency":[encoders["Currency"].transform([currency])[0]],

            "Product":[encoders["Product"].transform([product])[0]],

            "Country":[encoders["Country"].transform([country])[0]],

            "SettlementType":[encoders["SettlementType"].transform([settlement])[0]],

            "Priority":[encoders["Priority"].transform([priority])[0]]

        })

        prediction = model.predict(row)[0]

        owner = encoders["Owner"].inverse_transform([prediction])[0]

        print("\nPredicted Owner :",owner)

    # --------------------------
    # RAG Mode
    # --------------------------

    else:

        docs = db.similarity_search(question,k=3)

        context = "\n".join([doc.page_content for doc in docs])

        prompt=f"""
Answer only from context.

Context:

{context}

Question:

{question}

Answer:
"""

        response = llm.invoke(prompt)

        print("\nBot:\n")

        print(response.content)