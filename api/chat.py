from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = BASE_DIR / "chroma_db"

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

def ask_llm(question):

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a Trade Settlement AI Assistant.

Answer only from the context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content