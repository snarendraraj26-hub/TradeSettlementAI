from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# ==========================================
# Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = BASE_DIR / "chroma_db"

# ==========================================
# Embedding Model
# ==========================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# Load ChromaDB
# ==========================================

db = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding
)

# ==========================================
# Load Llama3.2
# ==========================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

print("=" * 60)
print("Trade Settlement AI Chatbot")
print("=" * 60)

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an AI assistant for trade settlement.

Answer ONLY using the context below.

If the answer is not available,
reply:

"I don't know based on the available documents."

Context:

{context}

Question:

{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nBot:\n")

    print(response.content)