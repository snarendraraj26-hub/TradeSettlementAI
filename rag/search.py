from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ======================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = BASE_DIR / "chroma_db"

# ======================================

print("=" * 60)
print("Loading Embedding Model...")
print("=" * 60)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading Vector Database...")

db = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding
)

print("Ready!")

# ======================================

while True:

    question = input("\nAsk a Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    results = db.similarity_search(question, k=3)

    print("\n" + "=" * 60)
    print("Top Matching Chunks")
    print("=" * 60)

    for i, doc in enumerate(results, start=1):

        print(f"\nChunk {i}\n")

        print(doc.page_content)

        print("-" * 60)