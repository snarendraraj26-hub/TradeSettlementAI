from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# Project Path
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_PATH = BASE_DIR / "documents"

# ==========================================
# Load Documents
# ==========================================

documents = []

print("=" * 60)
print("Loading Documents...")
print("=" * 60)

for file in DOCUMENTS_PATH.iterdir():

    suffix = file.suffix.lower()

    try:

        if suffix == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")

        elif suffix == ".pdf":
            loader = PyPDFLoader(str(file))

        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file))

        else:
            continue

        docs = loader.load()

        documents.extend(docs)

        print(f"Loaded: {file.name}")

    except Exception as e:
        print(f"Failed to load {file.name}: {e}")

print("\nTotal Documents Loaded:", len(documents))

# ==========================================
# Split Documents
# ==========================================

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100

)

chunks = splitter.split_documents(documents)

print("\nTotal Chunks:", len(chunks))

print("\nFirst Chunk:\n")

print(chunks[0].page_content)