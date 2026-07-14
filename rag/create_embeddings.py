from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = BASE_DIR / "documents"

CHROMA_PATH = BASE_DIR / "chroma_db"

# =========================================

documents = []

for file in DOCUMENT_PATH.iterdir():

    if file.suffix == ".txt":
        loader = TextLoader(str(file), encoding="utf-8")

    elif file.suffix == ".pdf":
        loader = PyPDFLoader(str(file))

    elif file.suffix == ".docx":
        loader = Docx2txtLoader(str(file))

    else:
        continue

    documents.extend(loader.load())

print("Documents Loaded :", len(documents))

# =========================================

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100

)

chunks = splitter.split_documents(documents)

print("Chunks Created :", len(chunks))

# =========================================

embedding = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)

print("Embedding Model Loaded")

# =========================================

db = Chroma.from_documents(

    documents=chunks,

    embedding=embedding,

    persist_directory=str(CHROMA_PATH)

)

print("Embeddings Stored Successfully")

print("Database Location :", CHROMA_PATH)