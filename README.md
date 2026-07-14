# 🤖 Trade Settlement AI

An AI-powered Trade Settlement Assistant that combines **Machine Learning**, **Retrieval-Augmented Generation (RAG)**, and **Large Language Models (LLMs)** to automate trade settlement owner prediction and answer trade settlement-related questions.

---

# 📌 Overview

Trade Settlement AI is an end-to-end GenAI application built using Python, Scikit-Learn, LangChain, ChromaDB, Ollama, FastAPI, and Streamlit.

The application provides two major capabilities:

- **Settlement Owner Prediction**
  - Predicts the appropriate settlement owner using a Machine Learning model.

- **Trade Settlement Knowledge Assistant**
  - Answers settlement-related questions using Retrieval-Augmented Generation (RAG) powered by Llama 3.2.

---

# 🚀 Features

✅ Machine Learning based Settlement Owner Prediction

✅ Random Forest Classification Model

✅ FastAPI REST APIs

✅ Streamlit Interactive UI

✅ RAG Pipeline

✅ ChromaDB Vector Database

✅ Sentence Transformer Embeddings

✅ Local LLM using Ollama (Llama 3.2)

✅ Semantic Search

✅ Document Retrieval

---

# 🏗️ Architecture

```
                     User
                       │
                       ▼
                Streamlit UI
                       │
                       ▼
                  FastAPI API
               ┌────────┴────────┐
               │                 │
               ▼                 ▼
      ML Prediction        RAG Chatbot
               │                 │
               ▼                 ▼
      Random Forest       ChromaDB Search
               │                 │
               ▼                 ▼
      Settlement Owner      Relevant Documents
                                   │
                                   ▼
                              Llama 3.2
                                   │
                                   ▼
                               AI Response
```

---

# 🤖 Machine Learning Workflow

```
Trade Dataset

        │

        ▼

Data Preprocessing

        │

        ▼

Label Encoding

        │

        ▼

Random Forest Training

        │

        ▼

Model Evaluation

        │

        ▼

Settlement Owner Prediction
```

---

# 🧠 RAG Workflow

```
Settlement Documents

        │

        ▼

Document Loader

        │

        ▼

Text Chunking

        │

        ▼

Sentence Transformer Embeddings

        │

        ▼

ChromaDB

        │

        ▼

Similarity Search

        │

        ▼

Context Retrieval

        │

        ▼

Llama 3.2 (Ollama)

        │

        ▼

AI Generated Response
```

---

# 📂 Project Structure

```
TradeSettlementAI/

│

├── api/
│   ├── app.py
│   ├── chat.py
│   └── schemas.py
│

├── frontend/
│   └── app.py
│

├── ml/
│   ├── generate_dataset.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── predict.py
│   └── test_data.py
│

├── rag/
│   ├── load_documents.py
│   ├── create_embeddings.py
│   ├── chatbot.py
│   ├── router.py
│   ├── search.py
│   └── prompt.py
│

├── documents/
│

├── data/
│

├── requirements.txt

└── README.md
```

---

# 🛠️ Tech Stack

## Programming

- Python 3.14

---

## Machine Learning

- Scikit-Learn
- Pandas
- NumPy
- Joblib

---

## Generative AI

- LangChain
- Ollama
- Llama 3.2

---

## Vector Database

- ChromaDB

---

## Embedding Model

- sentence-transformers/all-MiniLM-L6-v2

---

## Backend

- FastAPI
- Uvicorn

---

## Frontend

- Streamlit

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/snarendraraj26-hub/TradeSettlementAI.git

cd TradeSettlementAI
```

---

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

---

Install dependencies

```bash
pip install -r requirements.txt
```

---

Install Ollama

Download

https://ollama.com

Pull the Llama model

```bash
ollama pull llama3.2
```

---

# ▶️ Running the Project

Start FastAPI

```bash
uvicorn api.app:app --reload
```

Start Streamlit

```bash
streamlit run frontend/app.py
```

Open

```
http://localhost:8501
```

---

# 📚 API Endpoints

## Predict Settlement Owner

```
POST /predict
```

---

## Trade Settlement Chat

```
POST /chat
```

---

## API Documentation

```
http://127.0.0.1:8000/docs
```

---

# 📈 Machine Learning Model

Algorithm Used

- Random Forest Classifier

Input Features

- Amount
- Currency
- Product
- Country
- Settlement Type
- Priority

Prediction

- Settlement Owner

---

# 🧠 RAG Components

- LangChain
- ChromaDB
- Sentence Transformers
- Similarity Search
- Ollama
- Llama 3.2

---

# 🔮 Future Improvements

- LangGraph Agentic AI
- Multi-Agent Architecture
- MCP Server Integration
- Conversation Memory
- Pinecone/Qdrant Vector Database
- Docker Deployment
- Kubernetes Deployment
- CI/CD Pipeline
- JWT Authentication
- Azure OpenAI Integration

---

# 👨‍💻Owner

**Narendraraj S**

GitHub

https://github.com/snarendraraj26-hub

---
