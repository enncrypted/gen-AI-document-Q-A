# Document Intelligence & RAG Q&A System

An AI-powered Document Intelligence application built with **Streamlit**, **LangChain**, **FAISS**, and **Groq LLMs**. This system enables users to upload custom documents, generate vector embeddings, query document knowledge bases, and receive LLM-generated answers complete with source citations.

---

## 🌟 Key Features

- **Multi-Format Document Parsing**: Supports uploading and processing `.pdf`, `.txt`, `.md`, `.log`, `.csv`, `.xlsx`, `.xls`, `.docx`, and `.json` files.
- **Dynamic UI File Uploads**: Upload one or multiple document files directly through the Streamlit sidebar control panel at any time.
- **FAISS Vector Indexing**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) for local vector embeddings and fast similarity retrieval.
- **Context-Backed AI Answers**: Uses Groq LLMs to answer user questions strictly based on retrieved document passages.
- **Source Citations**: Displays expandable citations showing the source filename, page number, and matched text passage for full answer transparency.
- **Vector Store Control & Recovery**: 
  - **Reset Vector Store**: Clear indexed vectors and chat history in one click.
  - **Load Sample Documents**: Instant one-click re-indexing of bundled sample documents post-reset.

---

## 📁 Repository Structure

```
.
├── app.py                  # Main Streamlit web application interface
├── src/
│   ├── data_loader.py      # Document loading logic for disk & UI uploaded files
│   ├── embedding.py        # Text chunking and sentence-transformer embeddings pipeline
│   ├── vectorstore.py      # FAISS vector store manager (build, save, load, clear, query)
│   └── search.py           # RAG retrieval and Groq Chat LLM synthesis engine
├── data/                   # Default sample documents (PDFs, text files)
├── faiss_store/            # Persisted FAISS vector index and metadata pickle files
├── .env                    # Environment configuration (Groq API Key)
├── requirements.txt        # Python dependency specifications
└── pyproject.toml          # Project configuration & dependency declarations
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites

- Python **3.10+** (Python 3.12 recommended)
- A free **Groq API Key** from [Groq Console](https://console.groq.com)

### 2. Environment Setup

Create a `.env` file in the project root folder (or copy from `.env.example`):

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Installation

Using `uv` (recommended):

```bash
uv sync
```

Or using standard `pip`:

```bash
pip install -r requirements.txt
```

### 4. Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 💡 How to Use

1. **Configure API Key**: The application automatically detects `GROQ_API_KEY` from `.env`. You can also enter or override your Groq API key in the sidebar.
2. **Upload Documents**:
   - In the sidebar, expand **📁 Upload New Documents**.
   - Browse and select your document files (`.pdf`, `.txt`, `.csv`, `.docx`, etc.).
   - Click **📥 Index Uploaded Files** to embed and add them to the vector store.
3. **Ask Questions**: Type any query into the main chat box (e.g., *"What is the main topic of the uploaded document?"*).
4. **View Citations**: Expand **📚 View Source Citations** under assistant answers to verify source passages and page numbers.
5. **Reset / Reload Store**:
   - Click **🗑️ Reset Vector** in the sidebar to purge the vector store.
   - Click **🔄 Load Samples** to re-populate the vector store with initial sample documents.

---

## 🛠️ Tech Stack

- **Frontend Interface**: Streamlit
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector Database**: Meta FAISS (`faiss-cpu`)
- **Document Processing**: LangChain Community Loaders (`PyPDFLoader`, `Docx2txtLoader`, `CSVLoader`, `TextLoader`, `JSONLoader`)
- **LLM Integration**: `langchain-groq`
