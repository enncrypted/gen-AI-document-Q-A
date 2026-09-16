import os
import tempfile
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.log', '.csv', '.xlsx', '.xls', '.docx', '.json'}
IGNORED_EXTENSIONS = {'.sqlite3', '.bin', '.index', '.pkl', '.db', '.pyc', '.exe', '.dll', '.lock'}

def load_uploaded_file(uploaded_file) -> List[Any]:
    """
    Save a Streamlit UploadedFile to a temp file, parse it using load_single_document,
    and clean up temp file afterwards.
    """
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        documents = load_single_document(tmp_path)
        for doc in documents:
            doc.metadata['source'] = uploaded_file.name
            doc.metadata['filename'] = uploaded_file.name
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return documents


def load_single_document(file_path: str) -> List[Any]:
    """
    Load a single document given its file path.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    path = Path(file_path).resolve()
    if not path.exists():
        print(f"[ERROR] File does not exist: {path}")
        return []

    ext = path.suffix.lower()
    if ext in IGNORED_EXTENSIONS:
        return []

    documents = []
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(str(path))
            documents = loader.load()
        elif ext in ['.txt', '.md', '.log']:
            loader = TextLoader(str(path), encoding='utf-8')
            documents = loader.load()
        elif ext == '.csv':
            loader = CSVLoader(str(path))
            documents = loader.load()
        elif ext in ['.xlsx', '.xls']:
            try:
                loader = UnstructuredExcelLoader(str(path))
                documents = loader.load()
            except Exception:
                loader = TextLoader(str(path), encoding='utf-8')
                documents = loader.load()
        elif ext == '.docx':
            loader = Docx2txtLoader(str(path))
            documents = loader.load()
        elif ext == '.json':
            try:
                loader = JSONLoader(str(path), jq_schema='.', text_content=False)
                documents = loader.load()
            except Exception:
                loader = TextLoader(str(path), encoding='utf-8')
                documents = loader.load()
        elif ext in SUPPORTED_EXTENSIONS:
            loader = TextLoader(str(path), encoding='utf-8')
            documents = loader.load()
        else:
            return []
            
        # Ensure source metadata contains basename
        for doc in documents:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = path.name
            else:
                doc.metadata['filename'] = path.name
    except Exception as e:
        print(f"[ERROR] Failed to load document {path}: {e}")

    return documents

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    if not data_path.exists():
        return documents

    for file_path in data_path.glob('**/*'):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            loaded = load_single_document(str(file_path))
            documents.extend(loaded)

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)