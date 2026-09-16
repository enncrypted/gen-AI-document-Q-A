import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import load_single_document, load_all_documents, load_uploaded_file
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Intelligence & Q&A AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished presentation
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 10px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to initialize session state
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = FaissVectorStore(persist_dir="faiss_store")
        loaded = st.session_state.vector_store.load()
        if not loaded or st.session_state.vector_store.index is None:
            docs = load_all_documents("data")
            if docs:
                st.session_state.vector_store.build_from_documents(docs)
    if "uploaded_filenames" not in st.session_state:
        st.session_state.uploaded_filenames = []
        if st.session_state.vector_store and st.session_state.vector_store.metadata:
            sources = set(m.get("source") or m.get("filename") for m in st.session_state.vector_store.metadata if m)
            st.session_state.uploaded_filenames = list(filter(None, sources))

init_session()

# Sidebar Setup - File Upload Only
with st.sidebar:
    st.title("📁 File Upload")
    uploaded_files = st.file_uploader(
        "Upload files to index",
        type=["pdf", "txt", "md", "csv", "xlsx", "xls", "docx", "json"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, MD, CSV, XLSX, DOCX, JSON",
        key="file_uploader"
    )

    if uploaded_files:
        if st.button("📥 Index Uploaded Files", use_container_width=True, type="primary"):
            with st.spinner("Processing & indexing uploaded documents..."):
                all_docs = []
                new_filenames = []
                for uf in uploaded_files:
                    docs = load_uploaded_file(uf)
                    if docs:
                        all_docs.extend(docs)
                        new_filenames.append(uf.name)

                if all_docs:
                    st.session_state.vector_store.build_from_documents(all_docs)
                    for fn in new_filenames:
                        if fn not in st.session_state.uploaded_filenames:
                            st.session_state.uploaded_filenames.append(fn)
                    st.success(f"Indexed {len(new_filenames)} file(s) ({len(all_docs)} sections)!")
                    st.rerun()
                else:
                    st.error("Could not extract readable text from the uploaded file(s).")

# API Key & State Setup
env_api_key = os.getenv("GROQ_API_KEY", "")
effective_api_key = env_api_key
vector_count = st.session_state.vector_store.index.ntotal if (st.session_state.vector_store and st.session_state.vector_store.index) else 0

# Main Interface Header & Controls
col_title, col_opts = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">Document Intelligence & LLM Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your indexed documents and receive context-backed AI answers with citations.</div>', unsafe_allow_html=True)

with col_opts:
    with st.expander("⚙️ Controls & API Key"):
        api_key_override = st.text_input("Override Groq API Key", type="password", placeholder="gsk_...")
        if api_key_override.strip():
            effective_api_key = api_key_override.strip()
        st.metric("Total Indexed Chunks", vector_count)
        if st.button("🗑️ Reset Vector Store", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.uploaded_filenames = []
            st.session_state.messages = []
            st.success("Vector store & history reset!")
            st.rerun()
        if st.button("🔄 Load Sample Docs", use_container_width=True):
            docs = load_all_documents("data")
            if docs:
                st.session_state.vector_store.build_from_documents(docs)
                sources = set(m.get("source") or m.get("filename") for m in st.session_state.vector_store.metadata if m)
                st.session_state.uploaded_filenames = list(filter(None, sources))
                st.success("Loaded sample documents!")
                st.rerun()

if vector_count == 0:
    st.info("💡 **Vector store is empty.** Upload files in the sidebar to index new documents.")


# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Source Citations"):
                for idx, src in enumerate(message["sources"], 1):
                    page_str = f" (Page {src['page']})" if src.get("page") is not None else ""
                    st.markdown(f"**Source {idx}:** `{src['source']}`{page_str}")
                    st.markdown(f"> {src['text']}")

# Chat input
if query := st.chat_input("Ask a question about your uploaded documents..."):
    # Append user query to session
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            rag_search = RAGSearch(groq_api_key=effective_api_key)
            result = rag_search.query_with_sources(query, top_k=4)
            
            answer = result["answer"]
            sources = result["sources"]
            
            st.markdown(answer)
            
            if sources:
                with st.expander("📚 View Source Citations"):
                    for idx, src in enumerate(sources, 1):
                        page_str = f" (Page {src['page']})" if src.get("page") is not None else ""
                        st.markdown(f"**Source {idx}:** `{src['source']}`{page_str}")
                        st.markdown(f"> {src['text']}")
            
            # Save to chat session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

