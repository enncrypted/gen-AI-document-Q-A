import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b",
        groq_api_key: str = None
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.vectorstore.load()
        
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.llm_model = llm_model
        self.llm = None
        
        if self.groq_api_key:
            try:
                self.llm = ChatGroq(groq_api_key=self.groq_api_key, model_name=llm_model)
                print(f"[INFO] Groq LLM initialized: {llm_model}")
            except Exception as e:
                print(f"[WARNING] Could not initialize Groq LLM: {e}")
        else:
            print("[WARNING] GROQ_API_KEY not found. LLM queries will require an API key.")

    def set_api_key(self, api_key: str):
        if api_key:
            self.groq_api_key = api_key
            self.llm = ChatGroq(groq_api_key=self.groq_api_key, model_name=self.llm_model)
            print(f"[INFO] Groq LLM updated with provided API key: {self.llm_model}")

    def query_with_sources(self, query: str, top_k: int = 5) -> dict:
        results = self.vectorstore.query(query, top_k=top_k)
        if not results:
            return {
                "answer": "No relevant documents found in the vector store. Please upload documents first.",
                "sources": []
            }

        sources = []
        context_passages = []
        for r in results:
            meta = r.get("metadata", {})
            text = meta.get("text", "")
            source_name = meta.get("source") or meta.get("filename") or "Uploaded Document"
            page = meta.get("page")
            
            sources.append({
                "source": source_name,
                "page": page,
                "distance": r.get("distance"),
                "text": text
            })
            context_passages.append(f"--- Document: {source_name} (Page {page if page is not None else 'N/A'}) ---\n{text}")

        context = "\n\n".join(context_passages)

        if not self.llm:
            return {
                "answer": "⚠️ **GROQ API Key missing!** Please enter your Groq API key in the sidebar to generate answers.\n\nHere are the top retrieved passages from your documents:\n\n" + context[:1000] + "...",
                "sources": sources
            }

        prompt = f"""You are a helpful AI assistant answering queries based strictly on the provided document context.

Query: {query}

Document Context:
{context}

Answer the query comprehensively based only on the above context. If the answer cannot be found in the context, state that clearly.

Answer:"""
        try:
            response = self.llm.invoke([prompt])
            answer = response.content
        except Exception as e:
            answer = f"Error generating answer from LLM: {e}"

        return {
            "answer": answer,
            "sources": sources
        }

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        res = self.query_with_sources(query, top_k=top_k)
        return res["answer"]
