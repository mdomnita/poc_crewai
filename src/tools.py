from crewai.tools import tool
from .rag_pipeline import setup_rag_pipeline
import os

_cached_retriever = None
_cached_mtime = None

def get_retriever():
    global _cached_retriever, _cached_mtime
    try:
        if not os.path.exists("data"):
            return None
        pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
        if pdf_files:
            latest_mtime = max(os.path.getmtime(os.path.join("data", f)) for f in pdf_files)
            if _cached_retriever is None or latest_mtime != _cached_mtime:
                print("Updating Document Retriever...")
                _cached_retriever = setup_rag_pipeline(data_dir="data")
                _cached_mtime = latest_mtime
        else:
            _cached_retriever = None
            _cached_mtime = None
    except Exception as e:
        print(f"Error getting retriever: {e}")
        _cached_retriever = None
    return _cached_retriever

@tool("Document Query Tool")
def query_documents(query: str) -> str:
    """
    Use this tool to search and retrieve information from the uploaded PDF and CSV documents. 
    Pass a specific query string containing the core concepts or questions you want to search for.
    The tool will return the most relevant text excerpts from the documents.
    """
    retriever = get_retriever()
    if retriever is None:
        return "Error: No documents available with text content. Please upload a valid PDF."
        
    print(f"\n[Tool Executing] Searching documents for: '{query}'")
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant information found in the documents for the given query."
        
    # Combine the retrieved document chunks into a single text block
    retrieved_content = "\n\n---\n\n".join([f"Source ({d.metadata.get('source', 'Unknown')}): {d.page_content}" for d in docs])
    
    return retrieved_content
