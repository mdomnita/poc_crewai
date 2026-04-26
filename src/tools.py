from langchain.tools import tool
from .rag_pipeline import setup_rag_pipeline

# Initialize RAG pipeline globally so that it's loaded once and used by the tool
print("Initializing Global Document Retriever...")
retriever = setup_rag_pipeline(data_dir="data")

@tool("Document Query Tool")
def query_documents(query: str) -> str:
    """
    Use this tool to search and retrieve information from the uploaded PDF and CSV documents. 
    Pass a specific query string containing the core concepts or questions you want to search for.
    The tool will return the most relevant text excerpts from the documents.
    """
    if retriever is None:
        return "Error: No documents available in the data directory. Please upload PDF or CSV files."
        
    print(f"\n[Tool Executing] Searching documents for: '{query}'")
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant information found in the documents for the given query."
        
    # Combine the retrieved document chunks into a single text block
    retrieved_content = "\n\n---\n\n".join([f"Source ({d.metadata.get('source', 'Unknown')}): {d.page_content}" for d in docs])
    
    return retrieved_content
