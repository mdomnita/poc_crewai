import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import get_embeddings

def load_documents(data_dir="data"):
    """
    Load all PDF and CSV documents from the specified directory.
    Demonstrates simple data pre-processing using Python and Langchain loaders.
    """
    documents = []
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Creating it...")
        os.makedirs(data_dir, exist_ok=True)
        return documents

    # Load PDFs
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    for file_path in pdf_files:
        print(f"Loading PDF: {file_path}")
        loader = PyMuPDFLoader(file_path)
        documents.extend(loader.load())
        
    # Load CSVs
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    for file_path in csv_files:
        print(f"Loading CSV: {file_path}")
        loader = CSVLoader(file_path)
        documents.extend(loader.load())
        
    return documents

def build_vector_store(documents):
    """
    Split documents into chunks and build a FAISS vector store.
    """
    if not documents:
        return None
        
    print(f"Splitting {len(documents)} document pages/rows into chunks...")
    # Text splitting configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    
    print(f"Building vector store with {len(splits)} chunks...")
    if not splits:
        print("Warning: No text chunks could be extracted from the documents. The PDFs might be entirely image-based without a text layer.")
        return None
        
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    
    return vector_store

def setup_rag_pipeline(data_dir="data"):
    """
    Main function to set up the RAG pipeline. Returns a retriever object.
    """
    docs = load_documents(data_dir)
    if not docs:
        print("No documents found to process. Please add some PDFs or CSVs to the 'data' folder.")
        return None
        
    vector_store = build_vector_store(docs)
    if not vector_store:
        return None
    
    # Prompt engineering: configuring the retriever for better context
    # search_kwargs={"k": 5} retrieves the top 5 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever
