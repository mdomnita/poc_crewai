import os
from dotenv import load_dotenv

# For Google Gemini (if you switch back)
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# For Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

def get_llm():
    """
    Initialize and return the LLM model.
    Using Ollama to run models locally. Make sure you have Ollama installed and running.
    Example models: 'llama3', 'mistral', 'phi3'.
    """
    return ChatOllama(model="llama3", temperature=0.2)

def get_embeddings():
    """
    Initialize and return the embedding model for the RAG pipeline.
    Using Ollama embeddings (ensure the same model or a specific embedding model is pulled, like 'nomic-embed-text').
    """
    return OllamaEmbeddings(model="nomic-embed-text")
