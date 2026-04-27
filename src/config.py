import os
from crewai import LLM
from dotenv import load_dotenv

# For Google Gemini (if you switch back)
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# For Ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings

# For OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(override=True)

def get_llm():
    """
    Initialize and return the LLM model based on environment config.
    """
    model_provider = os.getenv("MODEL_PROVIDER", "ollama")
    model_name = os.getenv("MODEL_NAME", "llama3") # Default for Ollama if not specified
    
    if model_provider == "openai":
        return ChatOpenAI(model=model_name)
    else:
        # For Ollama
        raw_model = model_name.replace("ollama/", "")
        return ChatOllama(model=raw_model, temperature=0.2)

def get_crewai_llm():
    """
    Initialize and return a CrewAI-compatible LLM object.
    It wraps the model string correctly for CrewAI 1.x.
    """
    model_provider = os.getenv("MODEL_PROVIDER", "ollama")
    model_name = os.getenv("MODEL_NAME", "ollama/llama3.2")
    
    if model_provider == "openai":
        return LLM(model=model_name)
    else:
        if not model_name.startswith("ollama/"):
            model_name = f"ollama/{model_name}"
        return LLM(model=model_name, temperature=0.2)

def get_embeddings():
    """
    Initialize and return the embedding model based on environment config.
    """
    model_provider = os.getenv("MODEL_PROVIDER", "ollama")
    
    if model_provider == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # Default to Ollama embeddings
        return OllamaEmbeddings(model="nomic-embed-text")
