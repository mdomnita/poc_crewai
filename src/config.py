import os
from crewai import LLM
from dotenv import load_dotenv

# For Google Gemini (if you switch back)
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# For OpenAI imports are now local to functions to avoid ModuleNotFoundError

load_dotenv(override=True)

def get_llm():
    """
    Initialize and return the LLM model based on environment config.
    """
    model_provider = os.getenv("MODEL_PROVIDER", "ollama")
    model_name = os.getenv("MODEL_NAME", "llama3") # Default for Ollama if not specified
    
    if model_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name)
    elif model_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        raw_model = model_name.replace("gemini/", "")
        return ChatGoogleGenerativeAI(model=raw_model, temperature=0.2)
    else:
        # For Ollama
        from langchain_ollama import ChatOllama
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
    elif model_provider == "gemini":
        if not model_name.startswith("gemini/"):
            model_name = f"gemini/{model_name}"
        return LLM(model=model_name, temperature=0.2)
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
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    elif model_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    else:
        # Default to Ollama embeddings
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")
