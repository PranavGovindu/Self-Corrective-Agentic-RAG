import os
import getpass

# Pinecone Configuration
PINECONE_API_KEY = "your-pinecone-api-key"  # Replace with your key
INDEX_NAME = "rag"
DIMENSION = 1024

# Model Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "qwen2.5-coder:1.5b"  # Ollama model name

# Tavily Configuration
TAVILY_API_KEY = "your-tavily-api-key"  # Replace with your key

# Retrieval Configuration
RELEVANCE_THRESHOLD = 0.3

def setup_environment():
    """Set up environment variables if not already set"""
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    if not os.getenv("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY