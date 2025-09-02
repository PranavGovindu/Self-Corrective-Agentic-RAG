"""
Configuration Module for Synergistic RAG System

Centralizes all configuration constants and environment setup for the RAG system.
This module provides default values and environment variable management for
consistent configuration across all components.

Configuration Philosophy:
- Environment variables take precedence over hardcoded values
- Sensible defaults for development and testing
- Clear separation between required and optional configurations
- Easy deployment configuration through environment variables
"""

import os

# =============================================================================
# VECTOR DATABASE CONFIGURATION
# =============================================================================

# Pinecone API credentials - REQUIRED for vector storage
# Replace with your actual API key or set PINECONE_API_KEY environment variable
PINECONE_API_KEY = "your-pinecone-api-key"

# Pinecone index configuration
INDEX_NAME = "rag"           # Index name in Pinecone - must be unique per account
DIMENSION = 1024             # Embedding dimension - must match embedding model output

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Embedding model for document vectorization
# BAAI/bge-m3 provides excellent multilingual support and semantic understanding
# Alternative: "sentence-transformers/all-MiniLM-L6-v2" for lighter resource usage
EMBEDDING_MODEL = "BAAI/bge-m3"

# Large Language Model for query processing and response generation
# Uses Ollama for local inference - requires ollama serve to be running
# Alternative: OpenAI models like "gpt-3.5-turbo" for cloud-based inference
LLM_MODEL = "qwen2.5-coder:1.5b"

# =============================================================================
# WEB SEARCH CONFIGURATION
# =============================================================================

# Tavily API key for web search capabilities - OPTIONAL but recommended
# Web search provides fallback when local knowledge is insufficient
# Sign up at https://tavily.com to get your API key
TAVILY_API_KEY = "your-tavily-api-key"

# =============================================================================
# QUALITY CONTROL PARAMETERS
# =============================================================================

# Relevance threshold for document filtering in CRAG processing
# Lower values (0.1-0.2): More inclusive, potentially noisy results
# Higher values (0.4-0.5): More selective, potentially missing relevant content
# Current value (0.3): Balanced approach for most use cases
RELEVANCE_THRESHOLD = 0.3

def setup_environment():
    """
    Configure environment variables with fallback to configuration constants.
    
    This function ensures that the system has access to required configuration
    values either from environment variables (preferred for production) or
    from the constants defined in this module (useful for development).
    
    Environment Variable Priority:
    1. Existing environment variables (highest priority)
    2. Configuration constants from this module (fallback)
    
    Note:
    This approach allows flexible deployment where sensitive values like API keys
    can be injected via environment variables while maintaining working defaults
    for development environments.
    """
    # Set Pinecone API key if not already configured in environment
    # Production deployments should set this via environment variables
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    
    # Set Tavily API key for web search functionality
    # This is optional - system will work without web search if not provided
    if not os.getenv("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
