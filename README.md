# Agentic-RAG

Agentic-RAG is an autonomous knowledge orchestration system that transcends traditional retrieval-augmented generation by implementing intelligent agents that actively dissect, evaluate, and synthesize information. Unlike conventional RAG systems that passively match queries to documents, Agentic-RAG employs a cognitive architecture that:

- Decomposes complex queries into meaningful semantic units
- Actively evaluates and filters information based on contextual relevance
- Synthesizes knowledge through multi-step reasoning chains
- Maintains transparency by providing detailed reasoning paths and source citations
- Adapts its retrieval strategy based on query complexity and context needs

At its core, Agentic-RAG combines the precision of hybrid retrieval with the adaptability of autonomous agents, creating a system that not only answers questions but actively participates in the knowledge discovery process. The system's unique approach to knowledge processing enables it to handle nuanced queries with a level of sophistication that mirrors human-like understanding and reasoning.

## Features

- **Hybrid Retrieval System**: Combines vector-based (Pinecone) and lexical (BM25) search for improved document retrieval
- **Contextual RAG (CRAG)**: Advanced document processing with relevance evaluation and knowledge strip extraction
- **Multi-Query Generation**: Intelligent query expansion and reformulation for better coverage
- **Web Search Integration**: Optional Tavily-powered web search capability
- **Interactive UI**: Streamlit-based interface with chat history and source citations
- **Flexible Document Loading**: Support for PDFs, CSVs, text files, and web pages

## Project Structure

```plaintext
core/
├── config.py         # Configuration and environment setup
├── orchastrator.py   # Main Streamlit UI and system orchestration
├── test.py          # Test suite (to be implemented)
└── vector.py        # Core RAG implementation and document processing

# Configuration Files
requirements.txt    # Python package dependencies
.env               # Environment variables (API keys)
README.md          # Project documentation
```

## Dependencies

Major dependencies include:

- langchain and related packages for RAG pipeline
- Pinecone for vector storage
- Hugging Face for embeddings (BAAI/bge-m3)
- Ollama for LLM integration (qwen2.5-coder)
- Streamlit for UI
- Various document processing libraries (PyPDF, BS4, etc.)

See `requirements.txt` for the complete list of dependencies.

## Setup

1. Create a `.env` file with required API keys:

```env
PINECONE_API_KEY=your-pinecone-api-key
TAVILY_API_KEY=your-tavily-api-key (optional)
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run the application:

```bash
streamlit run core/orchastrator.py
```

## Usage

The system provides a web interface where you can:

1. Load documents from various sources (URLs, PDFs, etc.)
2. Ask questions about the loaded content
3. View detailed source citations and relevance scores
4. Clear or reset the knowledge base as needed

## License

This project is licensed under the MIT License.
