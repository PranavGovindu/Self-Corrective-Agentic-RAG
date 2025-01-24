# Agentic-RAG

Agentic-RAG is a retrieval-augmented generation (RAG) system designed to enhance decision-making and query processing using a combination of retrieval and generation techniques. The system is structured into several core components, each responsible for a specific aspect of the RAG process.

## Project Structure

```
agentic-rag/
├── core/
│   ├── agents/
│   │   ├── orchestrator.py     # Main decision-making
│   │   ├── reflector.py        # Self-critique
│   │   └── planner.py          # Query decomposition
│   ├── retrieval/
│   │   ├── hybrid_search.py    # BM25 + Vector
│   │   ├── graph_expander.py   # Neo4j relationships
│   │   └── reranker.py         # Lightweight cross-encoder
│   └── generation/
│       └── llm_gateway.py      # Phi-3/Mistral interface
├── data/
│   ├── knowledge_graph/        # Neo4j storage
│   └── vector_db/              # Chroma/Qdrant
└── config/
    └── agentic.yaml            # Central configuration
```

## Getting Started

To get started with Agentic-RAG, follow the instructions in the `agentic.yaml` configuration file and ensure all dependencies are installed.

## License

This project is licensed under the MIT License.
