"""
Streamlit UI for Agentic RAG System

Interactive web interface for document loading, querying, and result visualization.
Handles session state, error recovery, and real-time feedback.
"""

import os
import sys
from typing import Dict, List, Set

import streamlit as st
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Configure loguru for clean console output and file logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

logger.add(
    "logs/streamlit_app.log", 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
    level="DEBUG",
    rotation="5 MB",
    retention="3 days"
)

# Import RAG system with graceful error handling
try:
    from vector import RAGS
except ImportError:
    st.error("Failed to import RAGS from vector.py. Ensure it's in the correct path and doesn't have syntax errors.")
    logger.error("Failed to import RAGS module", exc_info=True)
    st.stop()
except Exception as e:
    st.error(f"An error occurred during import: {e}")
    logger.error(f"Import error: {e}", exc_info=True)
    st.stop()

def load_css():
    st.markdown("""
        <style>
        /* Your existing CSS styles here */
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem;
        }
        .chat-container {
            border-radius: 10px;
            padding: 15px;
            background-color:
            margin: 10px 0;
            border: 1px solid
        }
        .metadata-container {
            font-size: 0.8em;
            color:
            margin-top: 5px;
        }
        .source-citation {
            background-color:
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85em;
            margin: 0 2px;
            cursor: pointer;
            border: 1px solid
            display: inline-block;
        }
        .reference-container {
            border-left: 3px solid
            padding: 10px;
            margin: 10px 0;
            background-color:
            border-radius: 5px;
        }
        .reference-header {
            font-weight: bold;
            color:
            margin-bottom: 5px;
            word-wrap: break-word;
        }
        .reference-content {
            font-size: 0.9em;
            color:
            max-height: 150px;
            overflow-y: auto;
            background-color:
            padding: 5px;
            border-radius: 3px;
            border: 1px solid
        }
        .reference-metadata {
            font-size: 0.8em;
            color:
            margin-top: 5px;
        }
        .stSpinner > div {
            text-align: center;
            margin: 20px 0;
        }
        .stTextArea textarea {
             min-height: 100px !important;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state with RAG system and UI variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_system" not in st.session_state:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")

        # Validate API keys and initialize RAG system
        if not tavily_api_key:
             logger.warning("Tavily API key not found in environment. Web search will be disabled.")
        if not pinecone_api_key:
             logger.error("Pinecone API key not found in environment. RAG system cannot be initialized.")
             st.error("Pinecone API key missing. Please set the PINECONE_API_KEY environment variable.", icon="🚨")
             st.stop()

        try:
            logger.info("Attempting to initialize RAG system...")
            st.session_state.rag_system = RAGS(tavily_api_key=tavily_api_key)
            logger.info("RAG system initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
            st.error(f"Failed to initialize RAG system: {e}", icon="🚨")
            st.stop()

    if "document_sources" not in st.session_state:
        st.session_state.document_sources: Set[str] = set()

    if "retrieved_documents" not in st.session_state:
        st.session_state.retrieved_documents: List[Dict] = []

    if 'confirm_clear' not in st.session_state:
        st.session_state.confirm_clear = False

    if 'sources_input_area' not in st.session_state:
        st.session_state.sources_input_area = ""


def process_and_store_documents(sources: List[str]) -> int:
    """Process documents and store them in the RAG system. Returns number of chunks processed."""
    if 'rag_system' not in st.session_state:
        st.error("RAG system not initialized.")
        return 0
    try:
        new_sources = [s for s in sources if s not in st.session_state.document_sources]
        if not new_sources:
            st.info("All provided sources have already been processed.")
            return 0

        logger.info(f"Loading content for sources: {new_sources}")
        with st.spinner("Processing documents... This may take a while."):
            processed_chunks = st.session_state.rag_system.load_content(new_sources)
        st.session_state.document_sources.update(new_sources)
        logger.info(f"Processed {len(processed_chunks)} chunks from {len(new_sources)} new sources.")
        return len(processed_chunks)
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        st.error(f"Error processing documents: {str(e)}")
        return 0

def display_sources_summary(sources_info: List[Dict]):
    """Display source information cited in the chat response expander."""
    if not sources_info:
        st.write("No specific sources were cited for this response.")
        return

    for idx, source_data in enumerate(sources_info, 1):
        source_loc = source_data.get('source', 'Unknown')
        source_type = source_data.get('type', 'N/A').upper()
        content_preview = source_data.get('content', 'No preview available.')
        relevance = source_data.get('relevance', 'N/A')

        st.markdown(f"""
        <div class="reference-container">
            <div class="reference-header">
                Source {idx}: {source_loc} <span class="source-citation">{source_type}</span>
            </div>
            <div class="reference-content">
                {content_preview}
            </div>
            <div class="reference-metadata">
                Cited Relevance: {relevance if isinstance(relevance, (float, int)) else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_retrieved_documents_sidebar():
    """Display retrieved documents (context) for the last query in the sidebar."""
    if not st.session_state.get("retrieved_documents"):
        return

    with st.sidebar.expander("📑 Context Used for Last Answer", expanded=False):
        if not st.session_state.retrieved_documents:
             st.info("No documents were retrieved or processed for the last query.")
             return

        for idx, doc_info in enumerate(st.session_state.retrieved_documents, 1):
            source = doc_info.get('source', 'Unknown Source')
            doc_type = doc_info.get('type', 'Unknown').upper()
            content = doc_info.get('content', 'No content available')
            relevance = doc_info.get('relevance', 'N/A')

            st.markdown(f"""
            <div class="reference-container">
                <div class="reference-header">
                   Context Chunk {idx}: {source} <span class="source-citation">{doc_type}</span>
                </div>
                <div class="reference-content">
                    {content}
                </div>
                <div class="reference-metadata">
                    Retrieved Relevance: {relevance if isinstance(relevance, (float, int)) else 'N/A'}
                </div>
            </div>
            """, unsafe_allow_html=True)


def clear_system():
    """Clear the RAG system index and reset session state related to data."""
    if 'rag_system' not in st.session_state:
        st.warning("RAG system not initialized, nothing to clear.")
        st.session_state.document_sources = set()
        st.session_state.retrieved_documents = []
        st.session_state.messages = []
        return

    try:
        with st.spinner("Clearing Pinecone index and local data..."):
            st.session_state.rag_system.clear_index()
        st.session_state.document_sources = set()
        st.session_state.retrieved_documents = []
        st.session_state.messages = []
        st.success("✅ System index and chat history cleared successfully!")
    except Exception as e:
        logger.error(f"Error clearing system: {str(e)}", exc_info=True)
        st.error(f"Error clearing system: {str(e)}")

def handle_load_documents_click():
    """
    Callback function executed when 'Load Documents' is clicked.
    Processes documents and clears the input area upon success.
    Executes *before* the script reruns.
    """
    input_text = st.session_state.sources_input_area
    if not (input_text and input_text.strip()):
        st.warning("Please enter at least one URL or file path.")
        return

    all_entries = [url.strip() for url in input_text.split('\n') if url.strip()]
    valid_sources = [s for s in all_entries if s.startswith('http') or os.path.exists(s)]
    invalid_sources = [s for s in all_entries if not (s.startswith('http') or os.path.exists(s))]

    if invalid_sources:
         st.toast(f"Ignoring invalid/non-existent paths: {', '.join(invalid_sources)}", icon="⚠️")

    if not valid_sources:
        st.warning("No valid new URLs or existing file paths provided.")
        return

    num_chunks = process_and_store_documents(valid_sources)

    if num_chunks > 0:
        st.toast(f"✅ Processed {num_chunks} chunks from {len(valid_sources)} new source(s).", icon="🎉")
        st.session_state.sources_input_area = ""
    elif num_chunks == 0 and valid_sources:
         pass


def main():
    st.set_page_config(
        page_title="Synergistic   RAG System",
        page_icon="🤖",
        layout="wide"
    )

    load_css()
    initialize_session_state()

    st.title("🤖 Synergistic  RAG System")
    st.markdown("---")

    with st.sidebar:
        st.markdown("### 📚 Document Management")

        st.text_area(
            "Enter URLs or local file paths (one per line):",
            placeholder="https://example.com/doc1\n/path/to/your/file.pdf",
            height=150,
            key="sources_input_area"
        )

        col_process, col_clear = st.columns(2)
        with col_process:
            st.button(
                "➕ Load Documents",
                key="load_docs_button",
                on_click=handle_load_documents_click
            )

        with col_clear:
            if st.button("🧹 Clear All Data", key="clear_system_button"):
                 st.session_state.confirm_clear = True

        if st.session_state.get('confirm_clear', False):
            st.warning("**Are you sure you want to clear the index and chat history?** This cannot be undone.")
            c1, c2 = st.columns(2)
            if c1.button("Yes, Clear Everything", key="confirm_clear_yes"):
                clear_system()
                st.session_state.confirm_clear = False
                st.rerun()
            if c2.button("Cancel", key="confirm_clear_cancel"):
                st.session_state.confirm_clear = False
                st.rerun()


        st.markdown("### 📊 System Status")
        if 'rag_system' in st.session_state:
            if st.session_state.document_sources:
                st.write(f"Loaded Sources: {len(st.session_state.document_sources)}")
                with st.expander("View Loaded Source List"):
                    for source in sorted(list(st.session_state.document_sources)):
                        st.markdown(f"- `{source}`")
            else:
                st.info("No documents loaded yet.")
        else:
             st.error("RAG system not available.")

        display_retrieved_documents_sidebar()


    st.markdown("### 💬 Query Interface")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View Sources Cited in this Response"):
                    display_sources_summary(message["sources"])

    if query := st.chat_input("Ask a question about the loaded documents..."):
        st.session_state.messages.append({"role": "user", "content": query})

        if 'rag_system' not in st.session_state:
             st.error("Cannot process query: RAG system is not initialized.", icon="🚨")
        elif not st.session_state.document_sources:
            st.warning("⚠️ Please load documents using the sidebar before asking questions.", icon="⚠️")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I need documents to answer questions. Please load some using the 'Load Documents' button in the sidebar.",
                "sources": []
            })
            st.rerun()
        else:
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                sources = []
                answer = ""

                with st.spinner("🧠 Thinking... (Performing retrieval, CRAG, and generation)"):
                    try:
                        answer, sources = st.session_state.rag_system.query(query)
                        logger.info(f"Query '{query}' processed. Answer received. Sources returned: {len(sources)}")

                        st.session_state.retrieved_documents = sources

                        message_placeholder.markdown(answer, unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })


                    except Exception as e:
                        logger.error(f"Error processing query '{query}': {str(e)}", exc_info=True)
                        error_message = f"❌ Error processing query: {str(e)}"
                        message_placeholder.error(error_message)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "sources": []
                        })
                        st.session_state.retrieved_documents = []


if __name__ == "__main__":
    main()
