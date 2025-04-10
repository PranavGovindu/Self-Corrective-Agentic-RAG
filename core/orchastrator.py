# ui.py
import streamlit as st
# Ensure vector module is importable (e.g., same directory or in sys.path)
from vector import RAGS
import logging
import os
from typing import Dict, List, Set
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging for Streamlit app if needed
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_css():
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem;
        }
        .chat-container {
            border-radius: 10px;
            padding: 15px; /* Slightly reduced padding */
            background-color: #f8f9fa; /* Lighter background */
            margin: 10px 0;
            border: 1px solid #e9ecef; /* Subtle border */
        }
        .metadata-container {
            font-size: 0.8em;
            color: #6c757d; /* Adjusted color */
            margin-top: 5px;
        }
        .source-citation {
            background-color: #e6f3ff;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85em;
            margin: 0 2px;
            cursor: pointer;
            border: 1px solid #b8dfff; /* Add subtle border */
            display: inline-block; /* Ensure proper spacing */
        }
        .reference-container {
            border-left: 3px solid #007bff; /* Updated color */
            padding: 10px;
            margin: 10px 0;
            background-color: #f8f9fa;
            border-radius: 5px; /* Added border radius */
        }
        .reference-header {
            font-weight: bold;
            color: #0056b3; /* Darker blue */
            margin-bottom: 5px;
            word-wrap: break-word; /* Prevent long source names from overflowing */
        }
        .reference-content {
            font-size: 0.9em;
            color: #343a40; /* Darker text */
            max-height: 150px;
            overflow-y: auto;
            background-color: #ffffff; /* White background for contrast */
            padding: 5px;
            border-radius: 3px;
            border: 1px solid #dee2e6;
        }
        .reference-metadata {
            font-size: 0.8em;
            color: #6c757d;
            margin-top: 5px;
        }
        .stSpinner > div {
            text-align: center;
            margin: 20px 0;
        }
        /* Ensure text area has reasonable height */
        .stTextArea textarea {
             min-height: 100px !important;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all required session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_system" not in st.session_state:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY") # Ensure Pinecone key is available too

        if not tavily_api_key:
             logger.warning("Tavily API key not found in environment. Web search will be disabled.")
             # Optionally display a warning in the UI:
             # st.warning("Tavily API key missing. Web search disabled.", icon="⚠️")
        if not pinecone_api_key:
             logger.error("Pinecone API key not found in environment. RAG system cannot be initialized.")
             # Display error and stop execution if Pinecone is essential
             st.error("Pinecone API key missing. Please set the PINECONE_API_KEY environment variable.", icon="🚨")
             st.stop() # Stop the app if Pinecone connection fails

        try:
            # Pass Tavily key; RAGS init handles Pinecone key internally
            st.session_state.rag_system = RAGS(tavily_api_key=tavily_api_key)
            logger.info("RAG system initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
            st.error(f"Failed to initialize RAG system: {e}", icon="🚨")
            st.stop() # Stop if RAGS initialization fails

    if "document_sources" not in st.session_state:
        st.session_state.document_sources: Set[str] = set() # Use type hint

    if "retrieved_documents" not in st.session_state:
        # This will store the source info dictionaries for the latest query
        st.session_state.retrieved_documents: List[Dict] = []

def process_and_store_documents(sources: List[str]) -> int:
    """Process documents and store them in the RAG system. Returns number of chunks processed."""
    if 'rag_system' not in st.session_state:
        st.error("RAG system not initialized.")
        return 0
    try:
        # Filter out already processed sources
        new_sources = [s for s in sources if s not in st.session_state.document_sources]
        if not new_sources:
            st.info("All provided sources have already been processed.")
            return 0

        # Use load_content which returns the processed splits/chunks
        processed_chunks = st.session_state.rag_system.load_content(new_sources)
        st.session_state.document_sources.update(new_sources) # Add successfully processed sources
        logger.info(f"Processed {len(processed_chunks)} chunks from {len(new_sources)} new sources.")
        return len(processed_chunks)
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        st.error(f"Error processing documents: {str(e)}")
        return 0

def display_sources_summary(sources_info: List[Dict]):
    """Display source information in the chat response expander."""
    if not sources_info:
        st.write("No specific sources were cited for this response.")
        return

    for idx, source_data in enumerate(sources_info, 1):
        source_loc = source_data.get('source', 'Unknown')
        source_type = source_data.get('type', 'N/A').upper()
        content_preview = source_data.get('content', 'No preview available.')
        relevance = source_data.get('relevance', 'N/A') # Might be None or a score

        st.markdown(f"""
        <div class="reference-container">
            <div class="reference-header">
                Source {idx}: {source_loc} <span class="source-citation">{source_type}</span>
            </div>
            <div class="reference-content">
                {content_preview}
            </div>
            <div class="reference-metadata">
                Cited Relevance Score: {relevance if isinstance(relevance, (float, int)) else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)

# This function remains largely the same, used for the sidebar potentially
def display_retrieved_documents_sidebar():
    """Display retrieved documents with their metadata (e.g., in sidebar or dedicated area)"""
    if not st.session_state.retrieved_documents:
        return # Don't display if empty

    with st.expander("📑 Context Used for Last Answer", expanded=False): # Start collapsed
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
        # Reset other states anyway
        st.session_state.document_sources = set()
        st.session_state.retrieved_documents = []
        st.session_state.messages = []
        return

    try:
        with st.spinner("Clearing Pinecone index and local data..."):
            st.session_state.rag_system.clear_index()
        # Reset session state variables after successful clearing
        st.session_state.document_sources = set()
        st.session_state.retrieved_documents = []
        st.session_state.messages = [] # Clear chat history as well
        st.success("✅ System index and chat history cleared successfully!")
        # Force rerun to reflect cleared state, especially messages
        st.rerun()
    except Exception as e:
        logger.error(f"Error clearing system: {str(e)}", exc_info=True)
        st.error(f"Error clearing system: {str(e)}")

def main():
    st.set_page_config(
        page_title="Advanced RAG System",
        page_icon="🤖",
        layout="wide"
    )

    load_css()
    # Initialize state, potentially stopping execution if keys are missing
    initialize_session_state()

    st.title("🤖 Advanced RAG System")
    st.markdown("---")

    # Use sidebar for document management and status
    with st.sidebar:
        st.markdown("### 📚 Document Management")

        urls_input = st.text_area(
            "Enter URLs or local file paths (one per line):",
            placeholder="https://example.com/doc1\n/path/to/your/file.pdf",
            height=150, # Increased height
            key="sources_input_area" # Added key
        )

        col_process, col_clear = st.columns(2)
        with col_process:
            if st.button("➕ Load Documents", key="load_docs_button"):
                if urls_input and urls_input.strip():
                    # Validate basic format (simple check)
                    sources = [url.strip() for url in urls_input.split('\n') if url.strip() and (url.startswith('http') or os.path.exists(url.strip()))]
                    invalid_sources = [url.strip() for url in urls_input.split('\n') if url.strip() and not (url.startswith('http') or os.path.exists(url.strip()))]

                    if invalid_sources:
                         st.warning(f"Ignoring invalid or non-existent paths: {', '.join(invalid_sources)}", icon="⚠️")

                    if sources:
                        with st.spinner("Processing documents... This may take a while."):
                            num_chunks = process_and_store_documents(sources)
                            if num_chunks > 0:
                                st.success(f"✅ Processed {num_chunks} chunks from {len(sources)} new source(s).")
                            # Message for no new sources is handled inside process_and_store_documents
                    else:
                         st.warning("No valid new URLs or existing file paths provided.")
                else:
                    st.warning("Please enter at least one URL or file path.")

        with col_clear:
            # Add confirmation for clearing
            if st.button("🧹 Clear All Data", key="clear_system_button"):
                 # Use a flag in session state to manage confirmation dialog
                 st.session_state.confirm_clear = True

        # Confirmation dialog logic
        if st.session_state.get('confirm_clear', False):
            st.warning("**Are you sure you want to clear the index and chat history?** This cannot be undone.")
            c1, c2 = st.columns(2)
            if c1.button("Yes, Clear Everything"):
                clear_system()
                st.session_state.confirm_clear = False # Reset flag
                st.rerun() # Rerun to update UI after clear
            if c2.button("Cancel"):
                st.session_state.confirm_clear = False # Reset flag
                st.rerun() # Rerun to hide confirmation


        st.markdown("### 📊 System Status")
        if 'rag_system' in st.session_state:
            if st.session_state.document_sources:
                st.write(f"Loaded Sources: {len(st.session_state.document_sources)}")
                with st.expander("View Loaded Source List"):
                    # Sort sources for consistent display
                    for source in sorted(list(st.session_state.document_sources)):
                        st.markdown(f"- `{source}`")
            else:
                st.info("No documents loaded yet.")
        else:
             st.error("RAG system not available.")

        # Display retrieved documents from the *last* query in the sidebar
        display_retrieved_documents_sidebar()


    # Main chat interface
    st.markdown("### 💬 Query Interface")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            # Display sources associated with assistant message, if any
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("View Sources Cited in this Response"):
                    display_sources_summary(message["sources"])

    # Query input and processing
    if query := st.chat_input("Ask a question about the loaded documents..."):
        # Append user message immediately
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Process query with RAG system
        if 'rag_system' not in st.session_state:
             st.error("Cannot process query: RAG system is not initialized.", icon="🚨")
        elif not st.session_state.document_sources:
            st.warning("⚠️ Please load documents using the sidebar before asking questions.", icon="⚠️")
            # Add placeholder assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I need documents to answer questions. Please load some using the 'Load Documents' button in the sidebar."
            })
            # Rerun to display the warning message immediately
            st.rerun()
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty() # Placeholder for streaming or final answer
                with st.spinner("🧠 Thinking... (Performing retrieval, CRAG, and generation)"):
                    try:
                        # Get response and source info dictionaries from RAG system
                        answer, sources = st.session_state.rag_system.query(query)

                        # Store the source info for the *last* query for sidebar display
                        st.session_state.retrieved_documents = sources

                        # Display the final answer
                        message_placeholder.markdown(answer, unsafe_allow_html=True)

                        # Store assistant message with its specific sources
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources # Store the list of source dictionaries
                        })

                        # Optionally rerun to update the sidebar immediately with new context
                        # st.rerun()

                    except Exception as e:
                        logger.error(f"Error processing query: {str(e)}", exc_info=True)
                        error_message = f"❌ Error processing query: {str(e)}"
                        message_placeholder.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()
