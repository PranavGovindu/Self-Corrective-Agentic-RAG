# ui.py
import streamlit as st
from vector import RAGS
import logging
import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_css():
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chat-container {
            border-radius: 10px;
            padding: 20px;
            background-color: #f5f5f5;
            margin: 10px 0;
        }
        .metadata-container {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        .source-citation {
            background-color: #e6f3ff;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85em;
            margin: 0 2px;
            cursor: pointer;
        }
        .reference-container {
            border-left: 3px solid #0066cc;
            padding: 10px;
            margin: 10px 0;
            background-color: #f8f9fa;
        }
        .reference-header {
            font-weight: bold;
            color: #0066cc;
            margin-bottom: 5px;
        }
        .reference-content {
            font-size: 0.9em;
            color: #333;
            max-height: 150px;
            overflow-y: auto;
        }
        .reference-metadata {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        .stSpinner > div {
            text-align: center;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all required session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "rag_system" not in st.session_state:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        st.session_state.rag_system = RAGS(tavily_api_key=tavily_api_key)
        
    if "document_sources" not in st.session_state:
        st.session_state.document_sources = set()
        
    if "retrieved_documents" not in st.session_state:
        st.session_state.retrieved_documents = []

def process_and_store_documents(sources: List[str]) -> List[Dict]:
    """Process documents and store them in the RAG system"""
    try:
        documents = st.session_state.rag_system.load_content(sources)
        st.session_state.document_sources.update(sources)
        return documents
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return []

def display_sources(sources: List[Dict]):
    """Display source information in a structured format"""
    if not sources:
        return
        
    with st.expander("📚 Detailed Source Information", expanded=False):
        for idx, source in enumerate(sources, 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Source:** `{source.get('source', 'Unknown')}`")
                st.markdown(f"**Type:** `{source.get('type', 'unknown').upper()}`")
            with col2:
                if isinstance(source.get('relevance'), float):
                    st.metric("Relevance Score", f"{source['relevance']:.2f}")
            
            st.markdown("**Content Preview:**")
            st.markdown(f'<div class="reference-content">{source.get("content", "No preview available")}</div>', 
                       unsafe_allow_html=True)
            st.markdown("---")

def display_retrieved_documents():
    """Display retrieved documents with their metadata"""
    if not st.session_state.retrieved_documents:
        return
        
    with st.expander("📑 Retrieved Context Chunks (Used for Answer)", expanded=True):
        for doc in st.session_state.retrieved_documents:
            source = doc.get('source', 'Unknown Source')
            doc_type = doc.get('type', 'Unknown').upper()
            content = doc.get('content', 'No content available')
            relevance = doc.get('relevance', 'N/A')

            st.markdown(f"""
            <div class="reference-container">
                <div class="reference-header">
                    {source} <span class="source-citation">{doc_type}</span>
                </div>
                <div class="reference-content">
                    {content}
                </div>
                <div class="reference-metadata">
                    Relevance Score: {relevance if isinstance(relevance, float) else 'N/A'}
                </div>
            </div>
            """, unsafe_allow_html=True)

def clear_system():
    """Clear the RAG system and reset all session state"""
    try:
        st.session_state.rag_system.clear_index()
        st.session_state.document_sources.clear()
        st.session_state.retrieved_documents = []
        st.session_state.messages = []
        st.success("✅ System cleared successfully")
    except Exception as e:
        st.error(f"Error clearing system: {str(e)}")

def main():
    st.set_page_config(
        page_title="Advanced RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    load_css()
    initialize_session_state()
    
    st.title("🤖 Advanced RAG System")
    st.markdown("---")
    
    # Split the interface into two columns
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 📚 Document Management")
        
        urls_input = st.text_area(
            "Enter URLs or file paths (one per line)",
            placeholder="https://example.com/doc1\n/path/to/file.pdf",
            height=100
        )
        
        col_process, col_clear = st.columns(2)
        with col_process:
            if st.button("🚀 Load Documents"):
                if urls_input.strip():
                    sources = [url.strip() for url in urls_input.split('\n') if url.strip()]
                    with st.spinner("Processing documents..."):
                        processed_docs = process_and_store_documents(sources)
                        if processed_docs:
                            st.success(f"✅ Processed {len(processed_docs)} chunks from {len(sources)} sources")
                else:
                    st.warning("Please enter at least one URL or file path")
                    
        with col_clear:
            if st.button("🧹 Clear System"):
                clear_system()
        
        st.markdown("### 📊 System Status")
        if st.session_state.document_sources:
            st.write(f"Loaded Sources: {len(st.session_state.document_sources)}")
            with st.expander("View Source List"):
                for source in sorted(st.session_state.document_sources):
                    st.write(f"- `{source}`")
        else:
            st.info("No documents loaded yet")

    with col1:
        st.markdown("### 💬 Query Interface")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("View Sources for this Response"):
                        display_sources(message["sources"])
        
        # Display retrieved documents
        display_retrieved_documents()
        
        # Query input and processing
        if query := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("assistant"):
                if not st.session_state.document_sources:
                    st.warning("⚠️ Please load documents first!")
                    return
                    
                with st.spinner("🔍 Analyzing documents..."):
                    try:
                        # Get response and sources from RAG system
                        answer, sources = st.session_state.rag_system.query(query)
                        
                        # Store retrieved documents for display
                        st.session_state.retrieved_documents = sources
                        
                        # Display the response with citations
                        st.markdown(answer, unsafe_allow_html=True)
                        
                        # Store message with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
