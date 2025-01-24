import streamlit as st
from agents.orchestrator import Orchestrator
from config.load_config import load_config

config = load_config()
orchestrator = Orchestrator()

st.markdown("""
    <style>
    .stChatInput {position: fixed; bottom: 20px; width: 70%;}
    .stMarkdown {padding: 10px; border-radius: 5px;}
    div[data-testid="stExpander"] {border: 1px solid rgba(49, 51, 63, 0.2);}
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("Settings ⚙️")
    web_search_enabled = st.checkbox("Enable Web Search", value=False)
    st.markdown("---")
    st.markdown("**Models**")
    st.write(f"LLM: {config['ollama']['model']}")
    st.write(f"Embeddings: {config['embedding']['model']}")

st.title("🔍 Open Source RAG Explorer")
st.caption("Powered by Ollama + Qdrant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            thought_process, final_answer = orchestrator.run_with_thought_process(user_query, web_search_enabled)
            st.markdown("### 🤔 Thought Process")
            for step in thought_process:
                with st.expander(f"**Step {step['step']}**: {step['description']}"):
                    st.write(step["details"])
            st.markdown("### 🎯 Final Answer")
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})