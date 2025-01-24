import streamlit as st
import PyPDF2
from docx import Document

st.title("AI Assistant with RAG + Web Search")

st.sidebar.title("Configuration")
retrieval_method = st.sidebar.radio("Choose retrieval method:", ["RAG + Web", "RAG Only", "Web Only"])

uploaded_file = st.file_uploader("Upload a file (optional):", type=["txt", "pdf", "docx"])
query = st.text_input("Ask your question:", placeholder="Type your query here...")

if st.button("Submit"):
    if query.strip():
        with st.spinner("Processing your query..."):
            response = "Generated response will appear here."
            sources = ["Source 1: Example.com", "Source 2: Wikipedia"]
            plan = {
                "steps": [
                    "1. Check if the uploaded file contains relevant context.",
                    "2. Search RAG for relevant information.",
                    "3. Perform web search for additional sources.",
                    "4. Combine results from RAG and web search.",
                    "5. Generate the final response."
                ]
            }

        with st.expander("Plan of Action (Click to expand)"):
            for step in plan["steps"]:
                st.markdown(f"- {step}")

        st.subheader("Response")
        st.write(response)

        st.subheader("Sources")
        for i, source in enumerate(sources, 1):
            st.markdown(f"{i}. {source}")
    else:
        st.warning("Please enter a valid query.")

if uploaded_file is not None:
    st.subheader("Uploaded File Content")
    file_extension = uploaded_file.name.split(".")[-1]

    try:
        if file_extension == "txt":
            content = uploaded_file.read().decode("utf-8")
            st.text_area("File Content Preview", value=content, height=200)
        elif file_extension == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
            st.text_area("File Content Preview", value=pdf_text, height=200)
        elif file_extension == "docx":
            doc = Document(uploaded_file)
            doc_text = "\n".join([para.text for para in doc.paragraphs])
            st.text_area("File Content Preview", value=doc_text, height=200)
        else:
            st.error("Unsupported file format.")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
