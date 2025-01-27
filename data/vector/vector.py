import os
import typer
from typing import Optional, List
from rich.prompt import Prompt
import streamlit as st
from datetime import datetime
from textwrap import dedent
import subprocess

from phi.agent import Agent
from phi.tools.exa import ExaTools
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.knowledge.text import TextKnowledgeBase
from phi.vectordb.pineconedb import PineconeDB

api_key = os.getenv("PINECONE_API_KEY")
index_name = "agentiv-rag"

vector_db = PineconeDB(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    api_key=api_key,
    use_hybrid_search=True,
    hybrid_alpha=0.5,
)

def load_documents(doc_urls: List[str]):
    pdf_urls = [url for url in doc_urls if url.endswith('.pdf')]
    text_urls = [url for url in doc_urls if url.endswith('.txt')]

    knowledge_base = None
    
    if pdf_urls:
        knowledge_base = PDFUrlKnowledgeBase(
            urls=pdf_urls,
            vector_db=vector_db,
        )

    if text_urls:
        knowledge_base = TextKnowledgeBase(
            urls=text_urls,
            vector_db=vector_db,
        )

    if knowledge_base:
        knowledge_base.load(recreate=True, upsert=True)

    return knowledge_base

def run_ollama(prompt: str):
    result = subprocess.run(
        ["ollama", "generate", "--model", "llama2", "--text", prompt],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def pinecone_agent(user: str = "user", doc_urls: List[str] = []):
    run_id: Optional[str] = None

    knowledge_base = load_documents(doc_urls)

    agent = Agent(
        model=None,  # Use local model instead of OpenAIChat
        tools=[ExaTools(start_published_date=datetime.now().strftime("%Y-%m-%d"), type="keyword")],
        description="You are an advanced AI researcher writing a report on a topic.",
        instructions=[
            "For the provided topic, run 3 different searches.",
            "Read the results carefully and prepare a NYT worthy report.",
            "Focus on facts and make sure to provide references.",
        ],
        expected_output=dedent("""\
        An engaging, informative, and well-structured report in markdown format:

        ## Engaging Report Title

        ### Overview
        {give a brief introduction of the report and why the user should read this report}
        {make this section engaging and create a hook for the reader}

        ### Section 1
        {break the report into sections}
        {provide details/facts/processes in this section}

        ... more sections as necessary...

        ### Takeaways
        {provide key takeaways from the article}

        ### References
        - [Reference 1](link)
        - [Reference 2](link)
        - [Reference 3](link)

        - published on {date} in dd/mm/yyyy
        """),
        markdown=True,
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        save_response_to_file="tmp/{message}.md",
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        response = run_ollama(message)
        agent.print_response(response)

def streamlit_ui():
    st.title("Agent with Multiple Document Types")

    uploaded_files = st.file_uploader("Upload your documents (PDF/Text)", accept_multiple_files=True)
    
    if uploaded_files:
        doc_urls = []

        for file in uploaded_files:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            doc_urls.append(f"file://{file_path}")

        query = st.text_input("Ask your question:")

        if query:
            st.subheader("RAG Response")
            pinecone_agent(user="user", doc_urls=doc_urls)

            with st.expander("Web Search Response"):
                web_response = run_ollama(f"Search for: {query}")
                st.write(web_response)

if __name__ == "__main__":
    if "streamlit" in os.environ.get("WERKZEUG_RUN_MAIN", ""):
        streamlit_ui()
    else:
        typer.run(pinecone_agent)
