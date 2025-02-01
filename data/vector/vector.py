import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import getpass
import time
from typing import List, Optional
import logging

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(
        self,
        index_name: str = "rag",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "qwen2.5-coder:1.5b",
        dimension: int = 1024,
    ):
        """Initialize RAG system with Pinecone, HuggingFace embeddings, and Ollama."""
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.dimension = dimension
        
        # Initialize Pinecone
        self.setup_pinecone()
        
        # Initialize embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        self.retriever = self.vector_store.as_retriever()
        
        # Initialize LLM
        self.llm = OllamaLLM(model=self.llm_model)
        
        # Setup RAG chain
        self.setup_rag_chain()

    def setup_pinecone(self):
        """Initialize Pinecone and create index if it doesn't exist."""
        if not os.getenv("PINECONE_API_KEY"):
            os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
        
        try:
            self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)
                    logger.info("Waiting for index to be ready...")
            
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            raise Exception(f"Failed to initialize Pinecone: {str(e)}")

    def load_web_content(self, urls: List[str]) -> List[Document]:
        """Load and process content from web URLs."""
        try:
            loader = WebBaseLoader(
                web_paths=urls,
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            docs = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            
            # Add documents to vector store
            self.vector_store.add_documents(splits)
            
            return splits
        
        except Exception as e:
            raise Exception(f"Failed to load web content: {str(e)}")

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def reciprocal_rank_fusion(results: list[list], k=60):
        """Reciprocal Rank Fusion (RRF) to rerank multiple lists of documents."""
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
        
        # Sort by fused scores in descending order
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results

    def setup_rag_chain(self):
        """Setup the RAG chain with RRF-based retrieval."""
        # Step 1: Generate multiple queries
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_rag_fusion 
            | self.llm
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

        # Step 2: Retrieve documents for each query and apply RRF
        retrieval_chain_rag_fusion = (
            generate_queries 
            | (lambda queries: [self.retriever.invoke(query) for query in queries]) 
            | self.reciprocal_rank_fusion
        )

        # Step 3: Setup final RAG chain
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        self.final_rag_chain = (
            {"context": retrieval_chain_rag_fusion | (lambda docs: self.format_docs([doc[0] for doc in docs])), 
             "question": itemgetter("question")} 
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        try:
            # Log the original question
            logger.info(f"Original Question: {question}")
            
            # Step 1: Generate multiple queries
            template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
            Generate multiple search queries related to: {question} \n
            Output (4 queries):"""
            prompt_rag_fusion = ChatPromptTemplate.from_template(template)

            generate_queries = (
                prompt_rag_fusion 
                | self.llm
                | StrOutputParser() 
                | (lambda x: x.split("\n"))
            )
            rewritten_queries = generate_queries.invoke({"question": question})
            logger.info(f"Rewritten Queries: {rewritten_queries}")
            
            # Step 2: Retrieve documents for each query and apply RRF
            retrieval_chain_rag_fusion = (
                generate_queries 
                | (lambda queries: [self.retriever.invoke(query) for query in queries]) 
                | self.reciprocal_rank_fusion
            )
            reranked_docs = retrieval_chain_rag_fusion.invoke({"question": question})
            logger.info(f"Reranked Documents: {reranked_docs}")
            
            # Step 3: Generate the final answer
            template = """You are an expert assistant tasked with providing detailed, accurate, and well-structured answers to questions based on the given context. Follow these guidelines:

                    1. **Understand the Context**: Carefully analyze the provided context to ensure your answer is relevant and accurate.
                    2. **Answer in Detail**: Provide a comprehensive answer with a minimum of 200 words. Include examples, explanations, and supporting details where applicable.
                    3. **Structure Your Answer**:
                       - Start with a brief introduction summarizing the key points.
                       - Use bullet points or numbered lists for clarity if the answer involves steps, features, or categories.
                       - Conclude with a summary or key takeaway.
                    4. **Be Clear and Concise**: Avoid unnecessary jargon or overly complex language. Ensure the answer is easy to understand.
                    5. **Cite the Context**: If specific details from the context are used, mention them explicitly to support your answer.

                    Context:
                    {context}

                    Question: {question}

                    """
            prompt = ChatPromptTemplate.from_template(template)

            final_rag_chain = (
                {"context": retrieval_chain_rag_fusion | (lambda docs: self.format_docs([doc[0] for doc in docs])), 
                 "question": itemgetter("question")} 
                | prompt
                | self.llm
                | StrOutputParser()
            )
            answer = final_rag_chain.invoke({"question": question})
            logger.info(f"Final Answer: {answer}")
            
            return answer
        
        except Exception as e:
            raise Exception(f"Failed to process query: {str(e)}")
    
    

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem()
    
    # Load content
    urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    docs = rag.load_web_content(urls)
    logger.info(f"Loaded {len(docs)} document chunks")
    
    # Query the system
    question = "What is Task Decomposition?"
    try:
        answer = rag.query(question)
        logger.info(f"\nQuestion: {question}")
        logger.info(f"Answer: {answer}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")