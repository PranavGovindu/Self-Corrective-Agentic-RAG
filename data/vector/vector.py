import os
import getpass
import time
import logging
from typing import List
from operator import itemgetter

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.load import dumps, loads
from pinecone import Pinecone, ServerlessSpec

# For BM25 retrieval
from rank_bm25 import BM25Okapi

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGS:
    def __init__(
        self,
        index_name: str = "rag",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "qwen2.5-coder:1.5b",
        dimension: int = 1024,
    ):
        """Initialize RAG system with Pinecone, BM25, and Ollama."""
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.dimension = dimension

        # Initialize Pinecone
        self.setup_pinecone()

        # Initialize embeddings and vector store (dense search)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        self.retriever = self.vector_store.as_retriever()

        # Initialize BM25 components (sparse search)
        self.bm25_corpus = []  # list of document texts
        self.bm25 = None

        # Initialize LLM
        self.llm = OllamaLLM(model=self.llm_model)

        # Setup RAG chain (includes query rewriting and final answer generation)
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

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            logger.info(f"Adding {len(splits)} document chunks to Pinecone vector store.")
            self.vector_store.add_documents(splits)

            self.bm25_corpus = [doc.page_content for doc in splits]
            self.bm25 = BM25Okapi([doc.split() for doc in self.bm25_corpus])

            logger.info("BM25 corpus and index built.")
            return splits

        except Exception as e:
            raise Exception(f"Failed to load web content: {str(e)}")

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def bm25_retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using BM25 for a single query."""
        if not self.bm25:
            logger.warning("BM25 index not initialized.")
            return []
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        logger.info(f"BM25 retrieved {len(top_docs)} documents for query: '{query}'")
        return [Document(page_content=self.bm25_corpus[idx]) for idx, _ in top_docs]

    def hybrid_retrieve_for_query(self, query: str) -> List[Document]:
        """Retrieve documents using both Pinecone and BM25 for a given query and fuse the results."""
        logger.info(f"Hybrid retrieval for query: '{query}'")
        pinecone_results = self.retriever.invoke(query)
        logger.info(f"Pinecone retrieved {len(pinecone_results)} documents for query: '{query}'")
        bm25_results = self.bm25_retrieve(query)
        logger.info(f"BM25 retrieved {len(bm25_results)} documents for query: '{query}'")
        fused = self.reciprocal_rank_fusion([pinecone_results, bm25_results])
        logger.info(f"Fused result contains {len(fused)} documents for query: '{query}'")
        return fused

    @staticmethod
    def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60) -> List[Document]:
        """Reciprocal Rank Fusion (RRF) to rerank multiple lists of documents."""
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
        reranked_docs = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return [doc for doc, _ in reranked_docs]

    def setup_rag_chain(self):
        """Setup the RAG chain with query rewriting, multi-query generation, hybrid retrieval, and final answer generation."""
        # Step 1: Query Rewriting
        query_rewrite_template = (
            "You are a helpful assistant that rewrites queries to make them clearer and more precise.\n"
            "Rewrite the following query:\n\n"
            "Original Query: {question}\n"
            "Rewritten Query:"
        )
        prompt_rewrite = ChatPromptTemplate.from_template(query_rewrite_template)
        rewrite_query_chain = prompt_rewrite | self.llm | StrOutputParser()

        # Step 2: Multi-Query Generation (using the rewritten query)
        multi_query_generation_template = (
            "You are an assistant that generates multiple search queries based on a given rewritten query.\n"
            "Generate 4 distinct search queries related to:\n\n"
            "Rewritten Query: {rewritten_query}\n"
            "Output (one query per line):"
        )
        prompt_multi_query = ChatPromptTemplate.from_template(multi_query_generation_template)
        multi_query_chain = (
            prompt_multi_query
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        # Combine both steps into a single chain
        # Assign the chain to generate_queries so it can be invoked later.
        generate_queries = {"rewritten_query": rewrite_query_chain} | multi_query_chain

        def retrieve_all(queries):
            all_results = []
            for query in queries:
                logger.info(f"Retrieving documents for sub-query: '{query}'")
                results = self.hybrid_retrieve_for_query(query)
                all_results.append(results)
            return all_results

        def fuse_results(query_dict: dict):
            # Generate multiple queries from the original question
            queries = generate_queries.invoke(query_dict)
            logger.info(f"Rewritten and generated queries: {queries}")
            retrieved_lists = retrieve_all(queries)
            fused_docs = self.reciprocal_rank_fusion(retrieved_lists)
            logger.info(f"Total fused documents after RRF: {len(fused_docs)}")
            return fused_docs

        final_template = (
            """
You are an expert assistant tasked with providing detailed, accurate, and well-structured answers to questions based on the given context. Follow these guidelines:
If you don't know the answer, just say that you don't know. 
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
        )
        prompt_final = ChatPromptTemplate.from_template(final_template)

        # Create final chain that fuses the results and then generates an answer.
        self.final_rag_chain = (
            {
                "context": fuse_results | (lambda docs: self.format_docs(docs)),
                "question": itemgetter("question"),
            }
            | prompt_final
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        try:
            logger.info(f"Original Question: {question}")
            answer = self.final_rag_chain.invoke({"question": question})
            logger.info(f"Final Answer: {answer}")
            return answer
        except Exception as e:
            raise Exception(f"Failed to process query: {str(e)}")


if __name__ == "__main__":
    rag = RAGS()
    urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    docs = rag.load_web_content(urls)
    logger.info(f"Loaded {len(docs)} document chunks")

    question = "What is Task Decomposition?"
    try:
        answer = rag.query(question)
        logger.info(f"\nQuestion: {question}\nAnswer: {answer}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
