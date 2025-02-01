import os
import getpass
import time
import logging
from typing import List, Union
from operator import itemgetter
import hashlib

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
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi

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
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.dimension = dimension
        self.setup_pinecone()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        self.retriever = self.vector_store.as_retriever()
        self.bm25_corpus = []
        self.bm25 = None
        self.llm = OllamaLLM(model=self.llm_model)
        self.setup_rag_chain()

    def setup_pinecone(self):
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
                while not self.pc.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)
                    logger.info("Waiting for index to be ready...")

            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            raise Exception(f"Failed to initialize Pinecone: {str(e)}")

    def load_web_content(self, urls: List[str]) -> List[Document]:
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

    # Modified to handle both Document objects and strings
    def format_docs(self, docs: List[Union[Document, str]]) -> str:
        formatted_docs = []
        for doc in docs:
            if isinstance(doc, Document):
                formatted_docs.append(doc.page_content)
            elif isinstance(doc, str):
                formatted_docs.append(doc)
            else:
                logger.warning(f"Skipping document of unexpected type: {type(doc)}")
        
        return "\n\n".join(formatted_docs)

    def bm25_retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.bm25:
            logger.warning("BM25 index not initialized.")
            return []
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [Document(page_content=self.bm25_corpus[idx]) for idx, _ in top_docs]

    def hybrid_retrieve_for_query(self, query: str) -> List[Document]:
        logger.info(f"Hybrid retrieval for query: '{query}'")
        pinecone_results = self.retriever.invoke(query)
        logger.info(f"Pinecone retrieved {len(pinecone_results)} documents for query: '{query}'")
        
        # Added explicit conversion of Pinecone results to Documents
        pinecone_results = [
            Document(page_content=result.page_content) if isinstance(result, Document) 
            else Document(page_content=result) if isinstance(result, str)
            else result
            for result in pinecone_results
        ]
        
        bm25_results = self.bm25_retrieve(query)
        logger.info(f"BM25 retrieved {len(bm25_results)} documents for query: '{query}'")
        fused = self.reciprocal_rank_fusion([pinecone_results, bm25_results])
        logger.info(f"Fused result contains {len(fused)} documents for query: '{query}'")
        return fused

    # Modified to handle document type conversion and validation
    @staticmethod
    def reciprocal_rank_fusion(results: List[List[Union[Document, str]]], k: int = 60) -> List[Document]:
        fused_scores = {}
        doc_map = {}

        for result_list in results:
            for rank, doc in enumerate(result_list):
                if not isinstance(doc, (Document, str)):
                    logger.warning(f"Skipping invalid document type: {type(doc)}")
                    continue
                
                if isinstance(doc, str):
                    doc = Document(page_content=doc)

                key = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                if key not in fused_scores:
                    fused_scores[key] = 0
                    doc_map[key] = doc
                fused_scores[key] += 1 / (rank + k)

        reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, score in reranked]

    def setup_rag_chain(self):
        # Modified query rewrite template to be more direct
        query_rewrite_template = """You are a helpful assistant that rewrites queries to make them clearer and more specific.
Given the query below, rewrite it to be more detailed and search-friendly. Do not ask for clarification - just improve the query.

Original Query: {question}

Rewritten Query: """
        
        prompt_rewrite = ChatPromptTemplate.from_template(query_rewrite_template)
        rewrite_query_chain = prompt_rewrite | self.llm | StrOutputParser()

        # Modified multi-query template to focus on generating related queries
        multi_query_generation_template = """Given the search query below, generate 4 different but related search queries that could help find relevant information.
Each query should explore a different aspect of the topic. Output exactly 4 queries, one per line.

Search Query: {rewritten_query}

Queries:"""
        
        prompt_multi_query = ChatPromptTemplate.from_template(multi_query_generation_template)
        multi_query_chain = (
            prompt_multi_query
            | self.llm
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.split('\n') if q.strip() and not q.startswith('Queries:')])
        )

        def generate_queries(query_dict: dict):
            rewritten_query = rewrite_query_chain.invoke(query_dict)
            logger.info(f"Rewritten Query: {rewritten_query}")
            raw_queries = multi_query_chain.invoke({"rewritten_query": rewritten_query})
            logger.info(f"Raw Generated Queries: {raw_queries}")
            queries = [q for q in raw_queries if not q.lower().startswith(('sure', 'please', 'queries', 'query'))]
            if not queries:
                logger.warning("No valid queries generated, using original query")
                return [query_dict["question"]]
            logger.info(f"Valid Generated Queries: {queries}")
            return queries

        def retrieve_all(queries):
            all_results = []
            for query in queries:
                logger.info(f"Retrieving documents for sub-query: '{query}'")
                results = self.hybrid_retrieve_for_query(query)
                if results:  # Only add non-empty result lists
                    all_results.append(results)
            return all_results

        def fuse_results(query_dict: dict):
            queries = generate_queries(query_dict)
            logger.info(f"Generated queries: {queries}")
            retrieved_lists = retrieve_all(queries)
            if not retrieved_lists:
                logger.warning("No results found for any query")
                return []
            fused_docs = self.reciprocal_rank_fusion(retrieved_lists)
            logger.info(f"Total fused documents after RRF: {len(fused_docs)}")
            return fused_docs

        final_template = """You are an expert assistant tasked with providing detailed, accurate, and well-structured answers based on the given context. If you don't find relevant information in the context, say "I don't have enough information to answer this question accurately."

Context:
{context}

Question: {question}

Instructions:
1. If the context contains relevant information, provide a detailed answer (200+ words)
2. Include specific examples and details from the context
3. Structure your answer with clear sections
4. Use natural language and avoid unnecessary jargon
5. Begin with a clear introduction and end with a key takeaway

Answer:"""

        prompt_final = ChatPromptTemplate.from_template(final_template)

        context_chain = (
            RunnablePassthrough(fuse_results) 
            | (lambda docs: self.format_docs(docs) if isinstance(docs, list) else str(docs))
        )

        self.final_rag_chain = (
            {
                "context": context_chain,
                "question": itemgetter("question"),
            }
            | prompt_final
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> str:
        try:
            logger.info(f"Original Question: {question}")
            answer = self.final_rag_chain.invoke({"question": question})
            logger.info(f"Final Answer: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error: {str(e)}")
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