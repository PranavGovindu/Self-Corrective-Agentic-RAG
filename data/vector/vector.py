import os
import getpass
import time
import logging
from typing import List, Union
from operator import itemgetter
import hashlib
import json  # For JSON parsing

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
from langchain_community.tools.tavily_search import TavilySearchResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_rewritten_query(raw_output: str) -> str:
    """
    Parses the rewritten query from the JSON output.
    If the output is JSON with a "query" key, extract its value.
    Otherwise, return the raw output.
    """
    logger.info(f"Raw rewritten query output: {raw_output}")
    try:
        data = json.loads(raw_output)
        if isinstance(data, dict) and "query" in data:
            return data["query"]
    except json.JSONDecodeError:
        pass
    return raw_output.strip()


def parse_queries(raw_output: str) -> List[str]:
    """
    Parses the multi-query generation output.
    If the output is JSON (list or dict), extract the queries.
    This version also handles a JSON object with a 'queries' key.
    Otherwise, split by newline and filter out any numbering.
    """
    logger.info(f"Raw multi-query output: {raw_output}")
    try:
        data = json.loads(raw_output)
        if isinstance(data, list):
            # If it's a list, process each item.
            queries = []
            for item in data:
                if isinstance(item, dict) and "query" in item:
                    queries.append(item["query"])
                elif isinstance(item, str):
                    queries.append(item)
            return queries
        elif isinstance(data, dict):
            # Check for "query" key
            if "query" in data and isinstance(data["query"], str):
                return [data["query"]]
            # Check for "queries" key
            if "queries" in data and isinstance(data["queries"], list):
                queries = []
                for item in data["queries"]:
                    if isinstance(item, str):
                        queries.append(item)
                    elif isinstance(item, dict) and "query" in item:
                        queries.append(item["query"])
                return queries
    except json.JSONDecodeError:
        # Fallback to plain text parsing.
        lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
        queries = [line.lstrip("1234567890. ").strip() for line in lines]
        return [q for q in queries if len(q) > 10]
    return []


class RAGS:
    def __init__(
        self,
        index_name: str = "rag",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "qwen2.5-coder:1.5b",
        dimension: int = 1024,
        # Lowered relevance threshold to 0.6
        relevance_threshold: float = 0.6,
        tavily_api_key: str = None
    ):
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.dimension = dimension
        self.relevance_threshold = relevance_threshold

        # Set USER_AGENT environment variable
        os.environ["USER_AGENT"] = "MyRAGS/1.0 (violaze25@gmail.com)"

        self.setup_pinecone()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        self.retriever = self.vector_store.as_retriever()
        self.bm25_corpus = []
        self.bm25 = None

        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        self.web_search = TavilySearchResults(k=3)
        # Instantiate ChatOllama with JSON output and low temperature for deterministic outputs.
        self.llm = ChatOllama(model=self.llm_model, format="json", temperature=0.2)
        self.setup_rag_chain()
        self.setup_crag_chains()

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
            # Increase BM25 retrieval top_k from 5 to 10
            self.bm25 = BM25Okapi([doc.split() for doc in self.bm25_corpus])

            logger.info("BM25 corpus and index built.")
            return splits
        except Exception as e:
            raise Exception(f"Failed to load web content: {str(e)}")

    def setup_crag_chains(self):
        """Setup CRAG-specific chains for document evaluation and refinement."""
        relevance_template = (
            "You are an expert document evaluator. For the given question and document, output only a valid JSON object "
            "with a single key \"score\". The score must be a floating point number between 0 and 1, where 0 means "
            "the document is completely irrelevant and 1 means it is highly relevant.\n\n"
            "Question: {question}\n"
            "Document: {document}\n\n"
            "Output:"
        )
        self.relevance_prompt = ChatPromptTemplate.from_template(relevance_template)
        self.relevance_chain = (
            self.relevance_prompt
            | self.llm
            | JsonOutputParser()
        )

        strip_template = (
            "Break this document into distinct knowledge units (strips) that relate to answering the question. "
            "Each strip should be self-contained and coherent.\n\n"
            "Question: {question}\n"
            "Document: {document}\n\n"
            "Format each knowledge strip as a separate bullet point."
        )
        self.strip_prompt = ChatPromptTemplate.from_template(strip_template)
        self.strip_chain = (
            self.strip_prompt
            | self.llm
            | StrOutputParser()
        )

        strip_relevance_template = (
            "Evaluate the relevance of the following knowledge strip to the question. Output only a valid JSON object "
            "with a single key \"score\". The score must be a floating point number between 0 and 1, where 0 means "
            "the strip is completely irrelevant and 1 means it is highly relevant.\n\n"
            "Question: {question}\n"
            "Knowledge Strip: {strip}\n\n"
            "Output:"
        )
        self.strip_relevance_prompt = ChatPromptTemplate.from_template(strip_relevance_template)
        self.strip_relevance_chain = (
            self.strip_relevance_prompt
            | self.llm
            | JsonOutputParser()
        )

    def setup_rag_chain(self):
        """Setup the RAG chain including query rewriting and multi-query generation."""
        query_rewrite_template = (
            "You are a helpful assistant that rewrites queries to be clearer and more detailed. "
            "Rewrite the following query in a concise and detailed manner. For example, if given 'weather', output "
            "'What are the current weather conditions in New York City?'\n\n"
            "IMPORTANT: Output the rewritten query in JSON format as {{\"query\": \"your rewritten query\"}}.\n\n"
            "Original Query: {question}\n\n"
            "Rewritten Query:"
        )
        prompt_rewrite = ChatPromptTemplate.from_template(query_rewrite_template)

        def rewrite_query_with_logging(input_dict):
            raw = (prompt_rewrite | self.llm | StrOutputParser()).invoke(input_dict)
            rewritten = parse_rewritten_query(raw)
            logger.info(f"Rewritten Query: {rewritten}")
            return rewritten

        rewrite_query_chain = rewrite_query_with_logging

        multi_query_generation_template = (
            "Generate exactly 4 different search queries related to the topic below. Each query should explore a "
            "different aspect. Output the result as a JSON array of strings.\n\n"
            "Topic: {rewritten_query}\n\n"
            "Output:"
        )
        prompt_multi_query = ChatPromptTemplate.from_template(multi_query_generation_template)

        def generate_queries(input_dict: dict) -> List[str]:
            try:
                rewritten_query = rewrite_query_chain(input_dict)
                logger.info(f"Rewritten Query for multi-query generation: {rewritten_query}")

                raw_queries = (prompt_multi_query | self.llm | StrOutputParser()).invoke({"rewritten_query": rewritten_query})
                queries = parse_queries(raw_queries)
                logger.info(f"Raw Generated Queries: {queries}")

                if not queries or len(queries) < 4:
                    logger.warning("Did not get 4 queries; falling back to original and rewritten query.")
                    return [input_dict["question"], rewritten_query]

                queries = [rewritten_query] + queries
                seen = set()
                unique_queries = [q for q in queries if not (q in seen or seen.add(q))]
                logger.info(f"Final Queries: {unique_queries}")
                return unique_queries
            except Exception as e:
                logger.error(f"Error in query generation: {str(e)}")
                return [input_dict["question"]]

        self.generate_queries = generate_queries

        # Enhanced final prompt instructing a detailed and comprehensive answer with examples.
        final_template = (
            "You are an expert assistant. Using the context provided, compose a comprehensive, detailed, and well-structured answer "
            "in plain text. Your answer should include examples, explanations, and, if applicable, real-world applications. "
            "If the context is insufficient, say \"I don't have enough information to answer this question accurately.\" \n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer (please be as detailed as possible):"
        )
        prompt_final = ChatPromptTemplate.from_template(final_template)

        context_chain = (
            RunnablePassthrough(lambda queries: self.format_docs(queries))
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

    def format_docs(self, docs: List[Union[Document, str]]) -> str:
        formatted_docs = []
        for i, doc in enumerate(docs, start=1):
            if isinstance(doc, Document):
                content = doc.page_content.strip()
            elif isinstance(doc, str):
                content = doc.strip()
            else:
                logger.warning(f"Skipping document of unexpected type: {type(doc)}")
                continue
            formatted_docs.append(f"Document {i}:\n{content}")
        return "\n\n---\n\n".join(formatted_docs)

    def bm25_retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        if not self.bm25:
            logger.warning("BM25 index not initialized.")
            return []
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [Document(page_content=self.bm25_corpus[idx]) for idx, _ in top_docs]

    def web_search_documents(self, query: str) -> List[Document]:
        try:
            results = self.web_search.invoke(query)
            documents = [
                Document(
                    page_content=result.get("content", ""),
                    metadata={"source": "web", "url": result.get("url", "")}
                )
                for result in results
                if isinstance(result, dict)
            ]
            logger.info(f"Web search found {len(documents)} documents for query: {query}")
            return documents
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

    def reciprocal_rank_fusion(self, results: List[List[Union[Document, str]]], k: int = 60) -> List[Document]:
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

    def hybrid_retrieve_for_query(self, query: str) -> List[Document]:
        logger.info(f"Hybrid retrieval for query: '{query}'")
        pinecone_results = self.retriever.invoke(query)
        logger.info(f"Pinecone retrieved {len(pinecone_results)} documents for query: '{query}'")
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

    def evaluate_document_relevance(self, question: str, document: Document) -> float:
        try:
            raw_response = self.relevance_chain.invoke({
                "question": question,
                "document": document.page_content
            })
            logger.info(f"Raw relevance response: {raw_response}")
            return float(raw_response["score"])
        except Exception as e:
            logger.error(f"Error evaluating document relevance: {e}")
            return 0.0

    def create_knowledge_strips(self, question: str, document: Document) -> List[str]:
        try:
            raw_response = self.strip_chain.invoke({
                "question": question,
                "document": document.page_content
            })
            logger.info(f"Raw knowledge strips response: {raw_response}")
            strips = [
                strip.strip().lstrip('•-* ')
                for strip in raw_response.split('\n')
                if strip.strip() and not strip.isspace()
            ]
            return strips
        except Exception as e:
            logger.error(f"Error creating knowledge strips: {e}")
            return []

    def evaluate_strip_relevance(self, question: str, strip: str) -> float:
        try:
            raw_response = self.strip_relevance_chain.invoke({
                "question": question,
                "strip": strip
            })
            logger.info(f"Raw strip relevance response: {raw_response}")
            return float(raw_response["score"])
        except Exception as e:
            logger.error(f"Error evaluating strip relevance: {e}")
            return 0.0

    def process_with_crag(self, question: str, documents: List[Document]) -> List[Document]:
        """Process documents using the CRAG methodology"""
        logger.info("Starting CRAG document processing")

        processed_docs = []
        need_web_search = True

        for doc in documents:
            relevance_score = self.evaluate_document_relevance(question, doc)
            logger.info(f"Document relevance score: {relevance_score}")

            if relevance_score >= self.relevance_threshold:
                need_web_search = False
                strips = self.create_knowledge_strips(question, doc)
                relevant_strips = []

                for strip in strips:
                    strip_score = self.evaluate_strip_relevance(question, strip)
                    logger.info(f"Knowledge strip relevance score: {strip_score} for strip: {strip}")
                    if strip_score >= self.relevance_threshold:
                        relevant_strips.append(strip)

                if relevant_strips:
                    processed_docs.append(Document(
                        page_content="\n".join(relevant_strips),
                        metadata={
                            **doc.metadata,
                            "relevance_score": relevance_score,
                            "processed_by_crag": True
                        }
                    ))
            elif relevance_score == -1:
                need_web_search = True

        if need_web_search:
            logger.info("No highly relevant documents found or uncertain relevance, performing web search")
            web_docs = self.web_search_documents(question)

            for doc in web_docs:
                strips = self.create_knowledge_strips(question, doc)
                relevant_strips = []

                for strip in strips:
                    strip_score = self.evaluate_strip_relevance(question, strip)
                    logger.info(f"Web knowledge strip relevance score: {strip_score} for strip: {strip}")
                    if strip_score >= self.relevance_threshold:
                        relevant_strips.append(strip)

                if relevant_strips:
                    processed_docs.append(Document(
                        page_content="\n".join(relevant_strips),
                        metadata={
                            **doc.metadata,
                            "source": "web_search",
                            "processed_by_crag": True
                        }
                    ))

        logger.info(f"CRAG processing complete. Found {len(processed_docs)} relevant documents")
        return processed_docs

    def query(self, question: str) -> str:
        try:
            logger.info(f"Processing question: {question}")
            queries = self.generate_queries({"question": question})
            logger.info(f"Generated queries: {queries}")
            all_retrieved_docs = []
            for query in queries:
                retrieved_docs = self.hybrid_retrieve_for_query(query)
                all_retrieved_docs.extend(retrieved_docs)
            reranked_docs = self.reciprocal_rank_fusion([all_retrieved_docs])
            relevant_docs = self.process_with_crag(question, reranked_docs)
            if not relevant_docs:
                logger.warning("No relevant documents found after CRAG processing")
                return "I couldn't find any relevant information to answer your question accurately."
            answer = self.final_rag_chain.invoke({
                "question": question,
                "context": self.format_docs(relevant_docs)
            })
            logger.info("Generated answer using enhanced retrieval and CRAG")
            return answer
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            raise Exception(f"Failed to process query: {str(e)}")


def main():
    # Replace the tavily_api_key with your actual key.
    rag_system = RAGS(tavily_api_key="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")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent//",
    ]
    try:
        documents = rag_system.load_web_content(urls)
        logger.info(f"Loaded {len(documents)} document chunks from the provided URLs.")
    except Exception as e:
        logger.error(f"Error loading web content: {e}")
        documents = []

    # Example query:
    question = "explain about task decomposition?"
    try:
        answer = rag_system.query(question)
        print("Final Answer:")
        print(answer)
    except Exception as e:
        print(f"An error occurred while processing the query: {e}")


if __name__ == "__main__":
    main()
