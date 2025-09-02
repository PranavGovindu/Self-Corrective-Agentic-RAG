"""
Agentic RAG System - Core Implementation

Advanced retrieval-augmented generation system with contextual evaluation,
hybrid search, and intelligent document processing capabilities.
"""

import getpass
import hashlib
import json
import os
import re
import sys
import time
import threading
from operator import itemgetter
from typing import Dict, List, Tuple, Union

import chardet
from dotenv import load_dotenv
from loguru import logger
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

load_dotenv()


# Retrieval configuration - tuned for optimal precision/recall balance
PINECONE_RETRIEVAL_K = 10
BM25_RETRIEVAL_K = 10  
HYBRID_FINAL_K = 3

# CRAG processing limits - prevent excessive LLM calls while maintaining quality
CRAG_INITIAL_DOC_LIMIT = 3
CRAG_WEB_DOC_LIMIT = 2
CRAG_STRIPS_PER_DOC_LIMIT = 5

# Content preview lengths - optimized for context window efficiency
CRAG_DOC_CONTENT_PREVIEW_LEN = 2000
CRAG_STRIP_CREATION_DOC_PREVIEW_LEN = 3000
CRAG_STRIP_CONTENT_PREVIEW_LEN = 500

# Temperature settings - deterministic for structured output, creative for responses
JSON_LLM_TEMPERATURE = 0.0
GENERAL_LLM_TEMPERATURE = 0.1

def parse_rewritten_query(raw_output: str) -> str:
    """Extract query from LLM output - handles both clean JSON and markdown-wrapped responses."""
    logger.info(f"Raw rewritten query output: {raw_output}")
    
    try:
        # Try direct JSON parsing first
        data = json.loads(raw_output)
        if isinstance(data, dict) and "query" in data:
            return data["query"]
        elif isinstance(data, str):
            return data
    except json.JSONDecodeError:
        # Fallback: extract JSON from markdown code blocks or raw text
        match = re.search(r'```json\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
        if not match:
            match = re.search(r'(\{.*?\})', raw_output, re.DOTALL)

        if match:
            try:
                json_str = match.group(1)
                data = json.loads(json_str)
                if isinstance(data, dict) and "query" in data:
                    return data["query"]
            except json.JSONDecodeError as e_inner:
                logger.warning(f"Failed to parse extracted JSON: {e_inner}")
        
        # Last resort: return cleaned raw output
        return raw_output.strip().strip('"')
    except Exception as e:
        logger.error(f"Unexpected error parsing query: {e}")
        return raw_output.strip().strip('"')


def parse_queries(raw_output: str) -> List[str]:
    logger.info(f"Raw multi-query output: {raw_output}")
    try:
        data = json.loads(raw_output)
        if isinstance(data, dict) and "queries" in data and isinstance(data["queries"], list):
            queries = [str(item) for item in data["queries"] if isinstance(item, str)]
            if queries: return queries
    except json.JSONDecodeError:
        match = re.search(r'```json\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
        if not match:
            match = re.search(r'(\{.*?\})', raw_output, re.DOTALL)

        if match:
            try:
                json_str = match.group(1)
                data = json.loads(json_str)
                if isinstance(data, dict) and "queries" in data and isinstance(data["queries"], list):
                    queries = [str(item) for item in data["queries"] if isinstance(item, str)]
                    if queries: return queries
            except json.JSONDecodeError as e_inner:
                logger.warning(f"Failed to parse extracted JSON from multi-query output '{json_str}': {e_inner}")

        lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
        queries = [
            re.sub(r'^\d+\.\s*', '', line).strip('"').strip("'")
            for line in lines
            if len(line) > 5 and not line.lower().startswith(('output:', '{', '[', '```'))
        ]
        if queries:
            logger.warning(f"Could not parse multi-query JSON, falling back to newline splitting. Found {len(queries)} potential queries.")
            return queries
    except Exception as e:
         logger.error(f"Unexpected error parsing multi-queries '{raw_output}': {e}")
    return []


class DocumentProcessingError(Exception):
    """Raised when document loading/processing fails - provides context for debugging."""
    pass

class DocumentProcessor:
    """Handles multi-format document loading with automatic format detection."""
    
    def __init__(self):
        # Strategy pattern - route by file extension or URL
        self.supported_extensions = {
            '.pdf': self.load_pdf,
            '.csv': self.load_csv,
            '.txt': self.load_text,
            'url': self.load_url
        }

    def load_pdf(self, file_path: str) -> List[Document]:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.page_content = self.clean_pdf_text(doc.page_content)
                doc.metadata.update({'source_type': 'pdf', 'file_path': file_path})
            return docs
        except Exception as e:
            logger.error(f"PDF loading error: {str(e)}")
            raise DocumentProcessingError(f"Error loading PDF {file_path}: {str(e)}")

    def clean_pdf_text(self, text: str) -> str:
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())
        return text

    def load_csv(self, file_path: str) -> List[Document]:
        logger.info(f"Loading CSV: {file_path}")
        try:
            loader = CSVLoader(file_path, encoding='utf-8')
            docs = loader.load()
            if not docs:
                raise DocumentProcessingError(f"No content extracted from CSV: {file_path}")
            for doc in docs:
                 doc.metadata.update({'source_type': 'csv', 'file_path': file_path})
            logger.info(f"Successfully loaded CSV with {len(docs)} rows")
            return docs
        except Exception as e:
            raise DocumentProcessingError(f"Failed to process CSV {file_path}: {str(e)}")

    def load_text(self, file_path: str) -> List[Document]:
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                text = file.read()
            document = Document(
                page_content=text,
                metadata={'source_type': 'text', 'file_path': file_path}
            )
            return [document]
        except Exception as e:
            raise DocumentProcessingError(f"Error loading text file {file_path}: {str(e)}")

    def load_url(self, url: str) -> List[Document]:
        try:
            loader = WebBaseLoader(web_paths=[url])
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_type'] = 'url'
                doc.metadata['source'] = url
                doc.metadata['url'] = url
            return documents
        except Exception as e:
            raise DocumentProcessingError(f"Error loading URL {url}: {str(e)}")

    def load_document(self, source: str) -> List[Document]:
        if source.startswith(('http://', 'https://')):
            return self.supported_extensions['url'](source)
        _, extension = os.path.splitext(source)
        extension = extension.lower()
        if extension not in self.supported_extensions:
            logger.warning(f"Unsupported file format '{extension}', attempting to load as text.")
            try:
                return self.load_text(source)
            except Exception as e:
                 raise ValueError(f"Unsupported file format: {extension} and failed to load as text. Error: {e}")
        return self.supported_extensions[extension](source)


class RAGS:
    """
    Retrieval-Augmented Generation System with Contextual Evaluation.
    
    Combines hybrid search (vector + BM25) with intelligent document evaluation
    and knowledge extraction for high-quality responses.
    """
    
    def __init__(
        self,
        index_name: str = "rag",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "qwen2.5-coder:1.5b",
        dimension: int = 1024,
        relevance_threshold: float = 0.3,
        tavily_api_key: str = None
    ):
        self.index_name = index_name
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.dimension = dimension
        self.relevance_threshold = relevance_threshold
        self.document_loader = DocumentProcessor()
        self.pinecone_client = None

        # Enable LLM caching to reduce redundant API calls
        set_llm_cache(InMemoryCache())
        logger.info("In-memory LLM cache enabled.")

        logger.info(f"Initializing RAGS with LLM: {self.llm_model_name}, Embedding: {self.embedding_model_name}")

        self.setup_pinecone()

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key='text'
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': PINECONE_RETRIEVAL_K})

        # BM25 for keyword-based search - complements vector search
        self.bm25_corpus = []
        self.bm25_index_map = {}  # Maps corpus index to document metadata
        self.bm25 = None
        self.bm25_lock = threading.Lock()  # Thread safety for BM25 updates

        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        else:
             logger.warning("Tavily API key not provided. Web search functionality will be disabled.")
        self.web_search = TavilySearchResults(k=3) if tavily_api_key else None

        # Two LLM instances: creative for responses, deterministic for structured output
        self.llm = ChatOllama(
            model=self.llm_model_name,
            temperature=GENERAL_LLM_TEMPERATURE,
        )
        self.json_llm = ChatOllama(
            model=self.llm_model_name,
            temperature=JSON_LLM_TEMPERATURE,
        ).bind(format="json")  # Forces JSON output format

        self.setup_rag_chain()
        self.setup_crag_chains()
        logger.info("RAGS initialization complete.")


    def setup_pinecone(self):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            try:
                pinecone_api_key = getpass.getpass("Enter your Pinecone API key: ")
                os.environ["PINECONE_API_KEY"] = pinecone_api_key
            except Exception as e:
                 raise Exception(f"Could not get Pinecone API key: {e}")
        if not pinecone_api_key:
             raise ValueError("Pinecone API key is required.")
        try:
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            existing_indexes = self.pinecone_client.list_indexes().names()
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
                pinecone_region = os.getenv("PINECONE_REGION", "us-west-2")
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)
                )
                wait_time = 5
                while not self.pinecone_client.describe_index(self.index_name).status["ready"]:
                    logger.info(f"Waiting for index '{self.index_name}' to be ready (waiting {wait_time}s)...")
                    time.sleep(wait_time)
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
            self.index = self.pinecone_client.Index(self.index_name)
            logger.info(f"Pinecone index stats: {self.index.describe_index_stats()}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}") from e

    def _process_single_source(self, source: str) -> Tuple[List[Document], List[Dict]]:
        """Loads, splits, and prepares documents from a single source."""
        try:
            logger.info(f"Starting to process source: {source}")
            docs = self.document_loader.load_document(source)
            if not docs:
                logger.warning(f"No documents loaded from source: {source}")
                return [], []

            logger.info(f"Loaded {len(docs)} raw documents from {source}")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
            )
            splits = text_splitter.split_documents(docs)
            logger.info(f"Created {len(splits)} splits from {source}")

            valid_splits = [split for split in splits if len(split.page_content.strip()) > 50]
            logger.info(f"Keeping {len(valid_splits)} non-empty splits (min length 50)")

            if not valid_splits:
                logger.warning(f"No valid splits generated from source: {source}")
                return [], []

            bm25_data_for_source = []
            for i, split in enumerate(valid_splits):
                split.metadata['source'] = source
                split.metadata['chunk_index'] = i
                bm25_data_for_source.append({
                    'content': split.page_content,
                    'metadata': split.metadata.copy()
                })
            return valid_splits, bm25_data_for_source
        except DocumentProcessingError as dpe:
            logger.error(f"Document processing error for source {source}: {dpe}")
            return [], []
        except Exception as e:
            logger.error(f"Error processing source {source}: {e}")
            return [], []

    def load_content(self, sources: List[str]):
        all_valid_splits = []
        all_bm25_data = []

        for source in sources:
            try:
                splits, bm25_source_data = self._process_single_source(source)
                if splits:
                    all_valid_splits.extend(splits)
                if bm25_source_data:
                    all_bm25_data.extend(bm25_source_data)
            except Exception as e:
                logger.error(f"Source processing failed for {source}: {e}")

        if not all_valid_splits:
            logger.warning("No documents were successfully processed from any source.")
            self.bm25 = None
            return []

        try:
            if all_valid_splits:
                ids = self.vector_store.add_documents(all_valid_splits)
                logger.info(f"Added {len(ids)} documents to vector store from all processed sources.")
        except Exception as e:
            logger.error(f"Error adding to vector store: {e}")

        self.bm25_corpus = []
        self.bm25_index_map = {}
        current_bm25_offset = 0

        for i, item in enumerate(all_bm25_data):
            self.bm25_corpus.append(item['content'])
            self.bm25_index_map[current_bm25_offset + i] = {
                'content': item['content'],
                'metadata': item['metadata']
            }

        if not self.bm25_corpus:
            logger.warning("No documents available to build BM25 index after processing all sources.")
            self.bm25 = None
            return all_valid_splits

        try:
            logger.info(f"Building BM25 index with {len(self.bm25_corpus)} total documents...")
            tokenized_corpus = [self.custom_tokenize(text) for text in self.bm25_corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built successfully.")
        except ImportError:
            logger.error("BM25Okapi not found. Please install rank_bm25: pip install rank_bm25")
            self.bm25 = None
        except Exception as e:
             logger.error(f"Failed to build BM25 index: {e}")
             self.bm25 = None

        return all_valid_splits


    def custom_tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return tokens

    def extract_sources_from_docs(self, docs: List[Document]) -> List[Dict]:
        sources = []
        seen_sources_content_hashes = set()

        for doc in docs:
            content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            source_loc = doc.metadata.get('source', 'Unknown Source')

            if 'file_path' in doc.metadata:
                source_loc = doc.metadata['file_path']
            elif 'url' in doc.metadata:
                source_loc = doc.metadata['url']

            source_info = {
                'content_preview': doc.page_content[:150] + "...",
                'type': doc.metadata.get('source_type', 'unknown'),
                'relevance_score': doc.metadata.get('relevance_score', doc.metadata.get('fusion_score', None)),
                'source': source_loc
            }

            if isinstance(source_loc, str):
                if source_loc.startswith(('http://', 'https://')):
                    source_info['type'] = 'web'
                elif any(source_loc.lower().endswith(ext) for ext in ['.pdf', '.csv', '.txt']):
                    source_info['type'] = os.path.splitext(source_loc)[1].replace('.', '')

            unique_key = (source_loc, content_hash)
            if source_loc == 'Unknown Source' or unique_key not in seen_sources_content_hashes:
                 sources.append(source_info)
                 if source_loc != 'Unknown Source':
                     seen_sources_content_hashes.add(unique_key)

        sources.sort(key=lambda x: (-(x['relevance_score'] or -1), x['source']))
        return sources


    def setup_crag_chains(self):

        relevance_template = (
            "Evaluate the relevance of the document excerpt below to the following question. "
            "Output a JSON object containing only two keys: 'score' (float between 0.0 and 1.0, representing relevance) and 'justification' (string, max 2 sentences explaining the score).\n\n"
            "Question: {question}\n\n"
            "Document Excerpt:\n{document}\n\n"
            "JSON Output:"
        )
        self.relevance_prompt = ChatPromptTemplate.from_template(relevance_template)
        self.relevance_chain = self.relevance_prompt | self.json_llm | JsonOutputParser()

        strip_template = (
            "Break the following document excerpt into distinct, meaningful knowledge units (strips) relevant to the question. "
            "Each strip must be a concise, self-contained bullet point starting with '- '. Only include strips directly relevant to the question. If no relevant strips are found, output 'No relevant strips found.'\n\n"
            "Question: {question}\n\n"
            "Document Excerpt:\n{document}\n\n"
            "Relevant Knowledge Strips (bullet points):"
        )
        self.strip_prompt = ChatPromptTemplate.from_template(strip_template)
        self.strip_chain = self.strip_prompt | self.llm | StrOutputParser()

        strip_relevance_template = (
            "Evaluate the relevance of the single knowledge strip below to the question. "
            "Output a JSON object containing only two keys: 'score' (float between 0.0 and 1.0, representing relevance) and 'justification' (string, max 1 sentence explaining the score).\n\n"
            "Question: {question}\n\n"
            "Knowledge Strip:\n{strip}\n\n"
            "JSON Output:"
        )
        self.strip_relevance_prompt = ChatPromptTemplate.from_template(strip_relevance_template)
        self.strip_relevance_chain = self.strip_relevance_prompt | self.json_llm | JsonOutputParser()

    def setup_rag_chain(self):

        query_intent_template = """Analyze the user query below, considering the provided document context, to understand the core information need.
Extract key concepts and rephrase the query for optimal information retrieval. Avoid generic instructions.
Document Context:\n{document_chunk}\nOriginal Query: {question}\n
Output a JSON object with this structure:\n{{\n"intent": "Brief description...",\n"search_terms": ["list", "of", "keywords"],\n"expanded_query": "A refined query..."\n}}\nJSON Response:"""
        self.query_intent_prompt = ChatPromptTemplate.from_template(query_intent_template)
        self.query_intent_chain = self.query_intent_prompt | self.json_llm | JsonOutputParser()

        document_summary_template = "Summarize the key information in the following document chunk in 2-3 concise sentences. Focus on topics relevant to potential research questions.\nDocument Chunk:\n{document_chunk}\nConcise Summary:"
        self.document_summary_prompt = ChatPromptTemplate.from_template(document_summary_template)
        self.document_summary_chain = self.document_summary_prompt | self.llm | StrOutputParser()

        query_rewrite_template = """Rewrite the 'Original Query' to be more effective for searching academic documents, based on the 'Document Summary' and 'Query Intent Analysis'.
The rewritten query should be specific, use relevant terminology, and focus on the core information need identified in the intent. Output only the rewritten query text within a JSON object.
Document Summary:\n{document_summary}\nQuery Intent Analysis:\n{query_intent}\nOriginal Query:\n{question}\n
Output JSON:\n{{"query": "Your rewritten query goes here"}}"""
        self.prompt_rewrite = ChatPromptTemplate.from_template(query_rewrite_template)
        self.rewrite_query_chain = self.prompt_rewrite | self.json_llm | StrOutputParser()

        multi_query_generation_template = """Based on the 'Rewritten Query' and 'Document Summary', generate exactly 1 distinct search query exploring a different facet of the topic.
The query should be specific, academic-focused, and highly relevant to the document's themes.
Output only a JSON object containing a list of 1 query string.

Document Summary:
{document_summary}

Rewritten Query:
{rewritten_query}

Output JSON:
{{"queries": ["query1"]}}""" # Example format
        self.prompt_multi_query = ChatPromptTemplate.from_template(multi_query_generation_template)
        self.multi_query_chain = self.prompt_multi_query | self.json_llm | JsonOutputParser()

        final_template = """You are an AI research assistant. Synthesize the information from the 'Context' documents below to provide a comprehensive, detailed answer to the 'Question'.
Instructions:\n- Answer the question directly and thoroughly.\n- Integrate information from multiple relevant context documents.\n- Use specific examples or data points from the context.\n- Explain technical terms clearly.\n- Cite sources accurately in-line using the format [Doc X - Type: Source Name]. Example: [Doc 1 - Web: lilianweng.github.io/...] or [Doc 3 - File: my_paper.pdf].\n- Structure the answer logically in Markdown format.\n
Question:\n{question}\nContext:\n{context}\nComprehensive Answer:"""
        prompt_final = ChatPromptTemplate.from_template(final_template)
        def retrieve_and_format(input_dict: dict) -> str:
             docs = input_dict.get('processed_docs', [])
             if not docs:
                 return "No context documents available."
             return self.format_docs(docs)
        self.final_rag_chain = (
            {"context": RunnablePassthrough() | retrieve_and_format, "question": itemgetter("question")}
            | prompt_final
            | self.llm
            | StrOutputParser()
        )

    def generate_queries(self, input_dict: dict) -> List[str]:
        question = input_dict["question"]
        logger.info(f"Generating queries for original question: {question}")
        generated_queries_list = []
        try:
            context_query = f"General information related to: {question}"
            retrieved_docs = self.hybrid_retrieve_for_query(context_query, top_k=1)
            if not retrieved_docs:
                logger.warning("No documents retrieved for initial context summary. Using original query only for generation steps.")
                first_document_split = "No document context available."
            else:
                first_document_split = retrieved_docs[0].page_content

            document_summary = self.document_summary_chain.invoke({"document_chunk": first_document_split})
            logger.info(f"Document Summary for query generation: {document_summary}")

            try:
                query_intent_result = self.query_intent_chain.invoke({"document_chunk": first_document_split, "question": question})
                logger.info(f"Query Intent Analysis: {query_intent_result}")
                query_intent_str = json.dumps(query_intent_result) if isinstance(query_intent_result, dict) else "{}"
            except Exception as e:
                logger.error(f"Error getting query intent: {e}. Proceeding without intent analysis.")
                query_intent_str = "{}"

            try:
                rewritten_query_output_str = self.rewrite_query_chain.invoke({"question": question, "document_summary": document_summary, "query_intent": query_intent_str})
                rewritten_query = parse_rewritten_query(rewritten_query_output_str)
                logger.info(f"Rewritten Query: {rewritten_query}")
            except Exception as e:
                logger.error(f"Error rewriting query: {e}. Using original query.")
                rewritten_query = question

            try:
                multi_query_result_dict = self.multi_query_chain.invoke({
                     "rewritten_query": rewritten_query,
                     "document_summary": document_summary
                })
                if isinstance(multi_query_result_dict, dict) and "queries" in multi_query_result_dict and isinstance(multi_query_result_dict["queries"], list):
                     generated_queries_list = multi_query_result_dict["queries"]
                     logger.info(f"Generated {len(generated_queries_list)} additional query/queries: {generated_queries_list}")
                else:
                     logger.warning(f"Multi-query generation returned unexpected format or no queries: {multi_query_result_dict}")
                     generated_queries_list = []
            except Exception as e:
                logger.error(f"Error generating multiple queries: {e}. Skipping additional query generation.")
                generated_queries_list = []

            final_queries = [question, rewritten_query]
            if generated_queries_list:
                final_queries.extend(q for q in generated_queries_list if q)

            seen = set()
            unique_queries = [q for q in final_queries if q and not (q in seen or seen.add(q))]
            logger.info(f"Final Unique Queries for Retrieval: {unique_queries}")

            return unique_queries[:3]

        except Exception as e:
            logger.error(f"Error in query generation pipeline: {str(e)}")
            return [question]

    def format_docs(self, docs: List[Document]) -> str:
        formatted_docs = []
        for i, doc in enumerate(docs, start=1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            source_type = metadata.get('type', metadata.get('source_type', 'Unknown'))

            source_name = metadata.get('file_path', metadata.get('url', metadata.get('source', 'N/A')))

            if isinstance(source_name, str):
                 if source_name.startswith('http'):
                     try:
                          domain = source_name.split('/')[2]
                          path_part = "/".join(source_name.split('/')[3:])[:30]
                          source_name_display = f"{domain}/{path_part}..." if path_part else domain
                     except: source_name_display = source_name[:40] + "..." if len(source_name) > 40 else source_name
                 elif os.path.exists(source_name):
                      source_name_display = os.path.basename(source_name)
                 elif len(source_name) > 40:
                      source_name_display = "..." + source_name[-37:]
                 else:
                      source_name_display = source_name
            else:
                source_name_display = "N/A"

            citation = f"[Doc {i} - {source_type.capitalize()}: {source_name_display}]"
            formatted_docs.append(f"{citation}\n{content}")
        return "\n\n---\n\n".join(formatted_docs)


    def bm25_retrieve(self, query: str, top_k: int = BM25_RETRIEVAL_K) -> List[Document]:
        if not self.bm25 or not self.bm25_corpus:
            logger.warning("BM25 index not available or corpus is empty.")
            return []
        try:
            tokenized_query = self.custom_tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)

            top_indices_scores = sorted(
                [(idx, score) for idx, score in enumerate(scores) if score > 0],
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            retrieved = []
            for idx, score in top_indices_scores:
                doc_info = self.bm25_index_map.get(idx)
                if doc_info:
                     doc = Document(page_content=doc_info['content'], metadata=doc_info['metadata'].copy())
                     doc.metadata['bm25_score'] = score
                     doc.metadata['retrieval_method'] = 'bm25'
                     retrieved.append(doc)
                else:
                    logger.warning(f"BM25 index {idx} not found in map for query '{query}'. This might indicate an issue with bm25_index_map consistency.")

            logger.info(f"BM25 retrieved {len(retrieved)} documents for query '{query}'")
            return retrieved
        except Exception as e:
            logger.error(f"Error during BM25 retrieval for query '{query}': {e}")
            return []

    def web_search_documents(self, query: str) -> List[Document]:
        if not self.web_search:
             logger.warning("Web search tool (Tavily) is not configured.")
             return []
        try:
            logger.info(f"Performing web search for: {query}")
            results = self.web_search.invoke(query)
            documents = []
            if isinstance(results, list):
                for result_item in results:
                    if isinstance(result_item, dict) and result_item.get("content"):
                        doc_metadata = {
                            "source": result_item.get("url", "web_search_result"),
                            "source_type": "web",
                            "title": result_item.get("title", ""),
                            "url": result_item.get("url", ""),
                            "retrieval_method": "web_search"
                        }
                        documents.append(Document(
                            page_content=result_item.get("content", ""),
                            metadata=doc_metadata
                        ))
                logger.info(f"Web search found {len(documents)} documents for query: {query}")
            elif isinstance(results, str):
                logger.warning(f"Web search returned a string, possibly an error or no results: {results}")
            else:
                logger.warning(f"Web search returned unexpected format: {type(results)}. Results: {results}")
            return documents
        except Exception as e:
            logger.error(f"Web search error for query '{query}': {e}")
            return []

    def reciprocal_rank_fusion_direct(self, ranked_lists: List[List[Document]], k_rrf: int = 60) -> List[Document]:
        """
        Performs Reciprocal Rank Fusion on multiple lists of ranked documents.
        ranked_lists: A list where each element is a list of Document objects, sorted by relevance.
        k_rrf: Constant for RRF, typically 60.
        """
        fused_scores = {}
        doc_map = {}

        def get_doc_key(doc: Document) -> str:
            return hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                key = get_doc_key(doc)
                if key not in fused_scores:
                    fused_scores[key] = 0
                    doc_map[key] = doc

                score = 1.0 / (rank + k_rrf)
                fused_scores[key] += score


        sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)

        fused_docs = []
        for key in sorted_keys:
             doc = doc_map[key]
             final_metadata = doc.metadata.copy()
             final_metadata['fusion_score'] = fused_scores[key]
             final_metadata['retrieval_method'] = 'hybrid_rrf'


             fused_docs.append(Document(page_content=doc.page_content, metadata=final_metadata))

        logger.info(f"RRF Fusion: Input lists count = {len(ranked_lists)}. Total unique docs considered = {len(doc_map)}. Output fused docs = {len(fused_docs)}")
        return fused_docs


    def hybrid_retrieve_for_query(self, query: str, top_k: int = HYBRID_FINAL_K) -> List[Document]:
        """
        Hybrid retrieval combining vector similarity and keyword matching.
        Uses Reciprocal Rank Fusion to merge results from both approaches.
        """
        logger.info(f"Starting hybrid retrieval for query: '{query}' (target final top_k={top_k})")
        
        # Vector similarity search via Pinecone
        try:
            pinecone_results = self.retriever.invoke(query) 
            for doc in pinecone_results:
                doc.metadata['retrieval_method'] = 'pinecone_vector'
            logger.info(f"Pinecone retrieved {len(pinecone_results)} documents")
        except Exception as e:
            logger.error(f"Pinecone retrieval error for query '{query}': {e}")
            pinecone_results = []

        # Keyword-based search via BM25
        bm25_results = self.bm25_retrieve(query, top_k=BM25_RETRIEVAL_K) 

        # Combine results using RRF - handles different scoring scales gracefully
        all_ranked_lists = []
        if pinecone_results:
            all_ranked_lists.append(pinecone_results)
        if bm25_results:
            all_ranked_lists.append(bm25_results)

        if not all_ranked_lists:
            logger.warning(f"No results from any retrieval method for query '{query}'.")
            return []

        fused_results = self.reciprocal_rank_fusion_direct(all_ranked_lists)
        final_results = fused_results[:top_k] 
        logger.info(f"Hybrid retrieval returning {len(final_results)} fused documents for query '{query}'")
        return final_results

    def evaluate_document_relevance(self, question: str, document: Document) -> Tuple[float, str]:
        try:
            content_preview = document.page_content[:CRAG_DOC_CONTENT_PREVIEW_LEN]
            response = self.relevance_chain.invoke({"question": question, "document": content_preview})

            score = float(response.get("score", 0.0))
            justification = response.get("justification", "No justification provided.")
            return score, justification
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Error parsing relevance response (expected dict): {e}. Raw document source: {document.metadata.get('source', 'Unknown')}")
            return 0.0, "Error parsing relevance score from LLM."
        except Exception as e:
            logger.error(f"Error evaluating document relevance for '{document.metadata.get('source', 'Unknown')}': {e}")
            return 0.0, "Error during relevance evaluation."

    def create_knowledge_strips(self, question: str, document: Document) -> List[str]:
        try:
            content_preview = document.page_content[:CRAG_STRIP_CREATION_DOC_PREVIEW_LEN]
            raw_response = self.strip_chain.invoke({"question": question, "document": content_preview})

            if "No relevant strips found." in raw_response:
                return []

            lines = raw_response.split('\n')
            strips = [line.strip().lstrip('-').strip() for line in lines if line.strip().startswith('-')]

            if not strips and raw_response.strip() and "No relevant strips found." not in raw_response:
                strips = [line.strip() for line in lines if len(line.strip()) > 10]

            return strips
        except Exception as e:
            logger.error(f"Error creating knowledge strips for doc '{document.metadata.get('source', 'Unknown')}': {e}")
            return []

    def evaluate_strip_relevance(self, question: str, strip: str) -> Tuple[float, str]:
        if not strip or not strip.strip():
            return 0.0, "Empty strip provided."
        try:
            response = self.strip_relevance_chain.invoke({"question": question, "strip": strip[:CRAG_STRIP_CONTENT_PREVIEW_LEN]})

            score = float(response.get("score", 0.0))
            justification = response.get("justification", "No justification provided.")
            return score, justification
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Error parsing strip relevance response (expected dict): {e}. Strip: '{strip[:50]}...'")
            return 0.0, "Error parsing strip relevance score."
        except Exception as e:
            logger.error(f"Error evaluating strip relevance for strip '{strip[:50]}...': {e}")
            return 0.0, "Error during strip relevance evaluation."

    def _process_crag_batch(self, question: str, docs_to_process: List[Document], is_web_search: bool = False) -> List[Dict]:
        """Helper to process a batch of documents for CRAG, returning scored strips."""
        processed_strips_info = []

        if not docs_to_process:
            return []

        doc_eval_results = []
        for doc in docs_to_process:
            try:
                score, justification = self.evaluate_document_relevance(question, doc)
                doc_eval_results.append({"score": score, "justification": justification})
            except Exception as e:
                logger.error(f"Document relevance evaluation failed for doc: {e}")
                doc_eval_results.append({"score": 0.0, "justification": "Error during evaluation"})

        relevant_docs_for_stripping = []
        for i, doc in enumerate(docs_to_process):
            score = doc_eval_results[i].get("score", 0.0) if isinstance(doc_eval_results[i], dict) else float(doc_eval_results[i])
            justification = doc_eval_results[i].get("justification", "") if isinstance(doc_eval_results[i], dict) else ""
            doc_source_display = doc.metadata.get('source', 'Unknown')
            logger.info(f"CRAG {'Web' if is_web_search else 'Doc'} '{doc_source_display}' relevance: {score:.2f}. Justification: {justification}")
            if score >= self.relevance_threshold:
                relevant_docs_for_stripping.append(doc)
            else:
                if not is_web_search:
                    pass

        if not relevant_docs_for_stripping:
            return []

        strip_creation_raw_outputs = []
        for doc in relevant_docs_for_stripping:
            try:
                raw_output = self.strip_chain.invoke({"question": question, "document": doc.page_content[:CRAG_STRIP_CREATION_DOC_PREVIEW_LEN]})
                strip_creation_raw_outputs.append(raw_output)
            except Exception as e:
                logger.error(f"Strip creation failed for doc: {e}")
                strip_creation_raw_outputs.append("No relevant strips found.")

        all_strips_to_evaluate_relevance = []
        for doc, raw_strips_string in zip(relevant_docs_for_stripping, strip_creation_raw_outputs):
            if "No relevant strips found." in raw_strips_string:
                current_strips = []
            else:
                lines = raw_strips_string.split('\n')
                current_strips = [line.strip().lstrip('-').strip() for line in lines if line.strip().startswith('-')]
                if not current_strips and raw_strips_string.strip():
                    current_strips = [line.strip() for line in lines if len(line.strip()) > 10]

            for strip_text in current_strips[:CRAG_STRIPS_PER_DOC_LIMIT]:
                if strip_text:
                    all_strips_to_evaluate_relevance.append({
                        "text": strip_text,
                        "source_doc": doc
                    })

        if not all_strips_to_evaluate_relevance:
            return []

        strip_relevance_results = []
        for s_info in all_strips_to_evaluate_relevance:
            try:
                score, justification = self.evaluate_strip_relevance(question, s_info["text"])
                strip_relevance_results.append({"score": score, "justification": justification})
            except Exception as e:
                logger.error(f"Strip relevance evaluation failed: {e}")
                strip_relevance_results.append({"score": 0.0, "justification": "Error during evaluation"})

        for i, s_info in enumerate(all_strips_to_evaluate_relevance):
            score = strip_relevance_results[i].get("score", 0.0) if isinstance(strip_relevance_results[i], dict) else 0.0
            if score >= self.relevance_threshold:
                processed_strips_info.append({
                    'content': s_info["text"],
                    'score': score,
                    'source_doc_metadata': s_info["source_doc"].metadata.copy(),
                    'source_doc_content_preview': s_info["source_doc"].page_content[:100] + "..."
                })
        return processed_strips_info


    def process_with_crag(self, question: str, documents: List[Document]) -> List[Document]:
        """
        Contextual RAG processing - evaluates document relevance and extracts knowledge strips.
        Falls back to web search if initial documents are insufficient.
        """
        logger.info(f"Starting CRAG processing for {len(documents)} retrieved documents.")
        
        all_relevant_strips_info = []
        
        # Process initial retrieved documents
        initial_docs_to_process = documents[:CRAG_INITIAL_DOC_LIMIT]
        initial_strips_info = self._process_crag_batch(question, initial_docs_to_process, is_web_search=False)
        all_relevant_strips_info.extend(initial_strips_info)

        initial_docs_had_relevant_strips = any(
            s_info['source_doc_metadata']['source'] == doc.metadata['source']
            for s_info in initial_strips_info
            for doc in initial_docs_to_process
        )

        need_web_search = not initial_docs_had_relevant_strips
        if not initial_strips_info:
            need_web_search = True

        if need_web_search and self.web_search:
            logger.info("Performing web search as initial documents/strips were not sufficient or relevant enough.")
            web_search_query = question
            web_docs_retrieved = self.web_search_documents(web_search_query)

            if web_docs_retrieved:
                web_docs_to_process = web_docs_retrieved[:CRAG_WEB_DOC_LIMIT]
                web_strips_info = self._process_crag_batch(question, web_docs_to_process, is_web_search=True)
                all_relevant_strips_info.extend(web_strips_info)
            else:
                logger.info("Web search yielded no documents.")
        elif need_web_search and not self.web_search:
            logger.warning("Web search needed but not configured.")

        if not all_relevant_strips_info:
            logger.warning("CRAG processing (including potential web search) produced no relevant strips. Falling back to top raw retrieved documents.")
            final_docs_for_rag = documents[:max(1, CRAG_INITIAL_DOC_LIMIT // 2)]
            for doc_fallback in final_docs_for_rag:
                doc_fallback.metadata["processed_by_crag"] = False
                doc_fallback.metadata["crag_fallback"] = True
            return final_docs_for_rag

        strips_by_source = {}
        for s_info in all_relevant_strips_info:
            source_key = s_info['source_doc_metadata'].get('source', 'Unknown_Source_CRAG')
            if source_key not in strips_by_source:
                strips_by_source[source_key] = {
                    'strips': [],
                    'metadata': s_info['source_doc_metadata'],
                    'max_strip_score': 0.0
                }
            strips_by_source[source_key]['strips'].append(s_info['content'])
            strips_by_source[source_key]['max_strip_score'] = max(
                strips_by_source[source_key]['max_strip_score'], s_info['score']
            )

        final_docs_for_rag = []
        for source_key, data in strips_by_source.items():
            combined_content = "\n".join(f"- {s_text}" for s_text in data['strips'])
            doc_metadata = data['metadata'].copy()
            doc_metadata["processed_by_crag"] = True
            doc_metadata["relevant_strips_count"] = len(data['strips'])
            doc_metadata["max_strip_score"] = data['max_strip_score']
            doc_metadata["source"] = source_key

            doc_metadata['relevance_score'] = data['max_strip_score']

            final_docs_for_rag.append(Document(page_content=combined_content, metadata=doc_metadata))

        final_docs_for_rag.sort(key=lambda d: d.metadata.get("max_strip_score", 0.0), reverse=True)

        logger.info(f"CRAG assembled {len(final_docs_for_rag)} documents from processed strips.")
        return final_docs_for_rag


    def query(self, question: str) -> Tuple[str, List[Dict]]:
        start_time = time.time()
        logger.info(f"Received query: {question}")
        try:
            gen_start_time = time.time()
            queries_for_retrieval = self.generate_queries({"question": question})
            logger.info(f"Query generation took {time.time() - gen_start_time:.2f} seconds. Generated queries: {queries_for_retrieval}")

            if not queries_for_retrieval:
                logger.error("Query generation failed to produce any usable queries. Aborting.")
                return "Error: Could not generate effective search queries for your question.", []

            retrieval_start_time = time.time()
            unique_docs_map = {}
            for q_idx, q_text in enumerate(queries_for_retrieval):
                logger.info(f"Processing retrieval for query {q_idx+1}/{len(queries_for_retrieval)}: '{q_text}'")
                try:
                    docs_from_query = self.hybrid_retrieve_for_query(q_text, top_k=HYBRID_FINAL_K)
                    for doc in docs_from_query:
                        doc_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                        if doc_hash not in unique_docs_map:
                            unique_docs_map[doc_hash] = doc
                        else:
                            pass
                except Exception as e:
                    logger.error(f"Retrieval failed for query '{q_text}': {e}")

            initial_retrieved_docs = list(unique_docs_map.values())
            initial_retrieved_docs.sort(key=lambda d: d.metadata.get('fusion_score', 0.0), reverse=True)

            logger.info(f"Total unique documents initially retrieved across all queries: {len(initial_retrieved_docs)}")
            logger.info(f"Parallel retrieval loop took {time.time() - retrieval_start_time:.2f} seconds.")

            if not initial_retrieved_docs:
                logger.warning("No documents found from internal sources across all queries. Attempting direct web search as fallback.")
                if self.web_search:
                    web_fallback_docs = self.web_search_documents(question)
                    if not web_fallback_docs:
                        return "No relevant information found in indexed documents or through web search.", []
                    else:
                        logger.info(f"Using {len(web_fallback_docs)} web search results as fallback context.")
                        initial_retrieved_docs = web_fallback_docs[:CRAG_INITIAL_DOC_LIMIT + CRAG_WEB_DOC_LIMIT]
                else:
                    return "No relevant information found in indexed documents, and web search is not configured.", []

            crag_start_time = time.time()
            processed_docs_for_answer = self.process_with_crag(question, initial_retrieved_docs)
            logger.info(f"CRAG processing took {time.time() - crag_start_time:.2f} seconds.")

            if not processed_docs_for_answer:
                logger.error("CRAG processing (after retrieval and potential web search) failed to produce any relevant documents for the final answer.")
                return "Could not find or process relevant information to answer your question after extensive checks.", []

            sources_for_display = self.extract_sources_from_docs(processed_docs_for_answer)

            final_answer_start_time = time.time()
            logger.info(f"Generating final answer with {len(processed_docs_for_answer)} CRAG-processed documents...")

            final_answer_payload = {"question": question, "processed_docs": processed_docs_for_answer}
            final_answer = self.final_rag_chain.invoke(final_answer_payload)

            answer_str = final_answer if isinstance(final_answer, str) else json.dumps(final_answer, indent=2)
            logger.info(f"Final answer generation took {time.time() - final_answer_start_time:.2f} seconds.")

            total_time = time.time() - start_time
            logger.info(f"Total query processing time: {total_time:.2f} seconds.")
            return answer_str, sources_for_display

        except Exception as e:
            logger.exception(f"Critical error during query processing pipeline for question '{question}': {str(e)}")
            return f"An error occurred while processing your query: {str(e)}", []


    def clear_index(self):
        logger.warning(f"Attempting to clear all data from Pinecone index '{self.index_name}' and reset local BM25 state.")
        if not self.pinecone_client or not self.index:
            logger.error("Pinecone client or index not initialized. Cannot clear.")
            raise Exception("Pinecone client/index not initialized.")
        try:
            logger.info(f"Deleting all vectors from Pinecone index: {self.index_name}")
            self.index.delete(delete_all=True)

            with self.bm25_lock:
                self.bm25_corpus = []
                self.bm25_index_map = {}
                self.bm25 = None
            logger.info("Local BM25 data (corpus, index_map, index) cleared.")

            time.sleep(5)
            stats = self.index.describe_index_stats()
            logger.info(f"Pinecone index '{self.index_name}' stats after delete_all: {stats}")
            if stats.total_vector_count == 0:
                logger.info(f"Pinecone index '{self.index_name}' successfully cleared (0 vectors).")
            else:
                logger.warning(f"Pinecone index '{self.index_name}' shows {stats.total_vector_count} vectors after clearing. Deletion might be in progress or failed.")
            return True
        except Exception as e:
            logger.error(f"Error clearing Pinecone index '{self.index_name}' or local BM25 state: {str(e)}")
            raise Exception(f"Failed to clear index '{self.index_name}': {str(e)}")

def main():
    try:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True
        )

        logger.add(
            "logs/rag_system.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )

        import logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


        tavily_key = os.getenv("TAVILY_API_KEY")

        print("Initializing RAG system (from main)...")
        rag_system = RAGS(tavily_api_key=tavily_key)




        sources_to_load = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
        ]

        if sources_to_load:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError:
                print("CRITICAL: rank_bm25 not installed. BM25 retrieval will be skipped. Install with: pip install rank_bm25")

            print(f"Loading content from {len(sources_to_load)} sources using multiple workers...")
            processed_splits = rag_system.load_content(sources_to_load)
            print(f"Loaded and processed {len(processed_splits)} document chunks in total.")
            if rag_system.bm25:
                print(f"BM25 index built with {len(rag_system.bm25_corpus)} documents.")
            else:
                print("BM25 index not available or not built.")
        else:
             print("No sources provided to load. Querying existing index (if any).")

        question = "Explain task decomposition in autonomous agents based on the provided context. How is it achieved and what are the common strategies?"
        print(f"\nQuerying with: '{question}'")

        answer, sources_info = rag_system.query(question)

        print("\n------ Final Answer ------")
        print(answer)
        print("\n------ Sources Used (Summary) ------")
        if sources_info:
            for i, source_item in enumerate(sources_info):
                print(f"  {i+1}. Type: {source_item.get('type', 'N/A')}, Source: {source_item.get('source', 'N/A')}, "
                      f"Relevance: {source_item.get('relevance_score', 'N/A'):.2f if isinstance(source_item.get('relevance_score'), float) else 'N/A'}")
        else:
            print("No sources were cited for this answer, or no information found.")



    except Exception as e:
         print(f"\nAN UNEXPECTED ERROR OCCURRED IN MAIN EXECUTION: {e}")
         logger.exception("Main execution failed critically.")
    finally:
        print("\nMain execution finished.")


if __name__ == "__main__":
    main()
