import os
import getpass
import time
import logging # Keep import for logger usage
from typing import List, Union, Dict, Tuple
from operator import itemgetter
import hashlib
import json
import sys
import re
import chardet
import bs4
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
from langchain_community.tools.tavily_search import TavilySearchResults

# REMOVED logging.basicConfig - Configure logging ONCE in your main entry point (e.g., orchestrator.py)
logger = logging.getLogger(__name__) # Get logger for use in this module


def parse_rewritten_query(raw_output: str) -> str:
    # (Function unchanged)
    logger.info(f"Raw rewritten query output: {raw_output}")
    try:
        data = json.loads(raw_output)
        if isinstance(data, dict) and "query" in data:
            return data["query"]
        elif isinstance(data, str):
            return data
    except json.JSONDecodeError:
        match = re.search(r'json\s*(\{.*?\})\s*', raw_output, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "query" in data:
                    return data["query"]
            except json.JSONDecodeError:
                pass
        return raw_output.strip().strip('"')
    except Exception as e:
        logger.error(f"Unexpected error parsing rewritten query '{raw_output}': {e}")
    return raw_output.strip().strip('"')

def parse_queries(raw_output: str) -> List[str]:
    # (Function unchanged)
    logger.info(f"Raw multi-query output: {raw_output}")
    try:
        match = re.search(r'json\s*(\{.*?\})\s*', raw_output, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = raw_output
        data = json.loads(json_str)
        if isinstance(data, dict) and "queries" in data and isinstance(data["queries"], list):
            queries = [str(item) for item in data["queries"] if isinstance(item, str)]
            return queries
    except json.JSONDecodeError:
        lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
        queries = [line for line in lines if len(line) > 5 and not line.startswith(('Output:', '{', '['))]
        logger.warning(f"Could not parse multi-query JSON, falling back to newline splitting. Found {len(queries)} potential queries.")
        return queries
    except Exception as e:
         logger.error(f"Unexpected error parsing multi-queries '{raw_output}': {e}")
    return []

class DocumentProcessingError(Exception):
    # (Class unchanged)
    pass

class DocumentProcessor:
    # (Class unchanged)
    def __init__(self):
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
            raise Exception(f"Error loading PDF {file_path}: {str(e)}")

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
            raise Exception(f"Error loading text file {file_path}: {str(e)}")

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
            raise Exception(f"Error loading URL {url}: {str(e)}")

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
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.dimension = dimension
        self.relevance_threshold = relevance_threshold
        self.document_loader = DocumentProcessor()

        logger.info(f"Initializing RAGS with LLM: {self.llm_model}, Embedding: {self.embedding_model}")

        self.setup_pinecone()

        # --- FIX: Removed explicit device setting to avoid meta tensor error ---
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            # model_kwargs={'device': 'cpu'}, # REMOVED this line
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key='text'
        )

        # --- PARAMETER REDUCTION (Keep k low) ---
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 3})

        self.bm25_corpus = []
        self.bm25_index_map = {}
        self.bm25 = None

        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        else:
             logger.warning("Tavily API key not provided. Web search functionality will be disabled.")
        self.web_search = TavilySearchResults(k=3) if tavily_api_key else None

        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=0.1,
        )
        self.setup_rag_chain()
        self.setup_crag_chains()
        logger.info("RAGS initialization complete.")


    def setup_pinecone(self):
        # (Method unchanged)
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
            self.pc = Pinecone(api_key=pinecone_api_key)
            existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name, dimension=self.dimension, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                wait_time = 5
                while not self.pc.describe_index(self.index_name).status["ready"]:
                    logger.info(f"Waiting for index '{self.index_name}' to be ready (waiting {wait_time}s)...")
                    time.sleep(wait_time)
            else:
                 logger.info(f"Using existing Pinecone index: {self.index_name}")
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Pinecone index stats: {self.index.describe_index_stats()}")
        except Exception as e:
            # Wrap exception for clearer error reporting during init
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}") from e


    def load_content(self, sources: List[str]):
        # (Method unchanged)
        all_splits = []
        current_bm25_offset = len(self.bm25_corpus)
        for source in sources:
            try:
                logger.info(f"Starting to process source: {source}")
                docs = self.document_loader.load_document(source)
                if not docs:
                    logger.warning(f"No documents loaded from source: {source}")
                    continue
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
                     continue
                try:
                    for i, split in enumerate(valid_splits):
                         split.metadata['source'] = source
                         split.metadata['chunk_index'] = i
                    ids = self.vector_store.add_documents(valid_splits)
                    logger.info(f"Added {len(ids)} documents to vector store from {source}")
                except Exception as e:
                    logger.error(f"Error adding to vector store from {source}: {e}")
                    continue
                for i, split in enumerate(valid_splits):
                    bm25_idx = current_bm25_offset + len(self.bm25_corpus)
                    self.bm25_corpus.append(split.page_content)
                    self.bm25_index_map[bm25_idx] = {
                        'content': split.page_content,
                        'metadata': split.metadata
                    }
                all_splits.extend(valid_splits)
            except Exception as e:
                logger.error(f"Error processing source {source}: {e}")
        if not self.bm25_corpus:
            logger.warning("No documents were successfully processed to build BM25 index.")
            return []
        try:
            logger.info(f"Building BM25 index with {len(self.bm25_corpus)} total documents...")
            tokenized_corpus = [self.custom_tokenize(text) for text in self.bm25_corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built successfully.")
        except Exception as e:
             logger.error(f"Failed to build BM25 index: {e}")
             self.bm25 = None
        return all_splits

    def custom_tokenize(self, text: str) -> List[str]:
        # (Method unchanged)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return tokens

    def extract_sources_from_docs(self, docs: List[Document]) -> List[Dict]:
        # (Method unchanged)
        sources = []
        seen_sources = set()
        for doc in docs:
            source_loc = doc.metadata.get('source', 'Unknown Source')
            source_info = {
                'content': doc.page_content[:300] + "...",
                'type': doc.metadata.get('source_type', 'unknown'),
                'relevance': doc.metadata.get('relevance_score', None),
                'source': source_loc
            }
            if source_loc.startswith(('http://', 'https://')):
                source_info['type'] = 'web'
            elif any(source_loc.endswith(ext) for ext in ['.pdf', '.csv', '.txt']):
                 source_info['type'] = 'file'
            if source_loc not in seen_sources and source_loc != 'Unknown Source':
                 sources.append(source_info)
                 seen_sources.add(source_loc)
            elif source_loc == 'Unknown Source':
                 sources.append(source_info)
        sources.sort(key=lambda x: x['source'])
        return sources

    def setup_crag_chains(self):
        # (Method unchanged)
        json_llm = self.llm.bind(format="json")
        string_llm = self.llm
        relevance_template = (
            "Evaluate the relevance of the document excerpt below to the following question. "
            "Output a JSON object containing only two keys: 'score' (float between 0.0 and 1.0) and 'justification' (string, max 2 sentences).\n\n"
            "Question: {question}\n\n"
            "Document Excerpt:\n{document}\n\n"
            "JSON Output:"
        )
        self.relevance_prompt = ChatPromptTemplate.from_template(relevance_template)
        self.relevance_chain = self.relevance_prompt | json_llm | JsonOutputParser()
        strip_template = (
            "Break the following document excerpt into distinct, meaningful knowledge units (strips) relevant to the question. "
            "Each strip must be a concise, self-contained bullet point starting with '- '. Only include strips directly relevant to the question.\n\n"
            "Question: {question}\n\n"
            "Document Excerpt:\n{document}\n\n"
            "Relevant Knowledge Strips (bullet points):"
        )
        self.strip_prompt = ChatPromptTemplate.from_template(strip_template)
        self.strip_chain = self.strip_prompt | string_llm | StrOutputParser()
        strip_relevance_template = (
            "Evaluate the relevance of the single knowledge strip below to the question. "
            "Output a JSON object containing only two keys: 'score' (float between 0.0 and 1.0) and 'justification' (string, max 1 sentence).\n\n"
            "Question: {question}\n\n"
            "Knowledge Strip:\n{strip}\n\n"
            "JSON Output:"
        )
        self.strip_relevance_prompt = ChatPromptTemplate.from_template(strip_relevance_template)
        self.strip_relevance_chain = self.strip_relevance_prompt | json_llm | JsonOutputParser()

    def setup_rag_chain(self):
        # (Method largely unchanged, just adjusting multi-query prompt)
        json_llm = self.llm.bind(format="json")
        string_llm = self.llm
        query_intent_template = """Analyze the user query below, considering the provided document context, to understand the core information need.
Extract key concepts and rephrase the query for optimal information retrieval. Avoid generic instructions.
Document Context:\n{document_chunk}\nOriginal Query: {question}\n
Output a JSON object with this structure:\n{{\n"intent": "Brief description...",\n"search_terms": ["list", "of", "keywords"],\n"expanded_query": "A refined query..."\n}}\nJSON Response:"""
        self.query_intent_prompt = ChatPromptTemplate.from_template(query_intent_template)
        self.query_intent_chain = self.query_intent_prompt | json_llm | JsonOutputParser()
        document_summary_template = "Summarize the key information in the following document chunk in 2-3 concise sentences. Focus on topics relevant to potential research questions.\nDocument Chunk:\n{document_chunk}\nConcise Summary:"
        self.document_summary_prompt = ChatPromptTemplate.from_template(document_summary_template)
        self.document_summary_chain = self.document_summary_prompt | string_llm | StrOutputParser()
        query_rewrite_template = """Rewrite the 'Original Query' to be more effective for searching academic documents, based on the 'Document Summary' and 'Query Intent Analysis'.
The rewritten query should be specific, use relevant terminology, and focus on the core information need identified in the intent. Output only the rewritten query text within a JSON object.
Document Summary:\n{document_summary}\nQuery Intent Analysis:\n{query_intent}\nOriginal Query:\n{question}\n
Output JSON:\n{{"query": "Your rewritten query goes here"}}"""
        self.prompt_rewrite = ChatPromptTemplate.from_template(query_rewrite_template)
        self.rewrite_query_chain = self.prompt_rewrite | json_llm | StrOutputParser() # Keep StrOutputParser, parsing handled later

        # --- RESTORED Multi-Query Prompt ---
        # --- PARAMETER REDUCTION: Ask for only 1 extra query ---
        multi_query_generation_template = """Based on the 'Rewritten Query' and 'Document Summary', generate exactly 1 distinct search query exploring a different facet of the topic.
The query should be specific, academic-focused, and highly relevant to the document's themes.
Output only a JSON object containing a list of 1 query string.

Document Summary:
{document_summary}

Rewritten Query:
{rewritten_query}

Output JSON:
{{"queries": ["query1"]}}""" # Modified example output
        self.prompt_multi_query = ChatPromptTemplate.from_template(multi_query_generation_template)
        self.multi_query_chain = self.prompt_multi_query | json_llm | JsonOutputParser() # Use JsonOutputParser directly

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
            | string_llm
            | StrOutputParser()
        )

    def generate_queries(self, input_dict: dict) -> List[str]:
        # --- RESTORED Multi-Query Logic (Simplified) ---
        question = input_dict["question"]
        logger.info(f"Generating queries for original question: {question}")
        generated_queries = [] # Initialize
        try:
            # Retrieve context (unchanged)
            context_query = f"General information related to: {question}"
            retrieved_docs = self.hybrid_retrieve_for_query(context_query, top_k=1)
            if not retrieved_docs:
                logger.warning("No documents retrieved for initial context summary. Using original query only.")
                return [question]

            first_document_split = retrieved_docs[0].page_content
            document_summary = self.document_summary_chain.invoke({"document_chunk": first_document_split})
            logger.info(f"Document Summary for query generation: {document_summary}")

            # Get query intent (unchanged)
            try:
                query_intent_result = self.query_intent_chain.invoke({"document_chunk": first_document_split, "question": question})
                logger.info(f"Query Intent Analysis: {query_intent_result}")
                query_intent_str = json.dumps(query_intent_result)
            except Exception as e:
                logger.error(f"Error getting query intent: {e}. Proceeding without intent analysis.")
                query_intent_str = "{}"

            # Rewrite query (unchanged)
            try:
                rewritten_query_output = self.rewrite_query_chain.invoke({"question": question, "document_summary": document_summary, "query_intent": query_intent_str})
                rewritten_query = parse_rewritten_query(rewritten_query_output)
                logger.info(f"Rewritten Query: {rewritten_query}")
            except Exception as e:
                logger.error(f"Error rewriting query: {e}. Using original query.")
                rewritten_query = question

            # --- RESTORED: Generate 1 additional query ---
            try:
                multi_query_result = self.multi_query_chain.invoke({
                     "rewritten_query": rewritten_query,
                     "document_summary": document_summary
                })
                # The parser expects 'queries' key, result should be {"queries": ["query1"]}
                if isinstance(multi_query_result, dict) and "queries" in multi_query_result and isinstance(multi_query_result["queries"], list):
                     generated_queries = multi_query_result["queries"]
                     logger.info(f"Generated {len(generated_queries)} additional query/queries: {generated_queries}")
                else:
                     logger.warning(f"Multi-query generation returned unexpected format or no queries: {multi_query_result}")
                     generated_queries = []

            except Exception as e:
                logger.error(f"Error generating multiple queries: {e}. Skipping additional query generation.")
                generated_queries = [] # Ensure it's an empty list on error


            # Combine and deduplicate queries (Original + Rewritten + Generated[0] if exists)
            final_queries = [question, rewritten_query]
            if generated_queries:
                final_queries.append(generated_queries[0]) # Add the first (and only expected) generated query

            seen = set()
            unique_queries = [q for q in final_queries if q and not (q in seen or seen.add(q))]
            logger.info(f"Final Unique Queries for Retrieval: {unique_queries}")

            # Return max 3 queries (Original, Rewritten, 1 Generated)
            return unique_queries[:3]

        except Exception as e:
            logger.error(f"Error in query generation pipeline: {str(e)}")
            return [question] # Fallback to original question

    def format_docs(self, docs: List[Document]) -> str:
        # (Method unchanged)
        formatted_docs = []
        for i, doc in enumerate(docs, start=1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            source_type = metadata.get('type', metadata.get('source_type', 'Unknown'))
            source_name = metadata.get('source', 'N/A')
            if isinstance(source_name, str):
                 if source_name.startswith('http'):
                     try:
                          domain = source_name.split('/')[2]
                          path_part = "/".join(source_name.split('/')[3:])[:30]
                          source_name = f"{domain}/{path_part}..." if path_part else domain
                     except: pass
                 elif len(source_name) > 40:
                      source_name = "..." + source_name[-37:]
            citation = f"[Doc {i} - {source_type.capitalize()}: {source_name}]"
            formatted_docs.append(f"{citation}\n{content}")
        return "\n\n---\n\n".join(formatted_docs)


    def bm25_retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        # --- PARAMETER REDUCTION (Keep k low) ---
        # Default stays 5, but it's called with 5 below.
        if not self.bm25 or not self.bm25_corpus:
            logger.warning("BM25 index not available.")
            return []
        try:
            tokenized_query = self.custom_tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            top_indices_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
            retrieved = []
            for idx, score in top_indices_scores:
                if score > 0:
                    doc_info = self.bm25_index_map.get(idx)
                    if doc_info:
                         doc = Document(page_content=doc_info['content'], metadata=doc_info['metadata'])
                         doc.metadata['bm25_score'] = score
                         retrieved.append(doc)
                    else: logger.warning(f"BM25 index {idx} not found in map.")
            logger.info(f"BM25 retrieved {len(retrieved)} documents for query '{query}' with scores: {[f'{s:.2f}' for _, s in top_indices_scores[:len(retrieved)]]}")
            return retrieved
        except Exception as e:
            logger.error(f"Error during BM25 retrieval for query '{query}': {e}")
            return []

    def web_search_documents(self, query: str) -> List[Document]:
        # (Method unchanged)
        if not self.web_search:
             logger.warning("Web search tool (Tavily) is not configured.")
             return []
        try:
            logger.info(f"Performing web search for: {query}")
            results = self.web_search.invoke(query)
            documents = []
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict) and result.get("content"):
                        documents.append(Document(
                            page_content=result.get("content", ""),
                            metadata={"source": result.get("url", "web_search_result"), "source_type": "web", "title": result.get("title", ""), "url": result.get("url", "")}
                        ))
                logger.info(f"Web search found {len(documents)} documents for query: {query}")
            else:
                logger.warning(f"Web search returned unexpected format: {type(results)}")
            return documents
        except Exception as e:
            logger.error(f"Web search error for query '{query}': {e}")
            return []

    def reciprocal_rank_fusion_direct(self, pine_results: List[Document], bm25_results: List[Document], k: int = 60 ) -> List[Document]:
        # (Method unchanged)
        fused_scores = {}
        doc_map = {}
        def get_doc_key(doc): return hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        for rank, doc in enumerate(pine_results):
            key = get_doc_key(doc)
            if key not in fused_scores: fused_scores[key] = 0; doc_map[key] = doc
            score = 1.0 / (rank + k); fused_scores[key] += score
        for rank, doc in enumerate(bm25_results):
            key = get_doc_key(doc)
            if key not in fused_scores: fused_scores[key] = 0; doc_map[key] = doc
            score = 1.0 / (rank + k); fused_scores[key] += score
        sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
        fused_docs = []
        for key in sorted_keys:
             doc = doc_map[key]; doc.metadata['fusion_score'] = fused_scores[key]; fused_docs.append(doc)
        logger.info(f"RRF Fusion: Input Pinecone={len(pine_results)}, BM25={len(bm25_results)}. Output unique fused={len(fused_docs)}")
        return fused_docs

    def hybrid_retrieve_for_query(self, query: str, top_k=3) -> List[Document]:
        # --- PARAMETER REDUCTION (Keep k low) ---
        logger.info(f"Starting hybrid retrieval for query: '{query}' (target top_k={top_k})")
        try:
            # Retriever already limited to k=3 in init
            pinecone_results = self.retriever.invoke(query)
            logger.info(f"Pinecone retrieved {len(pinecone_results)} documents")
        except Exception as e:
            logger.error(f"Pinecone retrieval error: {e}")
            pinecone_results = []

        # --- PARAMETER REDUCTION (Keep k low) ---
        bm25_results = self.bm25_retrieve(query, top_k=5) # Retrieve 5 for BM25 pool

        if not pinecone_results and not bm25_results:
            logger.warning("No results from either Pinecone or BM25 retrieval.")
            return []

        fused_results = self.reciprocal_rank_fusion_direct(pinecone_results, bm25_results)
        final_results = fused_results[:top_k] # Apply the target top_k (3)
        logger.info(f"Hybrid retrieval returning {len(final_results)} fused documents for query '{query}'")
        return final_results

    def evaluate_document_relevance(self, question: str, document: Document) -> Tuple[float, str]:
        # (Method unchanged)
        try:
            content_preview = document.page_content[:2000]
            response = self.relevance_chain.invoke({"question": question, "document": content_preview})
            score = float(response.get("score", 0.0))
            justification = response.get("justification", "")
            return score, justification
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Error parsing relevance response: {e}. Raw: {response}")
            return 0.0, "Error parsing relevance score."
        except Exception as e:
            logger.error(f"Error evaluating document relevance: {e}")
            return 0.0, "Error during relevance evaluation."

    def create_knowledge_strips(self, question: str, document: Document) -> List[str]:
        # (Method unchanged)
        try:
            content_preview = document.page_content[:3000]
            raw_response = self.strip_chain.invoke({"question": question, "document": content_preview})
            lines = raw_response.split('\n')
            strips = [line.strip().lstrip('-').strip() for line in lines if line.strip().startswith('-')]
            if not strips:
                strips = [line.strip() for line in lines if len(line.strip()) > 10]
            return strips
        except Exception as e:
            logger.error(f"Error creating knowledge strips: {e}")
            return []

    def evaluate_strip_relevance(self, question: str, strip: str) -> Tuple[float, str]:
        # (Method unchanged)
        if not strip: return 0.0, "Empty strip."
        try:
            response = self.strip_relevance_chain.invoke({"question": question, "strip": strip[:500]})
            score = float(response.get("score", 0.0))
            justification = response.get("justification", "")
            return score, justification
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Error parsing strip relevance response: {e}. Raw: {response}")
            return 0.0, "Error parsing score."
        except Exception as e:
            logger.error(f"Error evaluating strip relevance: {e}")
            return 0.0, "Evaluation error."

    def process_with_crag(self, question: str, documents: List[Document]) -> List[Document]:
        # --- PARAMETER REDUCTION (Keep limits low) ---
        logger.info(f"Starting CRAG processing for {len(documents)} retrieved documents.")
        processed_docs_content = {}
        original_doc_metadata = {}
        all_processed_strips = []
        need_web_search = True

        # Check top 3 initial docs
        for doc in documents[:3]:
            doc_score, doc_just = self.evaluate_document_relevance(question, doc)
            source = doc.metadata.get('source', 'Unknown')
            logger.info(f"Doc '{source}' relevance: {doc_score:.2f}. Justification: {doc_just}")

            if doc_score >= self.relevance_threshold:
                need_web_search = False
                strips = self.create_knowledge_strips(question, doc)
                relevant_strips_for_doc = []
                # Process top 5 strips per doc
                for strip in strips[:5]:
                    strip_score, strip_just = self.evaluate_strip_relevance(question, strip)
                    if strip_score >= self.relevance_threshold:
                        relevant_strips_for_doc.append(strip)
                        all_processed_strips.append({'content': strip, 'score': strip_score, 'source_doc': doc})

                if relevant_strips_for_doc:
                    logger.info(f"  Found {len(relevant_strips_for_doc)} relevant strips in doc '{source}'.")
                    if source not in processed_docs_content:
                        processed_docs_content[source] = []
                        original_doc_metadata[source] = doc.metadata
                    processed_docs_content[source].extend(relevant_strips_for_doc)

        web_docs = []
        if need_web_search and self.web_search:
             logger.info("Performing web search as initial documents were not relevant enough.")
             web_docs = self.web_search_documents(question)
             # Process top 2 web results
             for doc in web_docs[:2]:
                 source = doc.metadata.get('source', 'Web Result')
                 strips = self.create_knowledge_strips(question, doc)
                 relevant_strips_for_doc = []
                 # Process top 5 strips per web doc
                 for strip in strips[:5]:
                     strip_score, strip_just = self.evaluate_strip_relevance(question, strip)
                     if strip_score >= self.relevance_threshold:
                         relevant_strips_for_doc.append(strip)
                         all_processed_strips.append({'content': strip, 'score': strip_score, 'source_doc': doc})
                 if relevant_strips_for_doc:
                     logger.info(f"  Found {len(relevant_strips_for_doc)} relevant strips in web doc '{source}'.")
                     if source not in processed_docs_content:
                         processed_docs_content[source] = []
                         original_doc_metadata[source] = doc.metadata
                     processed_docs_content[source].extend(relevant_strips_for_doc)

        final_docs = []
        if processed_docs_content:
             for source, strips in processed_docs_content.items():
                  combined_content = "\n".join(f"- {s}" for s in strips)
                  metadata = original_doc_metadata.get(source, {})
                  metadata["processed_by_crag"] = True
                  metadata["relevant_strips_count"] = len(strips)
                  final_docs.append(Document(page_content=combined_content, metadata=metadata))
             logger.info(f"CRAG assembled {len(final_docs)} documents from processed strips.")

        if not final_docs:
             logger.warning("CRAG processing (including web search) produced no relevant strips. Falling back to top raw retrieved documents.")
             # Fallback to top 2 raw docs
             final_docs = documents[:2]
             for doc in final_docs:
                 doc.metadata["processed_by_crag"] = False

        logger.info(f"CRAG processing finished. Returning {len(final_docs)} documents for final answer generation.")
        return final_docs

    def query(self, question: str) -> Tuple[str, List[Dict]]:
        start_time = time.time()
        logger.info(f"Received query: {question}")
        try:
            # 1. Generate queries (now includes multi-query again, limited to ~3 total)
            gen_start_time = time.time()
            queries = self.generate_queries({"question": question}) # Will generate max 3 queries
            logger.info(f"Query generation took {time.time() - gen_start_time:.2f} seconds.")
            if not queries:
                logger.error("Query generation failed. Aborting.")
                return "Error: Could not generate search queries.", []

            # 2. Retrieve documents (loop runs max 3 times)
            retrieval_start_time = time.time()
            all_retrieved_docs = []
            unique_doc_hashes = set()
            for q in queries: # Loop <= 3 times
                # Retrieve top 3 per query
                docs = self.hybrid_retrieve_for_query(q, top_k=3)
                for doc in docs:
                     doc_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                     if doc_hash not in unique_doc_hashes:
                          all_retrieved_docs.append(doc)
                          unique_doc_hashes.add(doc_hash)
            logger.info(f"Total unique documents retrieved across all queries: {len(all_retrieved_docs)}")
            logger.info(f"Retrieval loop took {time.time() - retrieval_start_time:.2f} seconds.")

            if not all_retrieved_docs:
                logger.warning("No documents found in internal sources. Attempting direct web search.")
                web_fallback_docs = self.web_search_documents(question) if self.web_search else []
                if not web_fallback_docs:
                     return "No relevant information found in indexed documents or web search.", []
                else:
                     logger.info("Using web search results as fallback context.")
                     all_retrieved_docs = web_fallback_docs[:3]


            # 3. Process with CRAG (limits remain reduced)
            crag_start_time = time.time()
            processed_docs = self.process_with_crag(question, all_retrieved_docs)
            logger.info(f"CRAG processing took {time.time() - crag_start_time:.2f} seconds.")

            if not processed_docs:
                logger.error("CRAG processing failed to produce any relevant documents.")
                return "Could not find relevant information after relevance checking and processing.", []

            # 4. Extract sources (unchanged)
            sources_for_display = self.extract_sources_from_docs(processed_docs)

            # 5. Generate final answer (unchanged)
            final_answer_start_time = time.time()
            logger.info("Generating final answer...")
            final_answer = self.final_rag_chain.invoke({
                "question": question,
                "processed_docs": processed_docs
            })
            answer_str = final_answer if isinstance(final_answer, str) else json.dumps(final_answer, indent=2)
            logger.info(f"Final answer generation took {time.time() - final_answer_start_time:.2f} seconds.")

            logger.info(f"Total query processing time: {time.time() - start_time:.2f} seconds.")
            return answer_str, sources_for_display

        except Exception as e:
            logger.exception(f"Critical error during query processing pipeline: {str(e)}")
            return f"An error occurred: {str(e)}", []


    def clear_index(self):
        # (Method unchanged)
        logger.warning(f"Attempting to clear all data from Pinecone index '{self.index_name}' and reset local state.")
        try:
            logger.info(f"Deleting vectors from Pinecone index: {self.index_name}")
            self.index.delete(delete_all=True)
            self.bm25_corpus = []
            self.bm25_index_map = {}
            self.bm25 = None
            logger.info("Local BM25 data cleared.")
            time.sleep(2)
            logger.info(f"Pinecone index '{self.index_name}' cleared successfully.")
            return True
        except Exception as e:
            logger.error(f"Error clearing Pinecone index '{self.index_name}': {str(e)}")
            raise Exception(f"Failed to clear index '{self.index_name}': {str(e)}")

# Keep the main function definition, but ensure it's only called when script is run directly
def main():
    # This block should only run if you execute this file directly,
    # not when imported by Streamlit (unless Streamlit also runs it directly somehow).
    try:
        # Ensure logging is configured *before* RAGS is initialized
        # If you configure logging in orchestrator.py, this basicConfig might be redundant
        # or cause issues if called twice. Consider removing it if orchestrator.py handles it.
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        tavily_key = os.getenv("TAVILY_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY") # Pinecone key check is implicit in setup_pinecone

        # Note: RAGS initialization might happen twice if Streamlit also imports and creates an instance.
        print("Initializing RAG system (from main)...") # Added identifier
        rag_system = RAGS(tavily_api_key=tavily_key)

        # sources_to_load = ["https://lilianweng.github.io/posts/2024-11-28-reward-hacking/"]
        # Use a different source for testing variety if needed
        sources_to_load = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]

        if sources_to_load:
            print(f"Loading content from: {sources_to_load}")
            documents = rag_system.load_content(sources_to_load)
            print(f"Loaded {len(documents)} document chunks.")
        else:
             print("No sources provided to load. Querying existing index (if any).")

        # question = "define reward hacking"
        question = "Explain task decomposition in autonomous agents based on the provided context."
        print(f"\nQuerying with: '{question}'")

        answer, sources_info = rag_system.query(question)

        print("\n------ Final Answer ------")
        print(answer)
        print("\n------ Sources Used (Summary) ------")
        if sources_info:
            for source in sources_info:
                print(f"- Type: {source.get('type', 'N/A')}, Source: {source.get('source', 'N/A')}, Relevance: {source.get('relevance', 'N/A')}")
        else:
            print("No sources were cited for this answer.")

    except Exception as e:
         print(f"\nAn error occurred during the main execution: {e}")
         # Use logger for traceback if logging is configured
         logger.exception("Main execution failed.")


# Ensure main() only runs when this script is executed directly
if __name__ == "__main__":
    main()
