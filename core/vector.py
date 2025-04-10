import os
import getpass
import time
import logging
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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores.pinecone import Pinecone as PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
from langchain_community.tools.tavily_search import TavilySearchResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_rewritten_query(raw_output: str) -> str:
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
    Attempt to parse output as JSON.
    Expect a JSON object with key "queries" mapping to an array of exactly 4 query strings.
    Fallback: split by newlines.
    """
    logger.info(f"Raw multi-query output: {raw_output}")
    try:
        data = json.loads(raw_output)
        if isinstance(data, dict) and "queries" in data and isinstance(data["queries"], list):
            queries = [str(item) for item in data["queries"] if isinstance(item, str)]
            return queries
    except json.JSONDecodeError:
        lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
        return lines
    return []


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass


class DocumentProcessor:
    """Handles loading and processing of various document formats"""

    def __init__(self):
        self.supported_extensions = {
            '.pdf': self.load_pdf,
            '.csv': self.load_csv,
            '.txt': self.load_text,
            'url': self.load_url
        }

    def load_pdf(self, file_path: str) -> List[Document]:
        """Post-process extracted text from PDF."""
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.page_content = self.clean_pdf_text(doc.page_content)
                doc.metadata.update({
                    'source_type': 'pdf',
                    'file_path': file_path
                })
            return docs
        except Exception as e:
            logger.error(f"PDF loading error: {str(e)}")
            raise Exception(f"Error loading PDF {file_path}: {str(e)}")

    def clean_pdf_text(self, text: str) -> str:
        """Clean common PDF extraction artifacts."""
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())
        return text

    def load_csv(self, file_path: str) -> List[Document]:
        """Load a CSV file."""
        logger.info(f"Loading CSV: {file_path}")
        try:
            loader = CSVLoader(file_path)
            docs = loader.load()
            if not docs:
                raise DocumentProcessingError(f"No content extracted from CSV: {file_path}")
            logger.info(f"Successfully loaded CSV with {len(docs)} rows")
            return docs
        except Exception as e:
            raise DocumentProcessingError(f"Failed to process CSV {file_path}: {str(e)}")

    def load_text(self, file_path: str) -> List[Document]:
        """Load and process text files with encoding detection."""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
            document = Document(
                page_content=text,
                metadata={
                    'source_type': 'text',
                    'file_path': file_path
                }
            )
            return [document]
        except Exception as e:
            raise Exception(f"Error loading text file {file_path}: {str(e)}")

    def load_url(self, url: str) -> List[Document]:
        """Load content from URLs."""
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
            )
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_type'] = 'url'
                doc.metadata['url'] = url
            return documents
        except Exception as e:
            raise Exception(f"Error loading URL {url}: {str(e)}")

    def load_document(self, source: str) -> List[Document]:
        """
        Load a document from a file path or URL.
        Args:
            source: Path to file or URL.
        Returns:
            List of Document objects.
        """
        if source.startswith(('http://', 'https://')):
            return self.supported_extensions['url'](source)

        _, extension = os.path.splitext(source)
        extension = extension.lower()

        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {extension}")

        return self.supported_extensions[extension](source)




class RAGS:
    def __init__(
        self,
        index_name: str = "rag",
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "qwen2.5-coder:1.5b",
        dimension: int = 1024,
        relevance_threshold: float = 0.3,  # Adjust threshold as needed
        tavily_api_key: str = None
    ):
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.dimension = dimension
        self.relevance_threshold = relevance_threshold
        self.document_loader = DocumentProcessor()

        self.setup_pinecone()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        self.retriever = self.vector_store.as_retriever()

        # For BM25, we store the document texts and will build the index once after processing all sources.
        self.bm25_corpus = []
        self.bm25 = None


        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        self.web_search = TavilySearchResults(k=3) # Increased k for more web results
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=0.2,
            format="json",
            system="You are a research assistant skilled in analyzing academic papers. Always respond with valid JSON."
        )
        self.setup_rag_chain()
        self.setup_crag_chains()

    def setup_pinecone(self):
        """Set up connection to Pinecone"""
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            pinecone_api_key = getpass.getpass("Enter your Pinecone API key: ")
            os.environ["PINECONE_API_KEY"] = pinecone_api_key
            
        try:
            self.pc = Pinecone(api_key=pinecone_api_key)
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


    def load_web_content(self, urls):
        return self.load_content(urls)

    def load_content(self, sources: List[str]):
        """Load and process content from various sources."""
        all_documents = []
        corpus_texts = []  # temporary list to collect texts for BM25

        for source in sources:
            try:
                logger.info(f"Starting to process source: {source}")
                docs = self.document_loader.load_document(source)
                logger.info(f"Loaded {len(docs)} raw documents from {source}")

                # Increase chunk size to preserve more context
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3500,  # Increased chunk size for more context
                    chunk_overlap=700, # Increased overlap to maintain continuity
                    separators=["\n\n", "\n", " ", ""]  # Better paragraph splitting
                )

                splits = text_splitter.split_documents(docs)
                logger.info(f"Created {len(splits)} splits from {source}")

                splits = [split for split in splits if split.page_content.strip()]
                logger.info(f"Have {len(splits)} non-empty splits")

                try:
                    ids = self.vector_store.add_documents(splits)
                    logger.info(f"Added {len(ids)} documents to vector store")
                except Exception as e:
                    logger.error(f"Error adding to vector store: {e}")
                    raise

                # Collect texts for BM25 indexing
                new_texts = [doc.page_content for doc in splits]
                corpus_texts.extend(new_texts)
                all_documents.extend(splits)

            except Exception as e:
                logger.error(f"Error processing {source}: {e}")
                raise

        if not all_documents:
            raise ValueError("No documents were successfully processed")

        # Build (or rebuild) the BM25 index using the complete corpus from all sources.
        self.bm25_corpus = corpus_texts
        tokenized_corpus = [self.custom_tokenize(text) for text in self.bm25_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(self.bm25_corpus)} documents")

        return all_documents

    def custom_tokenize(self, text: str) -> List[str]:
        """Tokenize text by lowercasing and removing punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return tokens

    

    def extract_sources_from_docs(self, docs: List[Document]) -> List[Dict]:
        """Extract source information from documents."""  
        sources = []
        for doc in docs:
            source_info = {
                'content': doc.page_content[:500] + "...",  # Increased preview length
                'type': doc.metadata.get('source_type', 'unknown'),
                'relevance': doc.metadata.get('relevance_score', None),
                'source': None
            }
            
            # Prioritize URL source
            if 'url' in doc.metadata:
                source_info.update({
                    'source': doc.metadata['url'],
                    'type': 'web'
                })
            elif 'file_path' in doc.metadata:
                source_info.update({
                    'source': os.path.basename(doc.metadata['file_path']),
                    'type': 'file'
                })
            elif 'source' in doc.metadata:  # For web search results
                source_info['source'] = doc.metadata['source']
                
            sources.append(source_info)
        return sources
    

    

    def setup_crag_chains(self):
        # CRAG relevance prompt for judging document relevance
        relevance_template = ("""
            Evaluate the relevance of the provided document to the question. On a scale of 0 to 1 (with 1 being highly relevant), output a JSON object with the key "score". Provide a brief justification to explain your score.

            Question: {question}
            Document: {document}

            Output:
                {{ "score": <relevance_score>, "justification": "<brief_justification>" }}
                                """
        )
        self.relevance_prompt = ChatPromptTemplate.from_template(relevance_template)
        self.relevance_chain = self.relevance_prompt | self.llm | JsonOutputParser()

        strip_template = (
            "Break this document into distinct knowledge units (strips) that relate to answering the question. "
            "Each strip should be a complete and coherent bullet point. Aim for conciseness but ensure each strip retains enough context to be meaningful. List each strip on a new line prefixed with a dash (-).\n\n"
            "Question: {question}\n"
            "Document: {document}\n\n"
            "Output:"
        )
        self.strip_prompt = ChatPromptTemplate.from_template(strip_template)
        self.strip_chain = self.strip_prompt | self.llm | StrOutputParser()

        strip_relevance_template = (
            "Evaluate the relevance of the following knowledge strip to the question. Output only a valid JSON object "
            "with a single key \"score\". The score must be a floating point number between 0 and 1, where 0 means "
            "the strip is completely irrelevant and 1 means it is highly relevant. Justify your score briefly.\n\n"
            "Question: {question}\n"
            "Knowledge Strip: {strip}\n\n"
            "Output: {{ \"score\": <strip_relevance_score>, "
            "\"justification\": \"<brief_justification>\" }}"
        )
        self.strip_relevance_prompt = ChatPromptTemplate.from_template(strip_relevance_template)
        self.strip_relevance_chain = self.strip_relevance_prompt | self.llm | JsonOutputParser()

    def setup_rag_chain(self):
        query_intent_template = """
            Analyze this query to identify the true search intent and generate appropriate search terms.
            Don't literally search for generic terms like "summary", "explain", or "tell me about".
            Instead, extract the key concepts that need to be found in the document context.

            Document Context: {document_chunk}
            Query: {question}

            Return your analysis in JSON format with the following structure:
            {{"intent": "what the user actually wants to do with the information",
              "search_terms": ["list", "of", "actual", "concepts", "to", "search", "for"],
              "expanded_query": "a more specific search query that will find relevant content"}}

            

            JSON Response:"""

        self.query_intent_prompt = ChatPromptTemplate.from_template(query_intent_template)
        self.query_intent_chain = self.query_intent_prompt | self.llm | JsonOutputParser()

        document_summary_template = """
        Summarize the content of the following document chunk, focusing on its main topics and themes.
        Extract key concepts, methodologies, and specific examples mentioned. 
        Keep the summary concise but informative, around 3-4 sentences.

        Document Chunk: {document_chunk}

        Summary:
        """
        self.document_summary_prompt = ChatPromptTemplate.from_template(document_summary_template)
        self.document_summary_chain = self.document_summary_prompt | self.llm | StrOutputParser()

        # Modified query rewrite template to use the processed intent
        query_rewrite_template = """
You are a helpful research assistant that rewrites queries to be clearer and more specific, based on the document context provided below.

Please carefully review the following information:

**Document Summary:**  
{document_summary}

**Query Intent Analysis:**  
{query_intent}

**Original Query:**  
{question}

Based on the above context, rewrite the query so that it:
1. Focuses on the key concepts identified.
2. Uses terminology directly from the document summary.
3. Is specific and tailored to the document’s subject matter.
4. Avoids generic terms like "summary", "explain", or "tell me about".

IMPORTANT: Output your rewritten query in strict JSON format as follows:  
{{"query": "your rewritten query"}}

Rewritten Query:
"""

        self.prompt_rewrite = ChatPromptTemplate.from_template(query_rewrite_template)
        self.rewrite_query_chain = self.prompt_rewrite | self.llm | StrOutputParser()

        self.rewrite_query_chain = self.prompt_rewrite | self.llm | StrOutputParser() 


        multi_query_generation_template = ( 
            "Generate exactly 4 distinct, yet related, search queries related to the research topic below. Each query should explore a different facet or sub-aspect of the topic, **while staying strictly relevant to the themes and context described in the provided document summary, especially concerning task decomposition in agents.** "
            "These queries will be used to retrieve academic papers and research documents that are closely related to the document's content. Make the queries detailed, specific to academic contexts, and highly relevant to the document summary.\n\n"
            "Document Summary: {document_summary}\n\n" # Added document summary context
            "Topic: {rewritten_query}\n\n"
            "Output: {{\"queries\": [\"query1\", \"query2\", \"query3\", \"query4\"]}}" # Specify JSON output format for multi-queries
        )
        self.prompt_multi_query = ChatPromptTemplate.from_template(multi_query_generation_template) # Assign prompt_multi_query to self


        final_template = """
You are a highly intelligent AI research assistant specialized in deeply analyzing and explaining academic papers and research topics.
Using the provided context from multiple academic sources, generate a detailed, comprehensive, and well-structured explanation of at least 600-800 words (8-10 paragraphs) that:

1. **Directly and thoroughly answers the question:** {question}
2. **Synthesizes information from all provided sources** to create a cohesive and integrated explanation.
3. **Provides specific examples and evidence** from the context to support your points.
4. **Clearly explains any technical terms or jargon** that might be present.
5. **Includes detailed in-line citations** to clearly indicate the source of information. Use the following format:
   - For web sources: ([Document X - URL: domain.com])
   - For papers: ([Document X - Paper: title])
   - For other files: ([Document X - File: filename])

Structure your response in Markdown format. Ensure a logical flow and clear organization into paragraphs.

### Context:
{context}

### Detailed Analysis:"""

        prompt_final = ChatPromptTemplate.from_template(final_template)
        context_chain = RunnablePassthrough(lambda input_dict: self.format_docs(input_dict['queries']))
        self.final_rag_chain = ({
            "context": context_chain,
            "question": itemgetter("question"),
            "retrieved_documents": RunnablePassthrough(lambda input_dict: self.hybrid_retrieve_for_query(input_dict['question'])) # **REPLACED "queries" with "retrieved_documents"**
        } | prompt_final | self.llm | StrOutputParser()) # Removed JsonOutputParser for string output


    def generate_queries(self, input_dict: dict) -> List[str]:
        try:
            question = input_dict["question"]

            # Get document context
            retrieved_docs = self.hybrid_retrieve_for_query("document summary", top_k=1)

            if not retrieved_docs:
                logger.warning("No documents retrieved for summary context")
                return [question]

            first_document_split = retrieved_docs[0].page_content

            # Get document summary
            document_summary = self.document_summary_chain.invoke({"document_chunk": first_document_split})
            logger.info(f"Document Summary: {document_summary}")

            # Get query intent
            query_intent_result = self.query_intent_chain.invoke({
                "document_chunk": first_document_split,
                "question": question
            })
            logger.info(f"Query Intent Analysis: {query_intent_result}")

            # Rewrite query with both context and intent
            rewritten_query_output = self.rewrite_query_chain.invoke({
                "question": question,
                "document_summary": document_summary,
                "query_intent": json.dumps(query_intent_result)
            })
            rewritten_query = parse_rewritten_query(rewritten_query_output)
            logger.info(f"Rewritten Query: {rewritten_query}")

            # Generate multiple queries
            raw_queries = (self.prompt_multi_query | self.llm | StrOutputParser()).invoke({
                "rewritten_query": rewritten_query,
                "document_summary": document_summary
            })
            queries = parse_queries(raw_queries)

            if len(queries) != 4:
                logger.warning("Did not get exactly 4 queries; falling back to using original and rewritten query.")
                return [question, rewritten_query]

            final_queries = [rewritten_query] + queries
            seen = set()
            unique_queries = [q for q in final_queries if not (q in seen or seen.add(q))]
            logger.info(f"Final Queries: {unique_queries}")
            return unique_queries

        except Exception as e:
            logger.error(f"Error in query generation: {str(e)}")
            return [question]

    def format_docs(self, docs: List[Union[Document, str]]) -> str:
        formatted_docs = []
        for i, doc in enumerate(docs, start=1):
            if isinstance(doc, Document):
                content = doc.page_content.strip()
                source_info = ""

                # Extract source information
                if 'url' in doc.metadata:
                    source_info = f"Source URL: {doc.metadata['url']}"
                elif 'file_path' in doc.metadata:
                    source_info = f"Source File: {os.path.basename(doc.metadata['file_path'])}"
                elif 'source' in doc.metadata:
                    source_info = f"Source: {doc.metadata['source']}"

                formatted_docs.append(f"Document {i}:\n{content}\n[{source_info}]")
            elif isinstance(doc, str):
                formatted_docs.append(f"Document {i}:\n{doc.strip()}")
        return "\n\n---\n\n".join(formatted_docs)

    def bm25_retrieve(self, query: str, top_k: int = 8) -> List[Document]: # Increased top_k for BM25
        if not self.bm25:
            logger.warning("BM25 index not initialized.")
            return []
        tokenized_query = self.custom_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # Retrieve top_k indices based on BM25 scores
        top_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        retrieved = [Document(page_content=self.bm25_corpus[idx]) for idx, _ in top_docs]
        logger.info(f"BM25 scores for query '{query}': {[scores[idx] for idx, _ in top_docs]}")
        return retrieved

    def web_search_documents(self, query: str) -> List[Document]:
        try:
            results = self.web_search.invoke(query)
            documents = [
                Document(page_content=result.get("content", ""), metadata={"source": "web", "url": result.get("url", "")})
                for result in results if isinstance(result, dict)
            ]
            logger.info(f"Web search found {len(documents)} documents for query: {query}")
            return documents
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

    def reciprocal_rank_fusion_direct(self, pine_results: List[Document], bm25_results: List[Document], k: int = 5 ) -> List[Document]:
        """
        Directly fuse two ranked lists (Pinecone and BM25) using Reciprocal Rank Fusion.
        Each document's score is the sum of reciprocal ranks from both sources.
        """
        fused_scores = {}
        doc_map = {}
        # Process Pinecone results
        for rank, doc in enumerate(pine_results):
            key = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            if key not in fused_scores:
                fused_scores[key] = 0
                doc_map[key] = doc
            score = 1 / (rank + k)
            fused_scores[key] += score
            logger.debug(f"Pinecone doc {key[:6]} rank {rank} -> score {score:.4f}")
        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            key = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            if key not in fused_scores:
                fused_scores[key] = 0
                doc_map[key] = doc
            score = 1 / (rank + k)
            fused_scores[key] += score
            logger.debug(f"BM25 doc {key[:6]} rank {rank} -> score {score:.4f}")
        # Log the combined scores for debugging
        logger.info("Fusion scores:")
        for key, score in fused_scores.items():
            logger.info(f"Doc {key[:6]}...: {score:.4f}")
        # Sort by final score (highest first)
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, score in sorted_docs]

    def hybrid_retrieve_for_query(self, query: str, top_k=4) -> List[Document]: # Increased top_k for hybrid retrieval
        """Hybrid retrieval for a query using direct RRF fusion."""
        logger.info(f"Starting hybrid retrieval for query: '{query}'")
        

        try:
            pinecone_results = self.retriever.invoke(query, k=top_k)
            logger.info(f"Pinecone retrieved {len(pinecone_results)} documents")
            if pinecone_results:
                logger.info(f"Sample Pinecone result: {pinecone_results[0].page_content[:200]}")
        except Exception as e:
            logger.error(f"Pinecone retrieval error: {e}")
            pinecone_results = []

        try:
            bm25_results = self.bm25_retrieve(query, top_k=4)
            logger.info(f"BM25 retrieved {len(bm25_results)} documents")
            if bm25_results:
                logger.info(f"Sample BM25 result: {bm25_results[0].page_content[:200]}")
        except Exception as e:
            logger.error(f"BM25 retrieval error: {e}")
            bm25_results = []
    
        if not pinecone_results and not bm25_results:
            logger.error("No results from either retriever!")
            return []

        fused_results = self.reciprocal_rank_fusion_direct(pinecone_results, bm25_results, k=2)[:top_k]
        logger.info(f"Final fused results: {len(fused_results)} documents")
        if fused_results:
            logger.info(f"Sample fused result: {fused_results[0].page_content[:200]}")
        return fused_results

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
            lines = raw_response.split('\n')
            strips = [line.strip().lstrip('-').strip() for line in lines if line.strip() and line.startswith('-')]
            if not strips:
                strips = [line.strip() for line in lines if line.strip()]
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
        logger.info("Starting CRAG document processing")
        processed_docs = []
        need_web_search = True

        for doc in documents[:6]:
            relevance_score = self.evaluate_document_relevance(question, doc)
            logger.info(f"Document relevance score: {relevance_score}")
            if relevance_score >= self.relevance_threshold:
                need_web_search = False
                strips = self.create_knowledge_strips(question, doc)
                relevant_strips = []
                for strip in strips[:5]:
                    strip_score = self.evaluate_strip_relevance(question, strip)
                    logger.info(f"Knowledge strip relevance score: {strip_score} for strip: {strip}")
                    if strip_score >= self.relevance_threshold:
                        relevant_strips.append(strip)
                if relevant_strips:
                    processed_docs.append(Document(
                        page_content="\n".join(relevant_strips),
                        metadata={**doc.metadata, "relevance_score": relevance_score, "processed_by_crag": True}
                    ))
            elif relevance_score == -1:
                need_web_search = True

        if need_web_search or not processed_docs:
            logger.info("CRAG processing did not yield sufficient results; performing web search fallback.")
            web_docs = self.web_search_documents(question)[:5]
            for doc in web_docs:
                strips = self.create_knowledge_strips(question, doc)
                relevant_strips = []
                for strip in strips[:5]:
                    strip_score = self.evaluate_strip_relevance(question, strip)
                    logger.info(f"Web knowledge strip relevance score: {strip_score} for strip: {strip}")
                    if strip_score >= self.relevance_threshold:
                        relevant_strips.append(strip)
                if relevant_strips:
                    processed_docs.append(Document(
                        page_content="\n".join(relevant_strips),
                        metadata={**doc.metadata, "source": "web_search", "processed_by_crag": True}
                    ))

        if not processed_docs:
            logger.warning("CRAG processing and web search fallback produced no processed documents; falling back to top retrieved documents.")
            processed_docs = documents[:3]
        logger.info(f"CRAG processing complete. Found {len(processed_docs)} relevant documents")
        return processed_docs


    def query(self, question: str) -> Tuple[str, List[Dict]]:
        """Modified query method that returns both the answer and source information."""
        try:
            logger.info(f"Processing question: {question}")
            queries = self.generate_queries({"question": question})[:3]
            if not queries:
                raise ValueError("No queries generated")
            
            all_retrieved_docs = []
            for query in queries:
                docs = self.hybrid_retrieve_for_query(query, top_k=4)
                if docs:
                    all_retrieved_docs.extend(docs)

            if not all_retrieved_docs:
                return "No relevant information found in the indexed documents.", []

            relevant_docs = self.process_with_crag(question, all_retrieved_docs)
            if not relevant_docs:
                return "Could not find relevant information after CRAG processing.", []

            # Extract source information before processing the answer
            sources = self.extract_sources_from_docs(relevant_docs)

            raw_answer = self.final_rag_chain.invoke({
                "question": question,
                "queries": relevant_docs
            })

            answer = raw_answer if isinstance(raw_answer, str) else json.dumps(raw_answer, indent=2)
            
            return answer, sources

        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}", []   
    

    def clear_index(self):
        """
        Clears the Pinecone index by deleting all vectors.
        Should be called when you want to start fresh with new documents.
        """
        try:
            # Delete all vectors in the index
            self.vector_store.delete(delete_all=True)

            # Reset any stored documents or cached data
            self.documents = []

            # Wait a moment for deletion to complete
            time.sleep(1)

            return True

        except Exception as e:
            logging.error(f"Error clearing Pinecone index: {str(e)}")
            raise Exception(f"Failed to clear index: {str(e)}")


def main():
    rag_system = RAGS(tavily_api_key="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")
    sources = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    try:
        documents = rag_system.load_content(sources)
        logger.info(f"Loaded {len(documents)} document chunks from the provided sources.")
    except Exception as e:  
        logger.error(f"Error loading content: {e}")
        documents = []
    question = "explain about task decomposition in agents"
    try:
        answer, sources = rag_system.query(question)
        print("Final Answer:")
        print(answer)
        print("\nSources Used:")
        for source in sources:
            print(f"- Type: {source['type']}")
            print(f"  Source: {source.get('source', 'Unknown')}")
            print(f"  Relevance: {source.get('relevance_score', 'N/A')}")
            print(f"  Preview: {source['content_preview']}\n")
    except Exception as e:
        print(f"An error occurred while processing the query: {e}")

if __name__ == "__main__":
    main()

