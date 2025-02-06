import os
import getpass
import time
import logging
from typing import List, Union, Dict
from operator import itemgetter
import hashlib
import json
import sys
import chardet
import bs4

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    WebBaseLoader
)
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

# Set the USER_AGENT environment variable
os.environ["USER_AGENT"] = "MyRAGS/1.0 (your_email@example.com)"


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
            '.docx': self.load_docx,
            '.doc': self.load_docx,
            '.csv': self.load_csv,
            '.txt': self.load_text,
            'url': self.load_url
        }

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and process PDF documents"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_type'] = 'pdf'
                doc.metadata['file_path'] = file_path
            return documents
        except Exception as e:
            raise Exception(f"Error loading PDF {file_path}: {str(e)}")

    def load_docx(self, file_path: str) -> List[Document]:
        """Load and process DOCX/DOC documents"""
        try:
            try:
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
            except Exception:
                loader = UnstructuredWordDocumentLoader(file_path)
                documents = loader.load()
            
            for doc in documents:
                doc.metadata['source_type'] = 'docx'
                doc.metadata['file_path'] = file_path
            return documents
        except Exception as e:
            raise Exception(f"Error loading DOCX {file_path}: {str(e)}")

    def load_csv(self, file_path: str) -> List[Document]:
        """Load a CSV file"""
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
        """Load and process text files with encoding detection"""
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
        """Load content from URLs"""
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
        Load a document from a file path or URL
        Args:
            source: Path to file or URL
        Returns:
            List of Document objects
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
        relevance_threshold: float = 0.5,  # You might lower this threshold if needed
        tavily_api_key: str = None
    ):
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.dimension = dimension
        self.relevance_threshold = relevance_threshold
        self.document_loader = DocumentProcessor()

        self.setup_pinecone()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        self.retriever = self.vector_store.as_retriever()
        self.bm25_corpus = []
        self.bm25 = None

        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        self.web_search = TavilySearchResults(k=3)
        # Increase max_tokens for longer outputs
        self.llm = ChatOllama(model=self.llm_model, format="json", temperature=0.2, max_tokens=1024)
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

    def load_web_content(self, urls):
        return self.load_content(urls)
    
    def load_content(self, sources: List[str]):
        """Load and process content from various sources"""
        all_documents = []

        for source in sources:
            try:
                logger.info(f"Starting to process source: {source}")
                docs = self.document_loader.load_document(source)
                logger.info(f"Loaded {len(docs)} raw documents from {source}")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=300
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
                
                texts = [doc.page_content for doc in splits]
                self.bm25_corpus.extend(texts)
                self.bm25 = BM25Okapi([text.split() for text in self.bm25_corpus])
                logger.info(f"Updated BM25 index with {len(texts)} new documents")

                all_documents.extend(splits)

            except Exception as e:
                logger.error(f"Error processing {source}: {e}")
                raise

        if not all_documents:
            raise ValueError("No documents were successfully processed")

        return all_documents

    def setup_crag_chains(self):
        # Updated relevance prompt for CRAG: simpler instructions to judge document relevance
        relevance_template = (
            "Evaluate the relevance of the provided document to the question. "
            "If the document contains sufficient information that directly addresses the question, output a JSON object with 'score' set to 1; otherwise, set 'score' to 0.\n\n"
            "Question: {question}\n"
            "Document: {document}\n\n"
            "Output:"
        )
        self.relevance_prompt = ChatPromptTemplate.from_template(relevance_template)
        self.relevance_chain = self.relevance_prompt | self.llm | JsonOutputParser()

        strip_template = (
            "Break this document into distinct knowledge units (strips) that relate to answering the question. "
            "Each strip should be a complete and coherent bullet point. List each strip on a new line prefixed with a dash (-).\n\n"
            "Question: {question}\n"
            "Document: {document}\n\n"
            "Output:"
        )
        self.strip_prompt = ChatPromptTemplate.from_template(strip_template)
        self.strip_chain = self.strip_prompt | self.llm | StrOutputParser()

        strip_relevance_template = (
            "Evaluate the relevance of the following knowledge strip to the question. Output only a valid JSON object "
            "with a single key \"score\". The score must be a floating point number between 0 and 1, where 0 means "
            "the strip is completely irrelevant and 1 means it is highly relevant.\n\n"
            "Question: {question}\n"
            "Knowledge Strip: {strip}\n\n"
            "Output:"
        )
        self.strip_relevance_prompt = ChatPromptTemplate.from_template(strip_relevance_template)
        self.strip_relevance_chain = self.strip_relevance_prompt | self.llm | JsonOutputParser()

    def setup_rag_chain(self):
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
            "Generate exactly 4 distinct search queries related to the topic below. Each query should cover a different aspect of the topic. "
            "Output a JSON object with a key \"queries\" mapping to an array of exactly 4 query strings. Do not include any additional keys.\n\n"
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
                if len(queries) != 4:
                    logger.warning("Did not get exactly 4 queries; falling back to using the original and rewritten query.")
                    return [input_dict["question"], rewritten_query]
                final_queries = [rewritten_query] + queries
                seen = set()
                unique_queries = [q for q in final_queries if not (q in seen or seen.add(q))]
                logger.info(f"Final Queries: {unique_queries}")
                return unique_queries
            except Exception as e:
                logger.error(f"Error in query generation: {str(e)}")
                return [input_dict["question"]]

        self.generate_queries = generate_queries

        final_template = """ 
You are a highly intelligent AI assistant specialized in analyzing and explaining concepts.
Provide a clear and detailed response using only the provided context.

### Context:  
{context}  

### Question:  
{question}  

### Instructions:  
- Focus on explaining the key concepts and their relationships
- Use clear, natural language (do not output JSON or other structured formats)
- If the context contains only partial information, explain what aspects are covered
- If the context doesn't provide enough information, say so directly
- Keep the response well-structured but conversational
"""

        prompt_final = ChatPromptTemplate.from_template(final_template)
        context_chain = RunnablePassthrough(lambda queries: self.format_docs(queries)) | (lambda docs: self.format_docs(docs) if isinstance(docs, list) else str(docs))
        self.final_rag_chain = ({
                "context": context_chain,
                "question": itemgetter("question"),
            } | prompt_final | self.llm | StrOutputParser())

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

    def bm25_retrieve(self, query: str, top_k: int = 4) -> List[Document]:
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
                Document(page_content=result.get("content", ""), metadata={"source": "web", "url": result.get("url", "")})
                for result in results if isinstance(result, dict)
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
    
    
    def combine_scores(self, pine: List[Document], bm25: List[Document]):
        """
        Combine the scores of the two retrieval methods.
        """
        combined = {}
        doc_map = {}
        for rank, doc in enumerate(pine):
            key = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            if key not in doc_map:
                doc_map[key] = doc
                combined[key] = 0
            combined[key] += 0.6 * (1 / (rank + 1))
        
        for rank, doc in enumerate(bm25):
            key = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            if key not in doc_map:
                doc_map[key] = doc
                combined[key] = 0
            combined[key] += 0.4 * (1 / (rank + 1))
        sorted_res = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in sorted_res]

    def hybrid_retrieve_for_query(self, query: str, top_k=4) -> List[Document]:
        """hybrid retrieval for a query"""
        logger.info(f"Starting hybrid retrieval for query: '{query}'")

        try:
            try:
                stats = self.vector_store.describe_index_stats()
                logger.info(f"Vector store stats: {stats}")
                if stats.get('total_vector_count', 0) == 0:
                    logger.error("Vector store is empty!")
                    return []
            except AttributeError:
                logger.warning("describe_index_stats not available on the vector store; proceeding without stats check.")
            except Exception as e:
                logger.error(f"Error checking vector store: {e}")
        except Exception as e:
            logger.error(f"Error in vector store verification: {e}")

        try:
            pinecone_results = self.retriever.invoke(query, k=top_k)
            logger.info(f"Pinecone retrieved {len(pinecone_results)} documents")
            logger.info(f"Sample Pinecone result: {pinecone_results[0].page_content[:200] if pinecone_results else 'None'}")
        except Exception as e:
            logger.error(f"Pinecone retrieval error: {e}")
            pinecone_results = []

        try:
            bm25_results = self.bm25_retrieve(query, top_k=top_k)
            logger.info(f"BM25 retrieved {len(bm25_results)} documents")
            logger.info(f"Sample BM25 result: {bm25_results[0].page_content[:200] if bm25_results else 'None'}")
        except Exception as e:
            logger.error(f"BM25 retrieval error: {e}")
            bm25_results = []

        if not pinecone_results and not bm25_results:
            logger.error("No results from either retriever!")
            return []

        combined_results = self.combine_scores(pinecone_results, bm25_results)
        fused = self.reciprocal_rank_fusion([combined_results])[:top_k]

        if not fused:
            logger.error("No results after fusion!")
            return []

        logger.info(f"Final fused results: {len(fused)} documents")
        logger.info(f"Sample fused result: {fused[0].page_content[:200] if fused else 'None'}")

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

        for doc in documents[:4]:
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
            web_docs = self.web_search_documents(question)[:3]
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


    def query(self, question: str) -> str:
        """Final query answering"""
        try:
            logger.info(f"Processing question: {question}")
            queries = self.generate_queries({"question": question})[:3]
            if not queries:
                raise ValueError("No queries generated")
            logger.info(f"Generated queries: {queries}")

            all_retrieved_docs = []
            for query in queries:
                docs = self.hybrid_retrieve_for_query(query, top_k=3)
                if docs:
                    all_retrieved_docs.extend(docs)

            if not all_retrieved_docs:
                return "No relevant information found in the indexed documents."

            relevant_docs = self.process_with_crag(question, all_retrieved_docs)
            if not relevant_docs:
                return "Could not find relevant information after CRAG processing."

            raw_answer = self.final_rag_chain.invoke({
                "question": question,
                "context": self.format_docs(relevant_docs)
            })
        
            if raw_answer.strip().startswith('{'):
                try:
                    answer_json = json.loads(raw_answer)
                    if isinstance(answer_json, dict):
                        if 'context' in answer_json:
                            return answer_json['context']
                        for key in ['answer', 'response', 'content', 'text']:
                            if key in answer_json:
                                return answer_json[key]
                except json.JSONDecodeError:
                    pass
                    
            return raw_answer.strip()
            

        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"


def main():
    rag_system = RAGS(tavily_api_key="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")
    sources = ["pasa.pdf"]
    try:
        documents = rag_system.load_content(sources)
        logger.info(f"Loaded {len(documents)} document chunks from the provided sources.")
    except Exception as e:
        logger.error(f"Error loading content: {e}")
        documents = []
    question = "explain about this paper?"
    try:
        answer = rag_system.query(question)
        print("Final Answer:")
        print(answer)
    except Exception as e:
        print(f"An error occurred while processing the query: {e}")


if __name__ == "__main__":
    main()
