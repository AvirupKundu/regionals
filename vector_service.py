import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import json
from datetime import datetime
import shutil

# Load environment variables
load_dotenv()

# Setup tiktoken cache
tiktoken_cache_dir = os.path.join(os.path.dirname(__file__), "..", "tiktoken_cache")
os.makedirs(tiktoken_cache_dir, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# HTTP client with SSL verification disabled
client = httpx.Client(verify=False)

class VectorEmbeddingService:
    """Service for creating and managing vector embeddings using ChromaDB"""
    
    def __init__(self):
        self.base_url = os.getenv("BASE_URL")
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Vector store directory
        self.vector_store_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        os.makedirs(self.vector_store_dir, exist_ok=True)
    
    def get_embedding_model(self, model_name: str = None):
        """Get embedding model instance"""
        if not model_name:
            model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        return OpenAIEmbeddings(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model_name,
            http_client=client
        )
    
    def create_embeddings(self, text_content: str, metadata: Dict = None, chunk_size: int = 3000, 
                         chunk_overlap: int = 200, embedding_model: str = None) -> Dict[str, Any]:
        """
        Create vector embeddings from text content using ChromaDB
        
        Args:
            text_content: Text to embed
            metadata: Optional metadata about the source
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Model name for embeddings
            
        Returns:
            Dictionary with embedding creation status and details
        """
        try:
            # Get embedding model
            embed_model = self.get_embedding_model(embedding_model)
            model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            
            # Step 1: Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_text(text_content)
            
            # Step 2: Create embeddings and store in ChromaDB
            vectordb = Chroma.from_texts(
                chunks,
                embed_model,
                persist_directory=self.vector_store_dir,
                metadatas=[metadata or {} for _ in chunks]
            )
            vectordb.persist()
            
            # Step 3: Get stats
            collection = vectordb._collection
            total_embeddings = collection.count()
            
            return {
                "status": "success",
                "message": "Vector embeddings created successfully",
                "details": {
                    "num_chunks": len(chunks),
                    "total_embeddings": total_embeddings,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embedding_model": model_name,
                    "vector_store_path": self.vector_store_dir,
                    "created_at": datetime.now().isoformat(),
                    "sample_chunks": chunks[:2] if len(chunks) > 0 else []
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating embeddings: {str(e)}"
            }
    
    def query_embeddings(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Query the vector database with RAG
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Dictionary with search results and generated answer
        """
        try:
            if not os.path.exists(self.vector_store_dir) or not os.listdir(self.vector_store_dir):
                return {
                    "status": "error",
                    "message": "No knowledge base found. Please upload documents and create embeddings first."
                }
            
            # Load ChromaDB
            embed_model = self.get_embedding_model()
            vectordb = Chroma(
                persist_directory=self.vector_store_dir,
                embedding_function=embed_model
            )
            
            # Get relevant documents
            results = vectordb.similarity_search(query, k=k)
            
            if not results:
                return {
                    "status": "success",
                    "query": query,
                    "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                    "sources": []
                }
            
            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in results])
            
            # Generate answer using LLM
            llm = ChatOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                model=os.getenv("LLM_MODEL", "gpt-4.1-nano"),
                http_client=client
            )
            
            prompt = f"""Based on the following context from the knowledge base, answer the question. 
If the answer is not in the context, say "I don't have enough information in the knowledge base to answer this question."

Context:
{context}

Question: {query}

Answer:"""
            
            response = llm.invoke(prompt)
            answer = response.content
            
            return {
                "status": "success",
                "query": query,
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in results
                ]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error querying knowledge base: {str(e)}"
            }
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        try:
            if not os.path.exists(self.vector_store_dir) or not os.listdir(self.vector_store_dir):
                return {
                    "status": "success",
                    "total_embeddings": 0,
                    "message": "No embeddings created yet"
                }
            
            embed_model = self.get_embedding_model()
            vectordb = Chroma(
                persist_directory=self.vector_store_dir,
                embedding_function=embed_model
            )
            
            collection = vectordb._collection
            total_embeddings = collection.count()
            
            return {
                "status": "success",
                "total_embeddings": total_embeddings,
                "vector_store_path": self.vector_store_dir,
                "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting stats: {str(e)}"
            }
    
    def clear_embeddings(self) -> Dict[str, Any]:
        """Clear all embeddings from ChromaDB"""
        try:
            if os.path.exists(self.vector_store_dir):
                shutil.rmtree(self.vector_store_dir)
                os.makedirs(self.vector_store_dir)
            
            return {
                "status": "success",
                "message": "All embeddings cleared successfully",
                "total_embeddings": 0
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error clearing embeddings: {str(e)}"
            }
