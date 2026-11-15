from typing import List, Dict, Any, Optional
from chromadb import Client, PersistentClient
from chromadb.config import Settings
from app.utils.config import config
from app.models import DocumentChunk
import os

class VectorStore:
    """Vector store for semantic search using ChromaDB"""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection_name = collection_name
        
        # Ensure persist directory exists
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = PersistentClient(path=config.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add document chunks with embeddings to the vector store"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        ids = [chunk.metadata.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source": chunk.metadata.source,
                "chunk_id": chunk.metadata.chunk_id,
                "question_id": chunk.metadata.question_id or "",
                "dataset": chunk.metadata.dataset or ""
            }
            for chunk in chunks
        ]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        
        return retrieved_docs
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }
