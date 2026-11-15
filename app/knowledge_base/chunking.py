from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.models import DocumentChunk, ChunkMetadata
import uuid

class ChunkingStrategy:
    """Text chunking strategies for knowledge base"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(self, text: str, source: str, dataset: str = None, question_id: str = None) -> List[DocumentChunk]:
        """Split a document into chunks"""
        chunks = self.text_splitter.split_text(text)
        
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            metadata = ChunkMetadata(
                source=source,
                chunk_id=str(uuid.uuid4()),
                question_id=question_id,
                dataset=dataset
            )
            document_chunks.append(DocumentChunk(content=chunk_text, metadata=metadata))
        
        return document_chunks
    
    def chunk_qa_pair(self, question: str, answer: str, source: str, dataset: str, question_id: str) -> List[DocumentChunk]:
        """Chunk a QA pair, keeping question and answer together"""
        combined_text = f"Question: {question}\nAnswer: {answer}"
        return self.chunk_document(combined_text, source, dataset, question_id)
