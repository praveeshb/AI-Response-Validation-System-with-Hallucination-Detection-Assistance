import uuid
from typing import List

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ModuleNotFoundError:  # pragma: no cover - optional dependency in demo mode.
    RecursiveCharacterTextSplitter = None

from app.models import DocumentChunk, ChunkMetadata


class ChunkingStrategy:
    """Text chunking strategies for knowledge base."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = None
        if RecursiveCharacterTextSplitter is not None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

    def chunk_document(self, text: str, source: str, dataset: str = None, question_id: str = None) -> List[DocumentChunk]:
        """Split a document into chunks."""
        chunks = self._split_text(text)

        document_chunks = []
        for chunk_text in chunks:
            metadata = ChunkMetadata(
                source=source,
                chunk_id=str(uuid.uuid4()),
                question_id=question_id,
                dataset=dataset
            )
            document_chunks.append(DocumentChunk(content=chunk_text, metadata=metadata))

        return document_chunks

    def chunk_qa_pair(self, question: str, answer: str, source: str, dataset: str, question_id: str) -> List[DocumentChunk]:
        """Chunk a QA pair, keeping question and answer together."""
        combined_text = f"Question: {question}\nAnswer: {answer}"
        return self.chunk_document(combined_text, source, dataset, question_id)

    def _split_text(self, text: str) -> List[str]:
        if self.text_splitter is not None:
            return self.text_splitter.split_text(text)

        text = (text or "").strip()
        if not text:
            return []

        words = text.split()
        chunks = []
        current = []
        for word in words:
            current.append(word)
            if len(" ".join(current)) >= self.chunk_size:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks
