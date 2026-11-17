from typing import List

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - optional dependency in demo mode.
    SentenceTransformer = None

from app.utils.config import config
import numpy as np


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self.model = None
        if SentenceTransformer is not None:
            self.model = SentenceTransformer(self.model_name)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.model is None:
            return self._fallback_embedding(text)
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.model is None:
            return [self._fallback_embedding(text) for text in texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        if self.model is None:
            return 384
        return self.model.get_sentence_embedding_dimension()

    def _fallback_embedding(self, text: str) -> List[float]:
        """Create a deterministic lightweight vector fallback."""
        tokens = [token.lower() for token in text.replace("\n", " ").split() if token.strip()]
        vector = [0.0] * 384
        if not tokens:
            return vector

        for index, token in enumerate(tokens[:384]):
            value = sum(ord(char) for char in token) % 1000
            vector[index % 384] = float(value) / 1000.0

        return vector
