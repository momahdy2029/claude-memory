"""Embedding service using Ollama with nomic-embed-text model."""
import os
from typing import List
import ollama
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


class EmbeddingService:
    """Service for generating embeddings using Ollama."""

    def __init__(self):
        self.model = EMBEDDING_MODEL
        self.client = ollama.Client(host=OLLAMA_HOST)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def get_dimension(self) -> int:
        """Return the embedding dimension (nomic-embed-text uses 768)."""
        return 768
