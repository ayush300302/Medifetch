"""
BioBERT Embedder — generates clinical embeddings using HuggingFace sentence-transformers.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from app.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class BioBERTEmbedder(BaseEmbedder):
    """
    Generates semantic vectors using a biomedical-specific sentence transformer model.
    """

    def __init__(
        self,
        model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        device: str = "cpu",
    ):
        """
        Initializes the model.

        Args:
            model_name: HuggingFace model identifier.
            device: 'cpu' or 'cuda' (GPU) for inference.
        """
        logger.info("Initializing BioBERT model '%s' on %s...", model_name, device)
        # Suppress verbose HuggingFace download messages
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info("BioBERT model loaded. Vector dimension: %d", self._dimension)

    def embed_text(self, text: str) -> List[float]:
        """
        Generates embedding for a single text query.
        """
        if not text.strip():
            return [0.0] * self._dimension

        # normalize_embeddings=True ensures embeddings are unit length (L2 norm)
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of documents/chunks.
        """
        cleaned_texts = [t.strip() if t.strip() else "" for t in texts]
        if not cleaned_texts:
            return []

        embeddings = self.model.encode(
            cleaned_texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
