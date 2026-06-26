"""
Base interface for embedding generators.
Enforces SOLID design by decoupling retrieval from specific model implementations.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Abstract base class for all embedding generators."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single text string.

        Args:
            text: Input string.

        Returns:
            A list of floats representing the embedding vector.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of text strings.

        Args:
            texts: List of input strings.

        Returns:
            A list of embedding vectors.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The size/dimension of the generated embedding vectors."""
        pass
