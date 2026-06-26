"""
Abstract base class for chunk reranking.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from app.loaders.schemas import DocumentChunk


class BaseReranker(ABC):
    """
    Abstract interface for rerankers.
    Forces implementation of the `rerank` method.
    """

    @abstractmethod
    def rerank(
        self, query: str, chunks: List[DocumentChunk], top_n: int = 3
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Scores and ranks a list of chunks relative to the user query.

        Args:
            query: User search query.
            chunks: List of retrieved DocumentChunks.
            top_n: Number of high-scoring chunks to retain.

        Returns:
            A list of tuples (DocumentChunk, similarity_score) sorted descending by score.
        """
        pass
