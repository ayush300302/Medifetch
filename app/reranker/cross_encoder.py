"""
Cross-Encoder Reranker using sentence-transformers.
"""

import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from app.loaders.schemas import DocumentChunk
from app.reranker.base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Reranks document chunks against a query using a HuggingFace Cross-Encoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        """
        Initializes the CrossEncoder model.

        Args:
            model_name: HuggingFace model identifier.
            device: 'cpu' or 'cuda' for inference.
        """
        logger.info("Initializing Reranker model '%s' on %s...", model_name, device)
        self.model = CrossEncoder(model_name, device=device)
        logger.info("Reranker model loaded successfully.")

    def rerank(
        self, query: str, chunks: List[DocumentChunk], top_n: int = 3
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Scores pairs of [query, chunk.text] and returns the top_n highest scoring chunks.
        """
        if not chunks:
            return []

        # Prepare pairs for cross-encoder prediction
        pairs = [[query, chunk.text] for chunk in chunks]
        
        logger.debug("Reranking %d chunks against query: '%s'", len(chunks), query)
        
        # Predict scores
        # show_progress_bar=False to suppress unnecessary logs
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Zip chunks with scores and convert score elements to standard float
        scored_chunks = [
            (chunk, float(score)) for chunk, score in zip(chunks, scores)
        ]

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        logger.info("Reranked %d chunks. Top score: %.4f", len(chunks), scored_chunks[0][1] if scored_chunks else 0.0)

        # Return top N
        return scored_chunks[:top_n]
