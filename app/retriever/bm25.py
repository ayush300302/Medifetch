"""
BM25 sparse search retriever wrapper using rank-bm25.
Handles indexing, search scoring, and serialization to disk.
"""

import logging
import pickle
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from app.loaders.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Integrates BM25 sparse retrieval with DocumentChunk metadata.
    Enables keyword search, local serialization via pickle, and merging.
    """

    def __init__(self):
        self.bm25 = None
        self.chunks: List[DocumentChunk] = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string by lowercasing, removing punctuation, and splitting.
        """
        # Convert to lowercase and strip non-word characters except hyphens
        clean_text = re.sub(r"[^\w\s-]", "", text.lower())
        return clean_text.split()

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Appends chunks to the corpus and builds/updates the BM25 model index.
        """
        if not chunks:
            logger.warning("No chunks provided to add to BM25 index.")
            return

        self.chunks.extend(chunks)
        logger.info("Tokenizing and indexing %d chunks for BM25...", len(chunks))

        # Build the tokenized corpus for rank-bm25
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info(
            "BM25 index updated successfully. Total indexed chunks: %d", 
            len(self.chunks)
        )

    def search(self, query: str, k: int = 15) -> List[DocumentChunk]:
        """
        Searches the BM25 index and returns up to k candidate chunks.
        Filters out candidates with a score of 0.0 (no keyword overlap).
        """
        if not self.bm25 or not self.chunks:
            logger.warning("BM25 search requested but index is empty.")
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        # Get keyword match scores
        scores = self.bm25.get_scores(tokenized_query)

        # Pair chunks with their scores
        scored_chunks = list(zip(self.chunks, scores))
        
        # Sort descending by BM25 relevance score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Filter out chunks with score <= 0 (i.e. zero term overlap)
        results = [chunk for chunk, score in scored_chunks[:k] if score > 0.0]
        
        logger.debug(
            "BM25 retrieved %d chunks (score > 0) out of top %d candidates.", 
            len(results), k
        )
        return results

    def save(self, directory: str | Path) -> None:
        """
        Persists both the list of chunks and the BM25 model to disk via pickle.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / "bm25.pkl"

        logger.info("Saving BM25 index to '%s'...", file_path)
        with open(file_path, "wb") as f:
            pickle.dump((self.chunks, self.bm25), f)
        
        logger.info("Saved BM25 index successfully.")

    def load(self, directory: str | Path) -> None:
        """
        Loads the persisted chunks and BM25 model from disk.
        """
        dir_path = Path(directory)
        file_path = dir_path / "bm25.pkl"

        if not file_path.exists():
            raise FileNotFoundError(
                f"BM25 serialized index not found under '{dir_path}'"
            )

        logger.info("Loading BM25 index from '%s'...", file_path)
        with open(file_path, "rb") as f:
            self.chunks, self.bm25 = pickle.load(f)

        logger.info("Loaded BM25 index successfully (%d items).", len(self.chunks))

    def is_empty(self) -> bool:
        """Checks if the BM25 index is empty."""
        return len(self.chunks) == 0
