"""
FAISS Vector Store wrapper.
Handles FAISS index management, metadata storage mapping, and persistence.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np

from app.embeddings.base import BaseEmbedder
from app.loaders.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Integrates FAISS vector index with DocumentChunk metadata.
    Enables adding vectors, similarity searching, saving to disk, and loading.
    """

    def __init__(self, embedder: BaseEmbedder):
        """
        Initializes the vector store.

        Args:
            embedder: Any subclass of BaseEmbedder (e.g., BioBERTEmbedder).
        """
        self.embedder = embedder
        self.dimension = embedder.dimension

        # FAISS IndexFlatIP computes inner product (equivalent to Cosine Similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)

        # Mapping of FAISS internal index integer ID to DocumentChunk metadata
        self.chunks_map: Dict[int, DocumentChunk] = {}

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Extracts texts, generates embeddings, adds them to FAISS index,
        and saves chunk metadata mapping.

        Args:
            chunks: List of DocumentChunk objects.
        """
        if not chunks:
            logger.warning("No chunks provided to add.")
            return

        texts = [chunk.text for chunk in chunks]
        logger.info("Generating embeddings for %d chunks...", len(chunks))

        embeddings_list = self.embedder.embed_documents(texts)
        embeddings = np.array(embeddings_list, dtype=np.float32)

        # FAISS requires floating point vectors
        start_idx = self.index.ntotal
        self.index.add(embeddings)

        # Record metadata for each vector using its integer index position
        for i, chunk in enumerate(chunks):
            faiss_id = start_idx + i
            self.chunks_map[faiss_id] = chunk

        logger.info(
            "Successfully indexed %d chunks. Total index size: %d",
            len(chunks),
            self.index.ntotal,
        )

    def save(self, directory: str | Path) -> None:
        """
        Persists both the FAISS index (binary) and metadata map (JSON) to disk.

        Args:
            directory: Path to directory where files will be written.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / "index.faiss"
        meta_file = dir_path / "metadata.json"

        # Save FAISS binary index
        faiss.write_index(self.index, str(index_file))

        # Save metadata mapping as serialized dict of Pydantic models
        meta_data = {
            str(k): v.model_dump() for k, v in self.chunks_map.items()
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        logger.info(
            "Saved vector store to '%s' (%d items).",
            dir_path.resolve(),
            self.index.ntotal,
        )

    def load(self, directory: str | Path) -> None:
        """
        Loads FAISS index and metadata map from disk.

        Args:
            directory: Path to directory containing index.faiss and metadata.json.
        """
        dir_path = Path(directory)
        index_file = dir_path / "index.faiss"
        meta_file = dir_path / "metadata.json"

        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(
                f"FAISS index or metadata file not found under '{dir_path}'"
            )

        # Load FAISS index
        self.index = faiss.read_index(str(index_file))

        # Load metadata
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # Reconstruct mapping of int -> DocumentChunk
        self.chunks_map = {
            int(k): DocumentChunk(**v) for k, v in meta_data.items()
        }

        logger.info(
            "Loaded vector store from '%s' (%d items).",
            dir_path.resolve(),
            self.index.ntotal,
        )

    def is_empty(self) -> bool:
        """Checks if the FAISS index is empty."""
        return self.index.ntotal == 0
