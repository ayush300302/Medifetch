"""
app.embeddings — BioBERT and Abstract base layers for embedding generation.
"""

from app.embeddings.base import BaseEmbedder
from app.embeddings.biobert import BioBERTEmbedder

__all__ = ["BaseEmbedder", "BioBERTEmbedder"]
