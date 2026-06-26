"""
app.loaders — Document ingestion and chunking layer.

Public API::

    from app.loaders import PDFLoader, LoadedDocument, PageDocument, DocumentChunker, DocumentChunk
"""

from app.loaders.pdf_loader import PDFLoader, PDFLoaderError
from app.loaders.schemas import LoadedDocument, PageDocument, DocumentChunk
from app.loaders.chunker import DocumentChunker

__all__ = [
    "PDFLoader",
    "PDFLoaderError",
    "LoadedDocument",
    "PageDocument",
    "DocumentChunk",
    "DocumentChunker",
]
