"""
app.loaders — Document ingestion layer.

Public API::

    from app.loaders import PDFLoader, LoadedDocument, PageDocument
"""

from app.loaders.pdf_loader import PDFLoader, PDFLoaderError
from app.loaders.schemas import LoadedDocument, PageDocument

__all__ = ["PDFLoader", "PDFLoaderError", "LoadedDocument", "PageDocument"]
