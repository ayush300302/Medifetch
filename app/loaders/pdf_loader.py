"""
PDF Loader — extracts structured text from PDF files using PyMuPDF.

Design decisions:
  - Stateless class: PDFLoader holds no mutable state between calls.
  - Each method is a single responsibility.
  - Page text is cleaned of excessive whitespace but otherwise unmodified
    (chunking is handled separately in Step 4).
  - All errors surface as typed exceptions, never silent failures.
"""

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from app.loaders.schemas import LoadedDocument, PageDocument

logger = logging.getLogger(__name__)


class PDFLoaderError(Exception):
    """Raised when a PDF cannot be loaded or parsed."""


class PDFLoader:
    """
    Loads one or more PDF files and returns structured page-level documents.

    Usage::

        loader = PDFLoader()
        doc = loader.load("data/clinical_guidelines.pdf")
        for page in doc.pages:
            print(page.page_number, page.text[:200])
    """

    # Minimum characters on a page for it to be considered non-empty
    MIN_PAGE_CHARS: int = 20

    def load(self, file_path: str | Path) -> LoadedDocument:
        """
        Load a single PDF file and extract text from every page.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            A :class:`LoadedDocument` containing all extracted pages.

        Raises:
            PDFLoaderError: If the file does not exist, is not a PDF,
                            or cannot be parsed by PyMuPDF.
        """
        path = self._validate_path(file_path)
        logger.info("Loading PDF: %s", path.name)

        try:
            pdf = fitz.open(str(path))
        except Exception as exc:
            raise PDFLoaderError(
                f"PyMuPDF could not open '{path.name}': {exc}"
            ) from exc

        pages: list[PageDocument] = []

        with pdf:
            total_pages = pdf.page_count
            logger.debug("'%s' has %d pages.", path.name, total_pages)

            for page_index in range(total_pages):
                page = pdf[page_index]
                raw_text = page.get_text("text")  # plain text extraction
                clean_text = self._clean_text(raw_text)

                pages.append(
                    PageDocument(
                        source=path.name,
                        page_number=page_index + 1,  # 1-indexed
                        text=clean_text,
                        char_count=len(clean_text),
                    )
                )

        non_empty = sum(1 for p in pages if p.char_count >= self.MIN_PAGE_CHARS)
        logger.info(
            "Loaded '%s': %d pages total, %d with content.",
            path.name,
            total_pages,
            non_empty,
        )

        return LoadedDocument(
            filename=path.name,
            total_pages=total_pages,
            pages=pages,
        )

    def load_many(self, file_paths: list[str | Path]) -> list[LoadedDocument]:
        """
        Load multiple PDF files sequentially.

        Args:
            file_paths: List of paths to PDF files.

        Returns:
            List of :class:`LoadedDocument`, one per file.
            Files that fail are logged and skipped (no silent data loss —
            a warning is always emitted).
        """
        results: list[LoadedDocument] = []

        for path in file_paths:
            try:
                results.append(self.load(path))
            except PDFLoaderError as exc:
                logger.warning("Skipping '%s': %s", path, exc)

        logger.info(
            "Loaded %d/%d PDFs successfully.", len(results), len(file_paths)
        )
        return results

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _validate_path(self, file_path: str | Path) -> Path:
        """
        Validate that the path exists and is a PDF file.

        Args:
            file_path: Raw path input from the caller.

        Returns:
            Resolved :class:`Path` object.

        Raises:
            PDFLoaderError: If the path is invalid or not a PDF.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise PDFLoaderError(f"File not found: '{path}'")

        if not path.is_file():
            raise PDFLoaderError(f"Path is not a file: '{path}'")

        if path.suffix.lower() != ".pdf":
            raise PDFLoaderError(
                f"Expected a .pdf file, got '{path.suffix}': '{path.name}'"
            )

        return path

    @staticmethod
    def _clean_text(raw: str) -> str:
        """
        Clean raw PyMuPDF text output.

        Transformations applied:
          1. Collapse runs of 3+ newlines into exactly 2 (preserve paragraphs).
          2. Strip leading/trailing whitespace from each line.
          3. Strip leading/trailing whitespace from the whole block.

        Args:
            raw: Raw text string from ``page.get_text()``.

        Returns:
            Cleaned text string.
        """
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", raw)

        # Strip trailing spaces from each line
        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines)

        return text.strip()
