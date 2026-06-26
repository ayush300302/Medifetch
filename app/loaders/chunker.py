"""
Document Chunker — splits loaded PDF pages into overlapping chunks using a recursive character text splitter.

Design decisions:
  - Recursive split on: paragraph ("\\n\\n"), newline ("\\n"), space (" "), and empty string ("").
  - Tracks page numbers so that we can pinpoint exact source pages when returning citations.
  - Generates unique IDs for each chunk.
  - Implemented cleanly with type hints and SOLID rules.
"""

import logging
from typing import List

from app.loaders.schemas import DocumentChunk, PageDocument

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Splits PageDocument objects into smaller, overlapping segments suitable
    for embedding generation and vector search.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlapping characters between consecutive chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def chunk_page(self, page: PageDocument) -> List[DocumentChunk]:
        """
        Splits a single page document into chunks.

        Args:
            page: A PageDocument object.

        Returns:
            A list of DocumentChunk objects.
        """
        raw_text = page.text
        if not raw_text.strip():
            return []

        # Split the text recursively
        split_texts = self._split_text(raw_text, self.separators, self.chunk_size)

        chunks: List[DocumentChunk] = []
        for idx, text in enumerate(split_texts):
            text = text.strip()
            if not text:
                continue

            chunk_id = f"{page.source}_page_{page.page_number}_chunk_{idx}"
            words = len(text.split())

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=page.source,
                    page_number=page.page_number,
                    text=text,
                    word_count=words,
                    char_count=len(text),
                )
            )

        return chunks

    def chunk_documents(self, pages: List[PageDocument]) -> List[DocumentChunk]:
        """
        Chunk a collection of PageDocument pages.

        Args:
            pages: List of PageDocument instances.

        Returns:
            Flat list of DocumentChunk objects.
        """
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            all_chunks.extend(self.chunk_page(page))

        logger.info(
            "Created %d chunks from %d pages.", len(all_chunks), len(pages)
        )
        return all_chunks

    def _split_text(self, text: str, separators: List[str], max_length: int) -> List[str]:
        """
        Recursively splits text on separators until each piece is less than max_length,
        recombining pieces up to max_length with self.chunk_overlap.
        """
        final_chunks: List[str] = []

        # Find the appropriate separator
        separator = separators[-1]
        new_separators = []

        for idx, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            # Check if this separator splits the text
            if sep in text:
                separator = sep
                new_separators = separators[idx + 1 :]
                break

        # Split the text on the chosen separator
        if separator != "":
            splits = text.split(separator)
        else:
            # If no separator works, split character-by-character
            splits = list(text)

        good_splits: List[str] = []
        for s in splits:
            if len(s) < max_length:
                good_splits.append(s)
            else:
                # Recursively split the long segment with remaining separators
                if good_splits:
                    # Merge current buffer first
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                recursive_splits = self._split_text(s, new_separators, max_length)
                final_chunks.extend(recursive_splits)

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merges small splits back into chunks of up to self.chunk_size,
        incorporating self.chunk_overlap.
        """
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for s in splits:
            s_len = len(s)
            sep_len = len(separator) if current_doc else 0

            # Check if adding this split exceeds max size
            if total_len + sep_len + s_len > self.chunk_size:
                if total_len > 0:
                    docs.append(separator.join(current_doc))
                    # Retain some elements for overlap
                    # Backtrack to add overlap
                    overlap_doc: List[str] = []
                    overlap_len = 0
                    for prev in reversed(current_doc):
                        prev_len = len(prev)
                        p_sep_len = len(separator) if overlap_doc else 0
                        if overlap_len + p_sep_len + prev_len > self.chunk_overlap:
                            break
                        overlap_doc.insert(0, prev)
                        overlap_len += p_sep_len + prev_len

                    current_doc = overlap_doc
                    total_len = overlap_len

            current_doc.append(s)
            total_len += (len(separator) if total_len > 0 else 0) + s_len

        if current_doc:
            docs.append(separator.join(current_doc))

        return docs
