"""
Pydantic schemas for the document loading and chunking layer.

These models represent the structured output of the PDF loader and chunker,
and are passed downstream to the embedder, vector store, and LLM generator.
"""

from pydantic import BaseModel, Field


class PageDocument(BaseModel):
    """Represents a single page extracted from a PDF document."""

    source: str = Field(
        ...,
        description="Filename of the source PDF.",
        examples=["clinical_guidelines_2024.pdf"],
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number within the source document.",
    )
    text: str = Field(
        ...,
        description="Raw extracted text content of the page.",
    )
    char_count: int = Field(
        ...,
        ge=0,
        description="Number of characters in the extracted text.",
    )

    class Config:
        frozen = True  # Immutable after creation — safe to cache


class LoadedDocument(BaseModel):
    """Represents a fully loaded PDF with all its pages."""

    filename: str = Field(..., description="Original PDF filename.")
    total_pages: int = Field(..., ge=1, description="Total number of pages in the PDF.")
    pages: list[PageDocument] = Field(
        ..., description="List of all extracted pages in order."
    )

    @property
    def total_chars(self) -> int:
        """Total character count across all pages."""
        return sum(p.char_count for p in self.pages)

    @property
    def non_empty_pages(self) -> list[PageDocument]:
        """Pages that contain actual text content."""
        return [p for p in self.pages if p.char_count > 0]


class DocumentChunk(BaseModel):
    """Represents a small segment of text extracted from a page or pages."""

    chunk_id: str = Field(
        ...,
        description="Unique identifier for the chunk, usually source_filename_page_index.",
        examples=["guidelines.pdf_page_2_chunk_0"],
    )
    source: str = Field(
        ...,
        description="Source PDF filename.",
        examples=["clinical_guidelines_2024.pdf"],
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number from which this chunk was extracted.",
    )
    text: str = Field(
        ...,
        description="The chunk's textual content.",
    )
    word_count: int = Field(
        ...,
        ge=0,
        description="Number of words in the chunk's text.",
    )
    char_count: int = Field(
        ...,
        ge=0,
        description="Number of characters in the chunk's text.",
    )

    class Config:
        frozen = True
