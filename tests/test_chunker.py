"""
Quick smoke test for the Document Chunker.

Drop any PDF into the data/ folder, then run:
    .\\venv\\Scripts\\python tests\\test_chunker.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.loaders import PDFLoader, DocumentChunker


def main() -> None:
    loader = PDFLoader()
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=80)

    # ── Find a test PDF ─────────────────────────────────────────
    data_dir = Path("data")
    pdfs = list(data_dir.glob("*.pdf"))

    if not pdfs:
        print("WARNING: No PDFs found in data/ folder.")
        print("   Drop any PDF into data/ and re-run this script.")
        return

    test_pdf = pdfs[0]
    print(f"Testing chunker with: {test_pdf.name}\n")

    # ── Load ─────────────────────────────────────────────────────
    doc = loader.load(test_pdf)

    # ── Chunk ────────────────────────────────────────────────────
    chunks = chunker.chunk_documents(doc.pages)

    # ── Report ───────────────────────────────────────────────────
    print(f"Chunking completed!")
    print(f"   Total Pages  : {len(doc.pages)}")
    print(f"   Total Chunks : {len(chunks)}")
    print(f"   Avg Chunks/Pg: {len(chunks) / max(1, len(doc.pages)):.1f}\n")

    # ── Show sample chunks ───────────────────────────────────────
    sample_limit = min(3, len(chunks))
    for i in range(sample_limit):
        chunk = chunks[i]
        print(f"--- Chunk {i+1} ID: {chunk.chunk_id} ---")
        print(f"   Source   : {chunk.source}")
        print(f"   Page     : {chunk.page_number}")
        print(f"   Stats    : {chunk.char_count} chars, {chunk.word_count} words")
        print(f"   Content  :\n[ {chunk.text} ]")
        print()

    # ── Verify overlap ───────────────────────────────────────────
    if len(chunks) > 1:
        print("Testing if chunk overlap is functional...")
        c1, c2 = chunks[0], chunks[1]
        print(f"Chunk 1 ending: ... {c1.text[-50:]}")
        print(f"Chunk 2 starting: {c2.text[:50]} ...")


if __name__ == "__main__":
    main()
