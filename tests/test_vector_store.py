"""
Smoke test to verify FAISSVectorStore operations, metadata mapping, and file serialization.

Run:
    .\\venv\\Scripts\\python tests\\test_vector_store.py
"""

import shutil
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.embeddings.biobert import BioBERTEmbedder
from app.loaders import PDFLoader, DocumentChunker
from app.retriever import FAISSVectorStore


def main() -> None:
    test_dir = Path("vector_store")
    test_dir.mkdir(exist_ok=True)

    # ── Initialize ───────────────────────────────────────────────
    print("Loading embedder...")
    embedder = BioBERTEmbedder(device="cpu")
    store = FAISSVectorStore(embedder)

    # ── Check PDF file existence ──────────────────────────────────
    data_dir = Path("data")
    pdfs = list(data_dir.glob("*.pdf"))

    if not pdfs:
        print("WARNING: No PDFs found in data/ folder. Please run tests/create_sample_pdf.py first.")
        return

    test_pdf = pdfs[0]
    print(f"Loading and chunking: {test_pdf.name}")
    loader = PDFLoader()
    doc = loader.load(test_pdf)
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=80)
    chunks = chunker.chunk_documents(doc.pages)

    # ── Add chunks to store ──────────────────────────────────────
    print(f"Adding {len(chunks)} chunks to FAISS...")
    store.add_chunks(chunks)

    # ── Save index ───────────────────────────────────────────────
    print("Saving FAISS index and metadata to vector_store/...")
    store.save(test_dir)

    # ── Verify files exist ────────────────────────────────────────
    idx_file = test_dir / "index.faiss"
    meta_file = test_dir / "metadata.json"

    if idx_file.exists() and meta_file.exists():
        print(f"SUCCESS: Files saved! Size of index: {idx_file.stat().st_size:,} bytes")
    else:
        print("FAILURE: Files not written correctly.")
        return

    # ── Load from disk into new store instance ────────────────────
    print("Loading files back into a clean FAISSVectorStore instance...")
    clean_store = FAISSVectorStore(embedder)
    clean_store.load(test_dir)

    # ── Validation ───────────────────────────────────────────────
    print(f"Loaded store vector count: {clean_store.index.ntotal}")
    print(f"Loaded store metadata items: {len(clean_store.chunks_map)}")

    if clean_store.index.ntotal == store.index.ntotal and len(clean_store.chunks_map) == len(store.chunks_map):
        print("SUCCESS: Vector storage loading matches original perfectly!")
        # Print a sample chunk from loaded memory to verify values
        sample_key = list(clean_store.chunks_map.keys())[0]
        sample_chunk = clean_store.chunks_map[sample_key]
        print(f"Sample chunk ID checked: {sample_chunk.chunk_id}")
    else:
        print("FAILURE: Count mismatch after loading.")


if __name__ == "__main__":
    main()
