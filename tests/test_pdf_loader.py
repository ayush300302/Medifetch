"""
Quick smoke test for the PDF loader.

Drop any PDF into the data/ folder and run:
    .\\venv\\Scripts\\python tests\\test_pdf_loader.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.loaders import PDFLoader, PDFLoaderError


def main() -> None:
    loader = PDFLoader()

    # ── Find a test PDF ─────────────────────────────────────────
    data_dir = Path("data")
    pdfs = list(data_dir.glob("*.pdf"))

    if not pdfs:
        print("⚠️  No PDFs found in data/ folder.")
        print("   Drop any PDF into data/ and re-run this script.")
        return

    test_pdf = pdfs[0]
    print(f"📄 Testing with: {test_pdf.name}\n")

    # ── Load ─────────────────────────────────────────────────────
    try:
        doc = loader.load(test_pdf)
    except PDFLoaderError as e:
        print(f"❌ Load failed: {e}")
        return

    # ── Report ───────────────────────────────────────────────────
    print(f"✅ Loaded successfully!")
    print(f"   Filename    : {doc.filename}")
    print(f"   Total pages : {doc.total_pages}")
    print(f"   Total chars : {doc.total_chars:,}")
    print(f"   Non-empty   : {len(doc.non_empty_pages)} pages\n")

    # ── Show first 3 pages ───────────────────────────────────────
    for page in doc.pages[:3]:
        print(f"── Page {page.page_number} ({page.char_count} chars) ──")
        print(page.text[:300])
        print("...")
        print()

    # ── Error handling test ──────────────────────────────────────
    print("🔍 Testing error handling...")
    try:
        loader.load("nonexistent.pdf")
    except PDFLoaderError as e:
        print(f"✅ Error caught correctly: {e}")


if __name__ == "__main__":
    main()
