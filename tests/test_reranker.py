"""
Smoke test to verify that the CrossEncoderReranker runs and scores medical chunks.

Run:
    .\\venv\\Scripts\\python tests\\test_reranker.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.loaders.schemas import DocumentChunk
from app.reranker import CrossEncoderReranker


def main() -> None:
    print("Loading Reranker (ms-marco-MiniLM-L-6-v2) on CPU...")
    # ms-marco-MiniLM-L-6-v2 is standard and lightweight
    reranker = CrossEncoderReranker(device="cpu")
    print("Reranker loaded successfully!")

    # ── Mock Chunks ──────────────────────────────────────────────
    query = "treatment guidelines for high blood pressure"
    chunks = [
        DocumentChunk(
            chunk_id="doc1_chunk0",
            source="hypertension.pdf",
            page_number=1,
            text="Nephroprotective Preferences: In patients with diabetes or chronic kidney disease (CKD), ACE inhibitors or ARBs are preferred first-line hypertension agents.",
            word_count=21,
            char_count=154
        ),
        DocumentChunk(
            chunk_id="doc1_chunk1",
            source="hypertension.pdf",
            page_number=2,
            text="Sodium Restriction: Limit sodium intake to <2,300 mg/day (ideal goal <1,500 mg/day) for Stage 1 or 2 hypertension patients.",
            word_count=19,
            char_count=124
        ),
        DocumentChunk(
            chunk_id="doc2_chunk0",
            source="asthma.pdf",
            page_number=1,
            text="Asthma is characterized by variable expiratory airflow limitation. Diagnosis is confirmed using spirometry showing reversible bronchoconstriction.",
            word_count=17,
            char_count=144
        )
    ]

    print(f"\nRerank Query: '{query}'")
    print("Chunks before reranking:")
    for idx, c in enumerate(chunks):
        print(f"  [{idx}] Source: {c.source} | Text: {c.text[:60]}...")

    top_n = 2
    results = reranker.rerank(query, chunks, top_n=top_n)

    print(f"\nTop {top_n} Reranked Results:")
    for idx, (chunk, score) in enumerate(results):
        print(f"  {idx + 1}. [Score: {score:.4f}] Source: {chunk.source} (Page {chunk.page_number})")
        print(f"     Text: {chunk.text}")

    # Assertions / Checks
    if len(results) == top_n:
        # Hypertension text should score higher than Asthma text for a hypertension query
        if results[0][0].source == "hypertension.pdf":
            print("\nSUCCESS: Reranker correctly scored domain-relevant context higher!")
        else:
            print("\nWARNING: Unrelated document ranked highest. Check cross-encoder model behavior.")
    else:
        print("\nFAILURE: Returned unexpected number of reranked results.")


if __name__ == "__main__":
    main()
