"""
Smoke test to verify that the BioBERT embedder can load, encode sentences,
and correctly evaluate semantic proximity of medical concepts.

Run:
    .\\venv\\Scripts\\python tests\\test_embeddings.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from app.embeddings import BioBERTEmbedder


def cosine_similarity(v1, v2) -> float:
    # Since our embedder normalizes vectors to unit length, cosine similarity is just the dot product!
    return float(np.dot(v1, v2))


def main() -> None:
    print("Loading BioBERT Embedder (might take a minute to download on first run)...")
    embedder = BioBERTEmbedder(device="cpu")

    print("\nModel Loaded Successfully!")
    print(f"Embedding dimensions: {embedder.dimension}")

    # ── Test Queries ─────────────────────────────────────────────
    # We test with clinical synonyms vs unrelated terms to check domain intelligence
    text1 = "High blood pressure treatment"
    text2 = "Hypertension therapy options"
    text3 = "Acute asthma attack bronchospasm"

    print(f"\n1. Encoding: '{text1}'")
    emb1 = embedder.embed_text(text1)
    print(f"   Result vector length: {len(emb1)}")
    print(f"   First 5 values: {emb1[:5]}")

    print(f"\n2. Encoding: '{text2}'")
    emb2 = embedder.embed_text(text2)

    print(f"\n3. Encoding: '{text3}'")
    emb3 = embedder.embed_text(text3)

    # Calculate similarity
    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)

    print("\nCosine Similarity Analysis:")
    print(f"   - Similarity('{text1}', '{text2}') = {sim_1_2:.4f} (Should be high, synonym/related)")
    print(f"   - Similarity('{text1}', '{text3}') = {sim_1_3:.4f} (Should be lower, different medical topic)")

    if sim_1_2 > sim_1_3:
        print("\nSUCCESS: BioBERT successfully captured medical semantic relationships!")
    else:
        print("\nFAILURE: Check embedding normalization or model configuration.")


if __name__ == "__main__":
    main()
