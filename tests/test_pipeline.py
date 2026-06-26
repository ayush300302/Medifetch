"""
Smoke test to verify the full RAG pipeline integration.
Uses a Mock LLM client so the test runs deterministically without calling APIs.

Run:
    .\\venv\\Scripts\\python tests\\test_pipeline.py
"""

import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.embeddings.biobert import BioBERTEmbedder
from app.loaders.schemas import DocumentChunk
from app.retriever import FAISSVectorStore, BM25Retriever
from app.reranker import CrossEncoderReranker
from app.llm import BaseLLM
from app.rag import RAGPipeline


class MockLLM(BaseLLM):
    """
    Mock LLM client that returns a predictable response and prints the constructed prompt.
    """

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        print("\n--- MOCK LLM CALLED ---")
        print(f"System Instruction:\n{system_instruction}")
        print(f"Constructed Prompt:\n{prompt}")
        print("-----------------------")
        return "Based on the guidelines, ACE inhibitors are preferred first-line for diabetes patients."

    async def generate_stream(
        self, prompt: str, system_instruction: str | None = None
    ) -> AsyncGenerator[str, None]:
        yield "Mock response stream text."


async def main() -> None:
    print("Initializing components on CPU...")
    embedder = BioBERTEmbedder(device="cpu")
    store = FAISSVectorStore(embedder)
    bm25 = BM25Retriever()
    reranker = CrossEncoderReranker(device="cpu")
    mock_llm = MockLLM()

    # ── Populate Vector Store ─────────────────────────────────────
    print("\nAdding test guidelines to FAISS vector store...")
    chunks = [
        DocumentChunk(
            chunk_id="guideline1",
            source="hypertension_treatment.pdf",
            page_number=1,
            text="Nephroprotective Preferences: In patients with diabetes or CKD, ACE inhibitors or ARBs are preferred first-line agents.",
            word_count=18,
            char_count=130
        ),
        DocumentChunk(
            chunk_id="guideline2",
            source="hypertension_treatment.pdf",
            page_number=1,
            text="General hypertension: First line thiazide diuretics, CCBs, ACEi, ARBs.",
            word_count=9,
            char_count=70
        ),
        DocumentChunk(
            chunk_id="guideline3",
            source="asthma_treatment.pdf",
            page_number=2,
            text="Asthma step 3: Low-dose maintenance ICS + LABA is recommended.",
            word_count=10,
            char_count=62
        )
    ]
    store.add_chunks(chunks)
    bm25.add_chunks(chunks)

    # ── Initialize Pipeline ────────────────────────────────────────
    pipeline = RAGPipeline(
        vector_store=store,
        bm25_retriever=bm25,
        reranker=reranker,
        llm=mock_llm
    )

    # ── Test Query ─────────────────────────────────────────────────
    query = "What is preferred for a patient with diabetes and hypertension?"
    print(f"\nSending Query to RAG Pipeline: '{query}'")
    
    result = await pipeline.answer_query(query)

    print("\nPipeline Result Output:")
    print(f"Query : {result['query']}")
    print(f"Answer: {result['answer']}")
    print("Citations:")
    for cite in result["citations"]:
        print(f"  - Source: {cite['source']} (Page {cite['page_number']}) | Rerank Score: {cite['score']:.4f}")

    # Verify that the hypertension document was retrieved and cited
    citations_sources = [c["source"] for c in result["citations"]]
    if "hypertension_treatment.pdf" in citations_sources:
        print("\nSUCCESS: RAG pipeline successfully retrieved, reranked, and generated mock answer!")
    else:
        print("\nFAILURE: Hypertension guidelines were not cited in retrieval results.")


if __name__ == "__main__":
    asyncio.run(main())
