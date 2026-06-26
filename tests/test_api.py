"""
FastAPI Endpoints Smoke Test using FastAPI TestClient.
Mocks the app state models and pipelines to run instantly and deterministically.

Run:
    .\\venv\\Scripts\\python tests\\test_api.py
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, AsyncGenerator

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app
from app.loaders.schemas import DocumentChunk
from app.llm import BaseLLM


# ── Mock Classes ──────────────────────────────────────────────────────

class MockEmbedder:
    def __init__(self):
        self.dimension = 384

    def embed_text(self, text: str) -> List[float]:
        return [0.1] * self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * self.dimension for _ in texts]


class MockVectorStore:
    def __init__(self):
        self.dimension = 384
        self.chunks_map = {}
        self.ntotal = 0

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        for idx, chunk in enumerate(chunks):
            self.chunks_map[self.ntotal + idx] = chunk
        self.ntotal += len(chunks)

    def save(self, directory: Any) -> None:
        pass

    def load(self, directory: Any) -> None:
        pass

    def is_empty(self) -> bool:
        return self.ntotal == 0

    def similarity_search(self, query: str, k: int = 10) -> List[tuple]:
        if self.is_empty():
            return []
        # Return first chunk as mock result
        first_key = list(self.chunks_map.keys())[0]
        return [(self.chunks_map[first_key], 0.95)]


class MockReranker:
    def rerank(self, query: str, chunks: List[DocumentChunk], top_n: int = 3) -> List[tuple]:
        return [(chunk, 0.95) for chunk in chunks[:top_n]]


class MockLLM(BaseLLM):
    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        return "Mock response: ACE inhibitors are preferred first-line therapy according to the guideline."

    async def generate_stream(
        self, prompt: str, system_instruction: str | None = None
    ) -> AsyncGenerator[str, None]:
        yield "Mock response stream."


class MockBM25Retriever:
    def __init__(self):
        self.chunks = []
        self.bm25 = None

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        self.chunks.extend(chunks)

    def save(self, directory: Any) -> None:
        pass

    def load(self, directory: Any) -> None:
        pass

    def search(self, query: str, k: int = 15) -> List[DocumentChunk]:
        return self.chunks[:k]


class MockRAGPipeline:
    def __init__(self, store):
        self.vector_store = store

    async def answer_query(self, query: str) -> Dict[str, Any]:
        if self.vector_store.is_empty():
            return {
                "query": query,
                "answer": "I cannot find the answer in the provided documents.",
                "citations": []
            }
        
        first_key = list(self.vector_store.chunks_map.keys())[0]
        chunk = self.vector_store.chunks_map[first_key]
        return {
            "query": query,
            "answer": "Mock response: ACE inhibitors are preferred first-line therapy according to the guideline.",
            "citations": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "score": 0.95,
                    "text": "Mock chunk text content"
                }
            ]

        }


# ── Run test script ───────────────────────────────────────────────────

def main() -> None:
    print("Setting up API TestClient and mocking App State...")
    
    # Instantiate mock singletons
    mock_store = MockVectorStore()
    mock_bm25 = MockBM25Retriever()
    mock_pipeline = MockRAGPipeline(mock_store)

    # Initialize app.state properties directly to bypass heavy startup loading
    app.state.embedder = MockEmbedder()
    app.state.vector_store = mock_store
    app.state.bm25_retriever = mock_bm25
    app.state.reranker = MockReranker()
    app.state.llm = MockLLM()
    app.state.rag_pipeline = mock_pipeline

    client = TestClient(app)

    # 1. Health Probe Test
    print("\n1. Testing GET /health...")
    resp = client.get("/health")
    assert resp.status_code == 200
    print(f"   Response: {resp.json()}")

    # 2. Get Documents List Test (Should be empty initially)
    print("\n2. Testing GET /api/documents (Before upload)...")
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    assert resp.json() == []
    print("   Response: empty list as expected.")

    # 3. Chat Endpoint (Should fail initially because no documents are uploaded)
    print("\n3. Testing POST /api/chat (Before upload - should return 400 error)...")
    resp = client.post("/api/chat", json={"query": "test query"})
    assert resp.status_code == 400
    print(f"   Response Status: {resp.status_code} | Msg: {resp.json()['detail']}")

    # 4. Upload PDF Document Test
    print("\n4. Testing POST /api/documents/upload (Mock PDF Upload)...")
    # Create a small dummy PDF file content
    dummy_pdf_content = b"%PDF-1.4 mock content..."
    # We use upload_document which calls fitz.open, so let's mock fitz.open inside loaders if we want it to parse
    # Wait, instead of mock, we can upload the actual sample guidelines PDF!
    # Let's check if the sample PDF exists, else create it using the creation script
    sample_pdf_path = Path("data/sample_medical_guidelines.pdf")
    if not sample_pdf_path.exists():
        print("   -> Creating sample guidelines PDF first...")
        from tests.create_sample_pdf import create_synthetic_pdf
        Path("data").mkdir(exist_ok=True)
        create_synthetic_pdf(str(sample_pdf_path))
        
    with open(sample_pdf_path, "rb") as f:
        resp = client.post(
            "/api/documents/upload",
            files={"file": (sample_pdf_path.name, f, "application/pdf")}
        )
    assert resp.status_code == 200
    upload_data = resp.json()
    print(f"   Response: {upload_data}")
    assert upload_data["chunks_created"] > 0

    # 5. List Documents Test (Should now list our sample PDF)
    print("\n5. Testing GET /api/documents (After upload)...")
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    print(f"   Response: {resp.json()}")
    assert sample_pdf_path.name in resp.json()

    # 6. Chat Query Test
    print("\n6. Testing POST /api/chat (After upload)...")
    resp = client.post("/api/chat", json={"query": "What treatment is preferred for CKD?"})
    assert resp.status_code == 200
    chat_data = resp.json()
    print(f"   Query   : {chat_data['query']}")
    print(f"   Answer  : {chat_data['answer']}")
    print(f"   Citations: {chat_data['citations']}")
    assert len(chat_data["citations"]) > 0

    # 7. Clear Documents Test
    print("\n7. Testing DELETE /api/documents...")
    resp = client.delete("/api/documents")
    assert resp.status_code == 200
    print(f"   Response: {resp.json()}")

    # 8. Verify List is Empty Again
    print("\n8. Testing GET /api/documents (After clear)...")
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    assert resp.json() == []
    print("   Response: empty list as expected. Clean up verified.")

    print("\nSUCCESS: All FastAPI endpoint tests completed successfully!")


if __name__ == "__main__":
    main()
