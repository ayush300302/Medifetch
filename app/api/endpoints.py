"""
FastAPI Router — implements upload, chat, listing, and cleaning endpoints.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from pydantic import BaseModel, Field

from app.loaders import PDFLoader, DocumentChunker, PDFLoaderError
from app.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ── Chat Request/Response Schemas ─────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., examples=["What is first line therapy for diabetes?"])


class CitationSchema(BaseModel):
    chunk_id: str
    source: str
    page_number: int
    score: float
    text: str



class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationSchema]
    confidence_score: float = Field(default=-10.0, description="Top Cross-Encoder similarity score")
    confidence_level: str = Field(default="NONE", description="Confidence level name")


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int = Field(..., description="1 for positive, -1 for negative")
    reason: str | None = Field(default=None, description="Optional feedback details")


# ── Upload Response Schema ────────────────────────────────────────────

class UploadResponse(BaseModel):
    filename: str
    total_pages: int
    chunks_created: int
    status: str


# ── Routes ────────────────────────────────────────────────────────────

@router.post("/documents/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """
    Uploads a clinical guideline PDF. Extracts text, chunks, embeds, indexes in FAISS, and serializes to disk.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 1. Save uploaded file to local data folder
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    temp_file_path = data_dir / file.filename

    logger.info("Saving uploaded file to %s", temp_file_path)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        logger.error("Failed to write uploaded file to disk: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}")

    # 2. Extract text pages using PDFLoader
    loader = PDFLoader()
    try:
        loaded_doc = loader.load(temp_file_path)
    except PDFLoaderError as exc:
        logger.error("Failed to parse PDF document: %s", exc)
        if temp_file_path.exists():
            os.remove(temp_file_path)
        raise HTTPException(status_code=422, detail=f"Invalid PDF: {exc}")

    # 3. Chunk documents
    # Using default values from chunker setting or reasonable defaults
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(loaded_doc.pages)

    if not chunks:
        logger.warning("No text extracted from PDF, empty index.")
        return UploadResponse(
            filename=file.filename,
            total_pages=loaded_doc.total_pages,
            chunks_created=0,
            status="Processed empty document (no text chunks created)."
        )

    # 4. Add to global vector store and BM25 index and save
    vector_store = request.app.state.vector_store
    bm25_retriever = request.app.state.bm25_retriever
    try:
        vector_store.add_chunks(chunks)
        vector_store.save(settings.VECTOR_STORE_PATH)
        
        bm25_retriever.add_chunks(chunks)
        bm25_retriever.save(settings.VECTOR_STORE_PATH)
    except Exception as exc:
        logger.error("Failed to embed/index text chunks in hybrid stores: %s", exc)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")

    return UploadResponse(
        filename=file.filename,
        total_pages=loaded_doc.total_pages,
        chunks_created=len(chunks),
        status="Successfully uploaded and indexed in FAISS."
    )


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_query(request: Request, chat_req: ChatRequest) -> ChatResponse:
    """
    Processes a clinical query using RAG: retrieves guidelines, reranks, and prompts the LLM.
    """
    rag_pipeline = request.app.state.rag_pipeline
    
    if request.app.state.vector_store.is_empty():
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload a PDF guideline first."
        )

    query_text = chat_req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = await rag_pipeline.answer_query(query_text)
    except Exception as exc:
        logger.error("RAG Pipeline execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return ChatResponse(
        query=result["query"],
        answer=result["answer"],
        citations=result["citations"],
        confidence_score=result.get("confidence_score", -10.0),
        confidence_level=result.get("confidence_level", "NONE"),
    )


@router.get("/documents", response_model=List[str], tags=["Documents"])
async def list_documents(request: Request) -> List[str]:
    """
    Lists unique filenames of all guidelines currently indexed in the vector store.
    """
    vector_store = request.app.state.vector_store
    if vector_store.is_empty():
        return []
        
    # Gather distinct sources from metadata mapping
    unique_sources = set(chunk.source for chunk in vector_store.chunks_map.values())
    return sorted(list(unique_sources))


@router.delete("/documents", tags=["Documents"])
async def clear_documents(request: Request) -> Dict[str, str]:
    """
    Resets the vector store, clears the index files from disk, and purges the data directory.
    """
    vector_store = request.app.state.vector_store
    bm25_retriever = request.app.state.bm25_retriever
    
    # 1. Purge memory store
    import faiss
    vector_store.index = faiss.IndexFlatIP(vector_store.dimension)
    vector_store.chunks_map.clear()

    # Clear BM25 retriever memory
    bm25_retriever.bm25 = None
    bm25_retriever.chunks.clear()

    # 2. Delete persisted files on disk
    store_dir = Path(settings.VECTOR_STORE_PATH)
    if store_dir.exists():
        try:
            shutil.rmtree(store_dir)
        except Exception as exc:
            logger.error("Failed to delete vector store files on disk: %s", exc)

    # 3. Clean up PDFs in data folder
    data_dir = Path("data")
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.is_file() and item.suffix.lower() == ".pdf":
                try:
                    os.remove(item)
                except Exception as exc:
                    logger.error("Failed to delete PDF file %s: %s", item, exc)

    logger.info("Cleared all vector store contents and uploaded guidelines.")
    return {"status": "success", "message": "Cleared FAISS index and local PDFs."}


@router.post("/feedback", tags=["Feedback"])
async def save_feedback(feedback_req: FeedbackRequest) -> dict:
    """
    Saves explicit user rating feedback for a query and response to a local JSONL file for RLHF fine-tuning.
    """
    feedback_file = Path("data/feedback.jsonl")
    feedback_file.parent.mkdir(exist_ok=True)
    
    import time
    feedback_data = {
        "timestamp": time.time(),
        "query": feedback_req.query,
        "answer": feedback_req.answer,
        "rating": feedback_req.rating,
        "reason": feedback_req.reason
    }
    
    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("Failed to write feedback to log: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not save feedback: {exc}")
        
    return {"status": "success", "message": "Feedback saved successfully."}
