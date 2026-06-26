"""
Healthcare RAG Chatbot — FastAPI Entry Point

This module initialises the FastAPI application, registers all API routers,
configures CORS, and wires up startup/shutdown lifecycle events.

Built step-by-step following SOLID principles.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Healthcare RAG Assistant",
    description="Answers medical questions ONLY from uploaded healthcare documents.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Liveness probe — returns 200 if the service is running."""
    return {"status": "ok", "service": "healthcare-rag"}
