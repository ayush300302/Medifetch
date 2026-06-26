import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI

from app.utils.config import settings
from app.embeddings.biobert import BioBERTEmbedder
from app.retriever import FAISSVectorStore, BM25Retriever
from app.reranker.cross_encoder import CrossEncoderReranker
from app.llm import OpenAILLM, GeminiLLM, MockLLM, OllamaLLM
from app.rag.pipeline import RAGPipeline
from app.api.endpoints import router

# Setup logging configuration
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous lifespan context manager for FastAPI.
    Handles startup loading of models/indexes and clean shutdown.
    """
    logger.info("Initializing Healthcare RAG application singletons...")

    # 1. Initialize clinical embedder (BioBERT)
    logger.info("Loading BioBERT embedder model...")
    embedder = BioBERTEmbedder(device="cpu")

    # 2. Setup Vector Store & BM25 Sparse Store
    logger.info("Loading FAISS vector database store...")
    vector_store = FAISSVectorStore(embedder)
    
    logger.info("Loading BM25 sparse retriever store...")
    bm25_retriever = BM25Retriever()
    
    store_dir = Path(settings.VECTOR_STORE_PATH)
    # Load FAISS index
    if (store_dir / "index.faiss").exists() and (store_dir / "metadata.json").exists():
        logger.info("Found existing FAISS index on disk under '%s'. Loading...", store_dir)
        try:
            vector_store.load(store_dir)
        except Exception as exc:
            logger.error("Failed to load saved FAISS index: %s. Defaulting to empty store.", exc)
    else:
        logger.info("No saved FAISS index found. Starting with empty store.")

    # Load BM25 index
    if (store_dir / "bm25.pkl").exists():
        logger.info("Found existing BM25 index on disk under '%s'. Loading...", store_dir)
        try:
            bm25_retriever.load(store_dir)
        except Exception as exc:
            logger.error("Failed to load saved BM25 index: %s. Defaulting to empty store.", exc)
    else:
        logger.info("No saved BM25 index found. Starting with empty store.")

    # 3. Setup Reranker (CrossEncoder)
    logger.info("Loading Cross-Encoder Reranker model...")
    reranker = CrossEncoderReranker(model_name=settings.RERANKER_MODEL, device="cpu")

    # 4. Setup LLM provider
    provider = settings.LLM_PROVIDER.lower().strip()
    if provider == "ollama":
        pass
    elif not settings.OPENAI_API_KEY and not settings.GEMINI_API_KEY:
        logger.warning("No API keys found for OpenAI or Gemini. Fallback to 'mock' offline mode.")
        provider = "mock"

    logger.info("Configuring LLM provider client: '%s'...", provider)
    if provider == "openai":
        llm = OpenAILLM(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.OPENAI_MODEL,
            base_url=settings.OPENAI_BASE_URL if settings.OPENAI_BASE_URL else None
        )
    elif provider == "gemini":
        llm = GeminiLLM(api_key=settings.GEMINI_API_KEY)
    elif provider == "ollama":
        llm = OllamaLLM(host=settings.OLLAMA_HOST, model_name=settings.OLLAMA_MODEL)
    elif provider == "mock":
        llm = MockLLM()
    else:
        logger.warning("Unknown provider '%s'. Defaulting to offline mock client.", provider)
        llm = MockLLM()

    # 5. Wire up the full RAG pipeline
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        llm=llm,
    )

    # Store references as application state singletons
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.bm25_retriever = bm25_retriever
    app.state.reranker = reranker
    app.state.llm = llm
    app.state.rag_pipeline = rag_pipeline

    logger.info("Startup complete. All models and singletons loaded successfully.")
    yield
    
    logger.info("Shutting down. Purging in-memory singletons.")


from fastapi.staticfiles import StaticFiles

# Create FastAPI application
app = FastAPI(
    title="Healthcare RAG Assistant",
    description="Answers clinical and medical queries based ONLY on uploaded documents using BioBERT & Cross-Encoder.",
    version="0.1.0",
    lifespan=lifespan,
)

# Register our API endpoints
app.include_router(router)


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Liveness probe — returns 200 if the service is running."""
    return {"status": "ok", "service": "healthcare-rag"}


# Mount static files folder to serve the frontend at root
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


