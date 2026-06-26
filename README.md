# 🏥 MediFetch — Clinical Healthcare RAG Assistant

> A **production-grade Retrieval-Augmented Generation (RAG)** system for answering clinical and medical queries grounded exclusively in uploaded healthcare guidelines — not internet data.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![BioBERT](https://img.shields.io/badge/Embeddings-BioBERT-orange)](https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Table of Contents

- [What is MediFetch?](#what-is-medifetch)
- [System Architecture](#system-architecture)
- [Pipeline Deep-Dive with Tradeoffs](#pipeline-deep-dive-with-tradeoffs)
  - [1. PDF Ingestion](#1-pdf-ingestion)
  - [2. Text Chunking](#2-text-chunking)
  - [3. Embeddings — BioBERT](#3-embeddings--biobert)
  - [4. Dense Retrieval — FAISS](#4-dense-retrieval--faiss)
  - [5. Sparse Retrieval — BM25](#5-sparse-retrieval--bm25)
  - [6. Hybrid Fusion — RRF](#6-hybrid-fusion--rrf)
  - [7. Reranking — Cross-Encoder](#7-reranking--cross-encoder)
  - [8. LLM Generation](#8-llm-generation)
  - [9. Safety Guardrails](#9-safety-guardrails)
  - [10. Confidence Scoring](#10-confidence-scoring)
  - [11. Feedback Loop (RLHF)](#11-feedback-loop-rlhf)
- [Directory Structure](#directory-structure)
- [Setup & Installation](#setup--installation)
- [Configuration Reference](#configuration-reference)
- [API Endpoints Reference](#api-endpoints-reference)
- [Running Tests](#running-tests)

---

## What is MediFetch?

MediFetch answers clinical questions **only from documents you upload** — no hallucinated internet data, no general knowledge leaking into clinical answers. It is designed for:

- Clinicians querying drug protocols, dosage guidelines, and treatment algorithms
- Hospital systems indexing internal SOPs and clinical guidelines
- Medical education platforms grounding AI answers in textbooks

It uses a **hybrid search pipeline** (dense + sparse retrieval, fused by Reciprocal Rank Fusion) with Cross-Encoder reranking, giving each answer a **confidence score** backed by the retrieved source context.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION                    │
│                                                          │
│  PDF Upload → PyMuPDF Loader → Recursive Chunker         │
│           → BioBERT Embedder → FAISS + BM25 Index        │
└──────────────────────────┬──────────────────────────────┘
                           │ (persisted to disk)
┌──────────────────────────▼──────────────────────────────┐
│                     QUERY PIPELINE                       │
│                                                          │
│  User Query                                              │
│     │                                                    │
│     ├─ Crisis Keyword Check ──────► Crisis Response      │
│     │                                                    │
│     ├─ BioBERT Embed ──► FAISS Search (top-15 dense)     │
│     ├─ Tokenize ────────► BM25 Search  (top-15 sparse)   │
│     │                                                    │
│     └─ RRF Fusion ──► top-10 unique candidates           │
│                │                                         │
│         Cross-Encoder Reranker ──► top-3 chunks          │
│                │                                         │
│         Confidence Score + Level                         │
│                │                                         │
│         LLM Generation (Groq / OpenAI / Gemini / Ollama) │
│                │                                         │
│         Answer + Citations + Confidence Badge            │
└─────────────────────────────────────────────────────────┘
```

---

## Pipeline Deep-Dive with Tradeoffs

Each component was chosen deliberately. Here is the reasoning and the tradeoffs at every stage.

---

### 1. PDF Ingestion

**Implementation**: [`app/loaders/pdf_loader.py`](app/loaders/pdf_loader.py)  
**Library**: `PyMuPDF` (fitz)

PyMuPDF extracts structured, page-level text from PDFs while preserving layout metadata (page numbers). Each page is returned as a `DocumentPage` Pydantic model.

| ✅ Why PyMuPDF | ⚖️ Tradeoff vs Alternatives |
|---|---|
| Fastest pure-Python PDF parser available | vs `pdfplumber`: pdfplumber has better table extraction but is ~3× slower |
| Handles scanned PDFs with embedded text | vs `pypdf`: pypdf often loses formatting and hyphenated words |
| Returns per-page text — source tracking is trivial | vs `Apache Tika`: Tika needs a Java runtime, adds ops overhead |
| Actively maintained, 1.5M downloads/week | vs `camelot`: only for tables, not general clinical text |

> **Limitation**: Does not OCR scanned image PDFs. For those, you'd need `pytesseract` + `pdf2image` as a pre-processing step.

---

### 2. Text Chunking

**Implementation**: [`app/loaders/chunker.py`](app/loaders/chunker.py)  
**Strategy**: Recursive character splitting with overlap

Text is split hierarchically: paragraphs (`\n\n`) → newlines (`\n`) → spaces. Chunks are 500 characters with a 50-character overlap, preserving the source page number on every chunk.

| ✅ Why Recursive Chunking | ⚖️ Tradeoff vs Alternatives |
|---|---|
| Respects natural paragraph boundaries | vs Fixed-size: fixed-size cuts mid-sentence, destroying semantic units |
| Overlap ensures context isn't lost at chunk edges | vs Sentence splitting (spaCy/NLTK): slower, needs NLP model loaded at ingest time |
| No external NLP dependency needed | vs Semantic chunking (embed, cluster): much slower, overkill for structured guidelines |
| 500-char chunks fit within BioBERT's 512-token limit | vs Large chunks: larger chunks exceed embedding model limits |

> **Tuning**: `chunk_size=500` and `chunk_overlap=50` are configurable via `config.py`. For dense clinical tables, reduce to 300/30. For narrative text, 700/70 works better.

---

### 3. Embeddings — BioBERT

**Implementation**: [`app/embeddings/biobert.py`](app/embeddings/biobert.py)  
**Model**: `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`

Chunks are embedded into 768-dimensional vectors using a BioBERT model fine-tuned on multiple biomedical NLI and STS datasets. The model runs entirely locally via `sentence-transformers`.

| ✅ Why BioBERT | ⚖️ Tradeoff vs Alternatives |
|---|---|
| Pre-trained on PubMed + PMC — understands medical terminology natively | vs `all-MiniLM-L6-v2`: smaller/faster but trained on general text, misses clinical synonyms |
| Bi-encoder architecture enables fast offline batch encoding | vs OpenAI `text-embedding-3-small`: better general quality but costs money per token, no offline use |
| 768-dim vectors offer rich semantic space | vs `PubMedBERT`: similar quality but not fine-tuned for similarity tasks |
| Runs on CPU — no GPU required | vs `MedCPT`: better clinical IR performance but harder to run locally |

> **Limitation**: BioBERT is slower than MiniLM (~3× at inference). On large document sets (>500 pages), consider batching with `batch_size=32` or switching to a faster model for indexing throughput.

---

### 4. Dense Retrieval — FAISS

**Implementation**: [`app/retriever/vector_store.py`](app/retriever/vector_store.py)  
**Index type**: `IndexFlatIP` (exact inner product / cosine similarity after L2 normalization)

All BioBERT chunk embeddings are stored in a FAISS flat index. At query time, the query is embedded and the top-15 most similar vectors are returned via exact nearest-neighbor search.

| ✅ Why FAISS Flat | ⚖️ Tradeoff vs Alternatives |
|---|---|
| Exact search — no approximation error | vs `IndexIVFFlat`: faster at scale (>100K chunks) but needs training step and introduces recall loss |
| Zero latency overhead for clinical-scale document sets (<10K chunks) | vs `IndexHNSW`: faster ANN with good recall, but uses more RAM |
| Fully offline, no network calls | vs Pinecone / Weaviate / Qdrant: managed vector DBs, scalable but need internet + cost money |
| Serializes to a single `.faiss` binary file | vs `ChromaDB`: embeds SQLite overhead, harder to migrate |

> **Scaling note**: `IndexFlatIP` is ideal up to ~50K chunks. Beyond that, switch to `IndexIVFFlat` with `nlist=100` for 10× speedup with <1% recall loss.

---

### 5. Sparse Retrieval — BM25

**Implementation**: [`app/retriever/bm25.py`](app/retriever/bm25.py)  
**Library**: `rank-bm25`

BM25 performs classic term-frequency/inverse-document-frequency keyword matching. The same chunks indexed in FAISS are also indexed in BM25, enabling exact keyword hits that dense search misses (drug names, ICD codes, numeric dosages).

| ✅ Why BM25 | ⚖️ Tradeoff vs Alternatives |
|---|---|
| Catches exact keyword matches BioBERT misses (e.g., "metformin 500mg", "ICD-10 E11.9") | vs Dense-only: dense search can miss exact drug names/codes due to embedding space compression |
| Zero additional model — pure Python, instant startup | vs SPLADE: learned sparse model, better recall but needs a transformer model loaded |
| Proven baseline in information retrieval research | vs TF-IDF: BM25 outperforms TF-IDF on short query IR tasks consistently |
| Serializable to `.pkl` for fast reload | vs Elasticsearch: full-text search at scale, but requires a running cluster |

> **Limitation**: BM25 has no semantic understanding. "heart attack" and "myocardial infarction" are zero overlap in BM25 — that's exactly why we pair it with dense retrieval.

---

### 6. Hybrid Fusion — RRF

**Implementation**: [`app/rag/pipeline.py`](app/rag/pipeline.py) → `reciprocal_rank_fusion()`  
**Formula**: `RRF(d) = Σ 1 / (k + rank(d))` where `k=60`

The top-15 dense candidates and top-15 sparse candidates are merged using Reciprocal Rank Fusion. Documents appearing in both lists get score contributions from both. The top-10 unique chunks are forwarded to the Cross-Encoder.

| ✅ Why RRF | ⚖️ Tradeoff vs Alternatives |
|---|---|
| Score-free: works with any ranking, no score normalization needed | vs Linear combination: requires normalizing FAISS cosine scores + BM25 scores to the same scale — notoriously tricky |
| Consistently outperforms individual retrievers in IR benchmarks | vs Re-embed with a fusion model: much more compute, no clear gain for this scale |
| `k=60` is the empirically validated default from the original RRF paper | vs Max/Min pooling: loses rank information |
| Simple, auditable, no trainable parameters | vs Learned sparse + dense fusion: needs labeled training data |

> **Tuning**: Increasing `k` beyond 60 reduces the influence of top ranks (flattens scores). Decrease `k` to amplify the impact of being ranked #1. Keep at 60 unless you have labeled data to tune against.

---

### 7. Reranking — Cross-Encoder

**Implementation**: [`app/reranker/cross_encoder.py`](app/reranker/cross_encoder.py)  
**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

The top-10 RRF candidates are individually scored by a Cross-Encoder model that reads the full (query, chunk) pair jointly — giving it much richer relevance signal than a bi-encoder. Only the top-3 scoring chunks (above threshold `-3.0`) are passed to the LLM.

| ✅ Why Cross-Encoder | ⚖️ Tradeoff vs Alternatives |
|---|---|
| Full joint attention over query + document — far more accurate than bi-encoder similarity | vs Bi-encoder-only: faster but loses fine-grained relevance signal |
| MiniLM-L6 is small (22M params) — low latency even on CPU | vs `ms-marco-MiniLM-L-12-v2`: ~2× more accurate but ~2× slower |
| Returns calibrated logit scores usable as confidence proxy | vs `monoT5`: better reranking quality but 220M params, too slow for CPU inference |
| Only runs on top-10 candidates (not all chunks) — keeps latency acceptable | vs Running on all chunks: would be 10–50× slower |

> **Threshold**: `RERANK_SCORE_THRESHOLD=-3.0` was calibrated on synthetic medical data. A score ≥ 1.0 reliably indicates high relevance; scores below -3.0 are near-random matches. Adjust based on your specific document domain.

---

### 8. LLM Generation

**Implementation**: `app/llm/` — `openai_client.py`, `gemini_client.py`, `ollama_client.py`, `mock_client.py`

The assembled context (top-3 reranked chunks) and query are formatted using a clinical system prompt and sent to the configured LLM. The system prompt strictly instructs the model to answer **only from the context** and cite sources.

| Provider | Model | Cost | Privacy | Speed |
|---|---|---|---|---|
| **Groq** (via OpenAI-compatible API) | `llama-3.3-70b-versatile` | Free tier (generous limits) | Cloud | ⚡ Very fast (LPU) |
| **OpenAI** | `gpt-4o-mini` | ~$0.15/1M tokens | Cloud | Fast |
| **Google Gemini** | `gemini-1.5-flash` | Free tier available | Cloud | Fast |
| **Ollama** (local) | `llama3`, `mistral`, etc. | Free | 🔒 Fully private | Slow on CPU |
| **Mock** | — | Free | Local | Instant (testing) |

| ✅ Why multi-provider design | ⚖️ Tradeoff |
|---|---|
| Users can switch providers without code changes — just `.env` config | vs Single-provider: simpler but locks you in |
| Groq gives GPT-4 class quality at zero cost via free tier | vs OpenAI: costs money, but has broader model selection |
| Ollama enables fully air-gapped / HIPAA-safe deployments | vs Cloud LLMs: offline means no data leaves the machine |
| Mock client enables CI/CD testing without API calls | vs No mock: tests become expensive and flaky |

---

### 9. Safety Guardrails

**Implementation**: [`app/rag/pipeline.py`](app/rag/pipeline.py) — crisis intercept at the top of `answer_query()`

Before any retrieval occurs, the query is scanned for self-harm / crisis keywords. On detection, the pipeline immediately returns crisis helpline information — bypassing all search and LLM calls.

**Crisis keywords monitored**: `"i want to die"`, `"how can i die"`, `"suicide"`, `"kill myself"`, `"end my life"`, `"self harm"`, `"harm myself"`, `"want to end it"`

| ✅ Why keyword intercept | ⚖️ Tradeoff vs Alternatives |
|---|---|
| Zero-latency — no LLM call needed, instant safe response | vs LLM-based classifier: slower, may miss edge cases, costs tokens |
| Deterministic — same input always produces same safe output | vs Embedding similarity to crisis phrases: probabilistic, harder to audit |
| Easy to audit and extend by non-ML engineers | vs Moderation API (OpenAI): requires an additional API call, adds latency |

> **Limitation**: Keyword matching can be evaded by paraphrasing. For higher-stakes deployments, layer in OpenAI's Moderation API or a dedicated safety classifier as a secondary check.

---

### 10. Confidence Scoring

**Implementation**: [`app/rag/pipeline.py`](app/rag/pipeline.py) + [`app/api/endpoints.py`](app/api/endpoints.py)

The Cross-Encoder's top logit score for the best-matching chunk is used directly as the confidence proxy:

| Score Range | Level | Meaning |
|---|---|---|
| ≥ 1.0 | **HIGH** | Strong match found in uploaded documents |
| ≥ -1.0 | **MEDIUM** | Partial match — answer likely grounded but verify |
| ≥ -3.0 | **LOW** | Weak match — treat with caution |
| < -3.0 (filtered) | **NONE (FALLBACK)** | No document match — answered from LLM general knowledge |

| ✅ Why Cross-Encoder score as confidence | ⚖️ Tradeoff |
|---|---|
| Already computed during reranking — zero extra cost | vs Separate confidence model: extra inference step |
| Interpretable: score directly reflects query-document relevance | vs LLM self-reported confidence: LLMs notoriously overstate confidence |
| Allows hard threshold filtering (no low-quality context sent to LLM) | vs Sending all context regardless: risks hallucinations from irrelevant chunks |

---

### 11. Feedback Loop (RLHF)

**Implementation**: [`app/api/endpoints.py`](app/api/endpoints.py) → `POST /api/feedback`

Users can upvote (👍) or downvote (👎) each response. Feedback is appended as JSON lines to `data/feedback.jsonl` with the query, answer, rating, and timestamp. This data is structured for future supervised fine-tuning or RLHF reward model training.

| ✅ Why JSONL feedback log | ⚖️ Tradeoff |
|---|---|
| Simplest possible collection mechanism — no DB needed | vs SQLite / PostgreSQL: needed for multi-user deduplication and analytics at scale |
| Directly compatible with HuggingFace `datasets` for fine-tuning | vs Custom feedback schema: JSONL is the standard format for SFT/RLHF pipelines |
| Zero dependencies — just file I/O | vs Specialized RLHF platforms (Scale AI, Argilla): overkill for early-stage collection |

---

## Directory Structure

```text
MediFetch/
├── app/
│   ├── api/
│   │   └── endpoints.py         # FastAPI routes: upload, chat, feedback, clear
│   ├── embeddings/
│   │   └── biobert.py           # BioBERT sentence-transformer wrapper
│   ├── llm/
│   │   ├── base.py              # BaseLLM abstract interface
│   │   ├── openai_client.py     # OpenAI + Groq-compatible async client
│   │   ├── gemini_client.py     # Google Gemini async client (thread-pooled)
│   │   ├── ollama_client.py     # Local Ollama REST client
│   │   └── mock_client.py       # Deterministic mock for tests
│   ├── loaders/
│   │   ├── pdf_loader.py        # PyMuPDF page-level text extractor
│   │   ├── chunker.py           # Recursive character text splitter
│   │   └── schemas.py           # DocumentPage, DocumentChunk Pydantic models
│   ├── prompts/
│   │   └── templates.py         # System instruction + user prompt template
│   ├── rag/
│   │   └── pipeline.py          # Core coordinator: RRF fusion + rerank + LLM
│   ├── reranker/
│   │   ├── base.py              # BaseReranker abstract interface
│   │   └── cross_encoder.py     # ms-marco-MiniLM-L-6-v2 reranker
│   ├── retriever/
│   │   ├── vector_store.py      # FAISS IndexFlatIP wrapper + disk persistence
│   │   └── bm25.py              # BM25 sparse keyword retriever
│   ├── utils/
│   │   └── config.py            # Pydantic-settings config from .env
│   └── main.py                  # FastAPI app + lifespan (model loading)
├── tests/
│   ├── test_pdf_loader.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_reranker.py
│   ├── test_pipeline.py
│   ├── test_api.py
│   └── generate_synthesized_data.py
├── data/                        # Uploaded PDFs + feedback.jsonl (gitignored)
├── vector_store/                # FAISS .faiss + metadata.json + bm25.pkl
├── .env.example                 # Config template — copy to .env and fill keys
├── requirements.txt             # Pinned Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python **3.12**
- A free [Groq API key](https://console.groq.com) — or OpenAI / Gemini key

### 1. Clone the Repository

```bash
git clone https://github.com/ayush300302/Medifetch.git
cd MediFetch
```

### 2. Create and Activate Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ First install downloads BioBERT and Cross-Encoder models (~500MB). Subsequent startups load from cache.

### 4. Configure Environment

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
```

Edit `.env` with your keys:

```ini
# Using Groq (free, fast — recommended)
LLM_PROVIDER=openai
OPENAI_API_KEY=gsk_your_groq_key_here
OPENAI_MODEL=llama-3.3-70b-versatile
OPENAI_BASE_URL=https://api.groq.com/openai/v1

# OR using OpenAI directly
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-your_openai_key_here
# OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=

# OR using local Ollama (fully offline, no API key needed)
# LLM_PROVIDER=ollama
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=llama3
```

### 5. Run the Application

```bash
.\venv\Scripts\uvicorn app.main:app --reload   # Windows
uvicorn app.main:app --reload                  # macOS/Linux
```

Open **http://127.0.0.1:8000** in your browser.

- Interactive API docs: **http://127.0.0.1:8000/docs**
- Health check: **http://127.0.0.1:8000/health**

---

## Configuration Reference

All settings are in `.env` / `app/utils/config.py`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `gemini`, `ollama`, `mock` |
| `OPENAI_API_KEY` | — | OpenAI or Groq API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_BASE_URL` | — | Override for Groq: `https://api.groq.com/openai/v1` |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `EMBEDDING_MODEL` | `pritamdeka/BioBERT-...` | HuggingFace model ID for embeddings |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-Encoder model |
| `VECTOR_STORE_PATH` | `./vector_store/faiss_index` | Where FAISS index is saved |
| `TOP_K_RETRIEVAL` | `15` | Candidates retrieved from FAISS and BM25 each |
| `TOP_K_RERANK` | `3` | Top N chunks forwarded to LLM after reranking |
| `RERANK_SCORE_THRESHOLD` | `-3.0` | Minimum Cross-Encoder score to use a chunk |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## API Endpoints Reference

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/documents/upload` | Upload a clinical PDF guideline |
| `GET` | `/api/documents` | List all indexed guideline filenames |
| `DELETE` | `/api/documents` | Reset all indexes and delete uploaded PDFs |
| `POST` | `/api/chat` | Submit a clinical query, get answer + citations |
| `POST` | `/api/feedback` | Submit thumbs up/down rating on a response |
| `GET` | `/health` | Liveness probe |
| `GET` | `/docs` | Swagger interactive API documentation |

### `POST /api/chat` — Request / Response

**Request:**
```json
{
  "query": "What is the first-line treatment for type 2 diabetes?"
}
```

**Response:**
```json
{
  "query": "What is the first-line treatment for type 2 diabetes?",
  "answer": "According to the uploaded guidelines, metformin is the recommended first-line pharmacological therapy...",
  "citations": [
    {
      "chunk_id": "sample_guidelines.pdf_p3_0",
      "source": "sample_guidelines.pdf",
      "page_number": 3,
      "score": 2.14,
      "text": "Metformin is recommended as first-line therapy..."
    }
  ],
  "confidence_score": 2.14,
  "confidence_level": "HIGH"
}
```

---

## Running Tests

```bash
# All tests
.\venv\Scripts\pytest tests/ -v

# Individual component tests
.\venv\Scripts\python tests/test_pdf_loader.py
.\venv\Scripts\python tests/test_chunker.py
.\venv\Scripts\python tests/test_embeddings.py
.\venv\Scripts\python tests/test_reranker.py
.\venv\Scripts\python tests/test_pipeline.py
.\venv\Scripts\python tests/test_api.py
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for safer, grounded clinical AI.*
