# MediFetch — Complete Technical Deep-Dive

> **Architecture, build journey, tradeoffs, HLD/LLD, and future roadmap.**  
> Every decision explained — what we picked, what we rejected, and why.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [High-Level Design (HLD)](#2-high-level-design-hld)
3. [Build Journey — Start to End](#3-build-journey--start-to-end)
4. [Component Deep-Dives with Tradeoffs](#4-component-deep-dives-with-tradeoffs)
   - [4.1 PDF Ingestion](#41-pdf-ingestion)
   - [4.2 Text Chunking](#42-text-chunking)
   - [4.3 Clinical Embeddings — BioBERT](#43-clinical-embeddings--biobert)
   - [4.4 Dense Retrieval — FAISS](#44-dense-retrieval--faiss)
   - [4.5 Sparse Retrieval — BM25](#45-sparse-retrieval--bm25)
   - [4.6 Hybrid Fusion — RRF](#46-hybrid-fusion--rrf)
   - [4.7 Reranking — Cross-Encoder](#47-reranking--cross-encoder)
   - [4.8 Prompt Engineering](#48-prompt-engineering)
   - [4.9 LLM Provider Layer](#49-llm-provider-layer)
   - [4.10 Safety Guardrails](#410-safety-guardrails)
   - [4.11 Confidence Scoring](#411-confidence-scoring)
   - [4.12 Feedback Loop (RLHF)](#412-feedback-loop-rlhf)
5. [Low-Level Design (LLD) — Request Data Flow](#5-low-level-design-lld--request-data-flow)
6. [Bugs We Hit & How We Fixed Them](#6-bugs-we-hit--how-we-fixed-them)
7. [Future Versions Roadmap](#7-future-versions-roadmap)

---

## 1. Problem Statement

**Core challenge**: LLMs hallucinate. In a clinical context, a hallucinated drug dosage or incorrect treatment guideline can harm a patient.

**Requirements we defined:**
- Answers must be grounded *exclusively* in uploaded clinical PDFs
- System must refuse dangerous or out-of-scope queries
- Crisis queries (self-harm, suicide) must be intercepted before any LLM call
- Confidence must be quantified and shown to the user — not just "here's an answer"
- Must work for free or near-free — Groq's free tier, local Ollama, no paid infrastructure

**What we did NOT want:**
- A simple LLM chatbot with a PDF uploaded as context (too naive, context window limited)
- A dependency on Pinecone, Weaviate, or any cloud vector DB (costs money, privacy risk)
- Black-box answers with no source tracing

---

## 2. High-Level Design (HLD)

### System Boundary Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                        MEDIFETCH SYSTEM                              ║
║                                                                      ║
║  ┌──────────┐     ┌──────────────────────────────────────────────┐  ║
║  │  Browser │────►│              FastAPI App (main.py)           │  ║
║  │  (HTML/  │◄────│         Lifespan: loads all singletons       │  ║
║  │  JS/CSS) │     └────────────────────┬─────────────────────────┘  ║
║  └──────────┘                          │                            ║
║                           ┌────────────┼─────────────┐             ║
║                           │            │              │             ║
║                   ┌───────▼───┐  ┌─────▼──────┐  ┌──▼──────────┐ ║
║                   │  /api/    │  │  /api/chat  │  │ /api/       │ ║
║                   │ documents │  │  (RAG       │  │ feedback    │ ║
║                   │ /upload   │  │  Pipeline)  │  │ (RLHF log)  │ ║
║                   └───────────┘  └─────────────┘  └─────────────┘ ║
║                        │                │                          ║
║              ┌──────────▼────────┐      │                          ║
║              │  INGESTION LAYER  │      │                          ║
║              │                   │      │                          ║
║              │ PDFLoader         │      │                          ║
║              │ DocumentChunker   │      │                          ║
║              │ BioBERTEmbedder   │      │                          ║
║              │        │          │      │                          ║
║              │  ┌─────▼──────┐  │      │                          ║
║              │  │   FAISS    │  │      │                          ║
║              │  │ IndexFlatIP│  │      │                          ║
║              │  └────────────┘  │      │                          ║
║              │  ┌─────────────┐ │      │                          ║
║              │  │ BM25Okapi   │ │      │                          ║
║              │  └─────────────┘ │      │                          ║
║              └───────────────────┘      │                          ║
║                                         │                          ║
║              ┌──────────────────────────▼──────────────────────┐  ║
║              │                RAG PIPELINE                      │  ║
║              │                                                  │  ║
║              │  1. Crisis keyword check ──► Crisis Response     │  ║
║              │  2. FAISS similarity_search (top 15 dense)       │  ║
║              │  3. BM25 keyword search    (top 15 sparse)       │  ║
║              │  4. RRF Fusion             (top 10 unique)       │  ║
║              │  5. CrossEncoder rerank    (top 3 scored)        │  ║
║              │  6. Threshold filter       (score ≥ -3.0)        │  ║
║              │       ├── Empty → classify → fallback/OOS        │  ║
║              │       └── Context → Prompt → LLM → Answer        │  ║
║              │  7. Confidence level assignment                  │  ║
║              └──────────────────┬───────────────────────────────┘  ║
║                                 │                                  ║
║              ┌──────────────────▼───────────────────────────────┐  ║
║              │                 LLM LAYER                         │  ║
║              │                                                   │  ║
║              │  OpenAILLM ──► Groq (llama-3.3-70b) [default]   │  ║
║              │  GeminiLLM ──► Google Gemini 1.5 Flash           │  ║
║              │  OllamaLLM ──► Local llama3 / mistral            │  ║
║              │  MockLLM   ──► Deterministic stub (tests)        │  ║
║              └───────────────────────────────────────────────────┘  ║
║                                                                      ║
║  PERSISTENCE:                                                        ║
║  ├── vector_store/faiss_index/index.faiss   (FAISS binary)          ║
║  ├── vector_store/faiss_index/metadata.json (chunk↔vector map)      ║
║  ├── vector_store/faiss_index/bm25.pkl      (BM25 pickled model)    ║
║  └── data/feedback.jsonl                    (RLHF feedback log)     ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Component Ownership

| Layer | Files | Role |
|---|---|---|
| API | `app/api/endpoints.py` | HTTP routes, request/response schemas |
| Ingestion | `app/loaders/`, `app/embeddings/` | PDF parse → chunk → embed |
| Storage | `app/retriever/vector_store.py`, `bm25.py` | Index + search + persist |
| RAG Pipeline | `app/rag/pipeline.py` | Orchestrates retrieval → reranking → generation |
| Reranker | `app/reranker/cross_encoder.py` | Cross-Encoder scoring |
| LLM | `app/llm/` | Abstract interface + 4 provider implementations |
| Prompts | `app/prompts/templates.py` | System instruction + user template |
| Config | `app/utils/config.py` | Pydantic-settings from `.env` |
| Frontend | `app/static/` | HTML + CSS + JS single-page UI |

---

## 3. Build Journey — Start to End

### Phase 0 — Foundation: Understanding the Problem

Before writing a single line of code, we defined the non-negotiables:

1. **Hallucination prevention**: RAG (Retrieval-Augmented Generation) grounds every answer in retrieved documents
2. **Source attribution**: every answer must cite which PDF and which page number
3. **No cloud lock-in**: the entire system works offline with Ollama
4. **Clinical domain awareness**: general-purpose embeddings (`all-MiniLM`) don't understand "myocardial infarction" = "heart attack" — we needed BioBERT

### Phase 1 — Document Ingestion Pipeline

The first thing we built was ingestion — getting PDFs into a searchable form.

```
PDF file → PDFLoader (PyMuPDF) → List[PageDocument]
                                         ↓
                             DocumentChunker (recursive split)
                                         ↓
                                List[DocumentChunk]
                                (with source, page_number, chunk_id)
```

**Key design decision**: We chose page-level extraction (not document-level) from the start. This gave us free source attribution — every chunk knows its PDF filename and page number without any extra tracking logic.

**First bug we hit**: PyMuPDF returns hyphenated words split across lines (e.g., "treat-\nment"). We added a post-processing step to rejoin these before chunking.

### Phase 2 — Embedding + FAISS (Dense-Only RAG)

The first working version used **only** dense FAISS retrieval:

```
Chunk → BioBERT → 768-dim vector → FAISS IndexFlatIP
Query → BioBERT → 768-dim vector → cosine similarity → top-K chunks
```

This worked well for semantic queries like *"what is the treatment for high blood sugar?"* which would correctly match *"first-line therapy for hyperglycemia"* even though the exact words differ.

**Problem discovered**: Queries with specific drug names, codes, or dosages (e.g., *"metformin 500mg"*, *"ICD-10 E11.9"*) sometimes missed because BioBERT compresses meaning into a vector and specific tokens get diluted.

### Phase 3 — Adding BM25 + Hybrid RRF

We added BM25 as a parallel sparse retriever. Now both systems run on every query and their results are fused.

```
Query
  ├── BioBERT embed → FAISS → top 15 dense chunks
  └── Tokenize     → BM25  → top 15 sparse chunks
                          ↓
                    RRF Fusion → top 10 unique candidates
```

**Why RRF and not a weighted sum?**  
Score normalization between FAISS cosine similarity (typically -1 to 1) and BM25 TF-IDF scores (0 to ~20) is non-trivial and brittle. RRF uses only *rank positions*, not raw scores — so it's scale-invariant and requires zero tuning beyond the `k=60` constant from the original 2009 paper.

### Phase 4 — Cross-Encoder Reranking

The RRF fusion gives us 10 good candidates, but ordering among them is still rough. We added a Cross-Encoder to score each (query, chunk) pair with full joint attention.

**Why reranking is a two-stage process (not a one-stage Cross-Encoder on all chunks)?**

Running a Cross-Encoder on *all* chunks for every query would mean:
- 27 chunks × 1 forward pass each = 27 inference calls
- At scale (10,000 chunks) = 10,000 inference calls per query → unacceptably slow

By pre-filtering with FAISS+BM25 first, the Cross-Encoder only sees 10 candidates — fast enough on CPU (~200ms).

### Phase 5 — Confidence Scoring + Fallback Logic

After reranking, we needed a way to know: *is this answer actually grounded?*

The Cross-Encoder score is a logit — higher means more relevant. We defined thresholds empirically:

- Score ≥ 1.0 → **HIGH** (clearly relevant, from the document)
- Score ≥ -1.0 → **MEDIUM** (probably relevant)
- Score ≥ -3.0 → **LOW** (marginally relevant)
- Score < -3.0 → filtered out → **FALLBACK mode**

In fallback mode, we first classify the query (is it even medical?) using the LLM, then either answer from general medical knowledge with a disclaimer, or refuse if it's completely off-topic.

### Phase 6 — Safety Guardrails

We added a hard crisis intercept at the very top of `answer_query()`, before any retrieval or LLM call. This was deliberately a keyword check — not an LLM call — because:

- **Speed**: instant, no latency
- **Determinism**: the same input always produces the same safe output
- **Auditability**: a simple list anyone can review and extend

### Phase 7 — Multi-LLM Support

The system was initially hardcoded to OpenAI. We refactored to an abstract `BaseLLM` interface with 4 concrete implementations:

```python
class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_instruction: str = "") -> str:
        ...
```

This meant switching from OpenAI to Groq (free) required zero pipeline changes — just a different `.env` config.

### Phase 8 — Frontend + RLHF Loop

The last phase was the UI: a single-page app served directly by FastAPI's `StaticFiles`. We added:
- Confidence badges rendered from the API response
- Thumbs up/down feedback buttons
- Context inspector drawer showing the raw retrieved chunk text
- Markdown rendering for bold, links, and newlines in LLM responses

---

## 4. Component Deep-Dives with Tradeoffs

---

### 4.1 PDF Ingestion

**File**: `app/loaders/pdf_loader.py`  
**Library**: `PyMuPDF` (fitz)

#### How we built it

```python
import fitz  # PyMuPDF

doc = fitz.open(pdf_path)
for page in doc:
    text = page.get_text("text")  # plain text extraction
    pages.append(PageDocument(source=filename, page_number=page.number + 1, text=text))
```

We extract page-by-page, not the full document as one string. This is the foundational choice that enables page-level citations everywhere downstream.

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| PyMuPDF | Fastest, handles most PDFs, pure Python | `pdfplumber` | 3× slower, better tables but clinical PDFs rarely need table extraction |
| PyMuPDF | Active maintenance, 1.5M weekly downloads | `pypdf` | Loses hyphenation, whitespace mangling on complex layouts |
| Page-level extraction | Free source attribution downstream | Full-doc extraction | Would require expensive post-hoc page mapping |
| Plain text mode | Works for text-based PDFs | Layout mode | Layout mode is slower, rarely needed for narrative clinical text |

#### What v2 should do

- **OCR support**: Scanned image PDFs (common in hospitals) need `pytesseract` + `pdf2image`
- **Table extraction**: Drug dosage tables via `pdfplumber` or `camelot` for structured data queries
- **DOCX/HTML support**: Extend to other document formats via `python-docx` and `BeautifulSoup`

---

### 4.2 Text Chunking

**File**: `app/loaders/chunker.py`  
**Strategy**: Recursive character splitter with overlap

#### How we built it

We implemented the recursive splitter from scratch (not LangChain) to stay dependency-free. The algorithm:

1. Try to split on `\n\n` (paragraphs)
2. If still too big, split on `\n` (lines)
3. If still too big, split on ` ` (words)
4. Last resort: character-by-character

After splitting, small pieces are **merged back** into chunks up to `chunk_size=500` characters, with `chunk_overlap=50` characters retained between consecutive chunks.

```
Input text: "A B C D E" (simplified)
chunk_size=3, chunk_overlap=1

Chunks: ["A B C", "C D E"]  ← "C" appears in both (overlap)
```

The overlap ensures a sentence split at a chunk boundary doesn't lose context.

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| Recursive separator hierarchy | Respects paragraph structure | Fixed-size character split | Cuts mid-sentence, destroys semantic units |
| 500 char / 50 overlap | Fits within BioBERT's 512-token limit | 1000 chars | Exceeds BioBERT max tokens, truncation kills tail context |
| Custom implementation | Zero external dependencies | LangChain `RecursiveCharacterTextSplitter` | LangChain is a 50MB dependency for one class |
| Source/page tracking per chunk | Enables per-chunk citations | Post-hoc page inference | Post-hoc mapping is brittle when chunks span pages |

#### What v2 should do

- **Semantic chunking**: embed sentences, cluster by cosine similarity, cut at low-similarity boundaries — far better chunk coherence at the cost of indexing time
- **Sentence-aware splitting**: use spaCy's sentence segmenter as a primary splitter before falling back to characters
- **Dynamic chunk size**: shorter chunks (200 chars) for Q&A, larger (800 chars) for summarization queries

---

### 4.3 Clinical Embeddings — BioBERT

**File**: `app/embeddings/biobert.py`  
**Model**: `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`

#### How we built it

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", device="cpu")
vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
```

We normalize embeddings to unit length so that inner product (used by FAISS `IndexFlatIP`) equals cosine similarity. This means we can use the fast `IndexFlatIP` instead of the slower `IndexFlatL2`.

#### Why BioBERT specifically?

This model was fine-tuned on **6 biomedical NLI/STS datasets**:
- MedNLI (clinical notes inference)
- SciTail (science question entailment)  
- SNLI + MNLI (general NLI for transfer)
- BioASQ-style STS pairs

It understands that *"myocardial infarction"* ≈ *"heart attack"* ≈ *"MI"* in embedding space. A general model (`all-MiniLM`) does not reliably represent these as similar.

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| BioBERT (clinical fine-tuned) | Understands medical synonyms natively | `all-MiniLM-L6-v2` | 6× faster but fails on clinical synonyms — tested and confirmed |
| 768-dim vectors | Rich semantic space for clinical text | 384-dim (MiniLM) | Less expressive, struggles with multi-concept clinical queries |
| Local model (HuggingFace) | Fully offline, no per-token cost | OpenAI `text-embedding-3-small` | $0.02/1M tokens, can't work offline, privacy risk |
| Normalize embeddings | Enables IP ≡ cosine, faster FAISS | No normalization | Would require L2 index, slightly slower at search |

#### What v2 should do

- **MedCPT**: Microsoft's medical contrastive pre-training model — specifically built for clinical IR, outperforms our BioBERT on PubMed benchmarks
- **GPU support**: Enable `device="cuda"` via config for 10× embedding throughput
- **Embedding cache**: Hash chunk text → cache vector to avoid re-embedding unchanged documents

---

### 4.4 Dense Retrieval — FAISS

**File**: `app/retriever/vector_store.py`  
**Index**: `faiss.IndexFlatIP`

#### How we built it

```python
import faiss
import numpy as np

dimension = 768  # BioBERT output size
index = faiss.IndexFlatIP(dimension)

# Add vectors
vectors = np.array(embeddings, dtype=np.float32)
index.add(vectors)

# Search
query_vector = np.array([query_embedding], dtype=np.float32)
scores, indices = index.search(query_vector, k=15)
```

We maintain a `chunks_map: Dict[int, DocumentChunk]` that maps FAISS vector index positions → chunk metadata. This is serialized as `metadata.json` alongside the `.faiss` binary.

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| `IndexFlatIP` (exact search) | Zero recall loss, correct for <50K chunks | `IndexIVFFlat` (approximate) | Needs training step, 1-5% recall loss, overkill below 50K |
| `IndexFlatIP` | Fast for our scale (~27 chunks currently) | `IndexHNSWFlat` | Better ANN quality, but uses 2× RAM and is harder to serialize |
| Local FAISS binary | Fully offline, zero cost | Pinecone / Weaviate | Managed DBs cost money, require internet, privacy risk for clinical data |
| `metadata.json` sidecar | Human-readable, easy to debug | SQLite sidecar | Overkill for single-user deployment |

#### What v2 should do

- Switch to `IndexIVFFlat` at 10K+ chunks for sub-linear search time
- Add **filtered search**: FAISS metadata filtering (e.g., "only search in cardiology guidelines")
- **Multi-tenant isolation**: separate FAISS indexes per user/organization

---

### 4.5 Sparse Retrieval — BM25

**File**: `app/retriever/bm25.py`  
**Library**: `rank-bm25` (BM25Okapi variant)

#### How we built it

```python
from rank_bm25 import BM25Okapi

# Indexing
tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
bm25 = BM25Okapi(tokenized_corpus)

# Search
tokenized_query = tokenize(query)
scores = bm25.get_scores(tokenized_query)
```

Our tokenizer: lowercase → strip non-word characters (except hyphens, important for drug names like "5-fluorouracil") → split on whitespace.

We filter out results with score `<= 0.0` — zero score means zero term overlap, not useful.

The model is persisted to `bm25.pkl` via `pickle` for fast reload.

#### Why BM25 is non-negotiable in a clinical RAG system

Dense embeddings compress semantic meaning but lose precision on:
- Exact drug names (`metformin`, `lisinopril`, `amlodipine`)
- Dosage specifics (`500mg BID`, `10mg once daily`)
- ICD-10/ICD-11 codes (`E11.9`, `I10`, `J45.20`)
- Procedure codes (CPT, SNOMED CT)

BM25 treats these as exact tokens — a query for `metformin 500mg` will always score higher for chunks containing `metformin 500mg` than chunks containing only `diabetes medication`.

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| BM25Okapi | Industry standard for sparse IR, proven formula | TF-IDF | BM25 consistently outperforms TF-IDF on short-query IR benchmarks |
| BM25 | Zero additional model, instant | SPLADE | Learned sparse model, better recall, but requires a transformer — another model to load |
| Pickle serialization | Simple, fast save/load | Redis / Elasticsearch | Massive ops overhead for a single-machine deployment |

#### What v2 should do

- **SPLADE**: Sparse Lexical And Dense Expansion — learns to expand query and document tokens with related terms. Outperforms BM25 while remaining interpretable.
- **Stopword filtering**: Remove common medical filler words (`patient`, `history`, `showed`) to reduce noise in BM25 scores

---

### 4.6 Hybrid Fusion — RRF

**File**: `app/rag/pipeline.py` → `reciprocal_rank_fusion()`

#### How we built it

```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60, limit=10):
    rrf_scores = {}
    chunk_lookup = {}

    for rank, (chunk, _) in enumerate(dense_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(sparse_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup[chunk.chunk_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [chunk_lookup[cid] for cid in sorted_ids[:limit]]
```

A chunk ranked #1 in dense gets score `1/(60+1) = 0.0164`. If it's also ranked #3 in BM25: `1/(60+3) = 0.0159`. Combined: `0.033` — clearly ahead of a chunk only appearing in one list.

#### Why `k=60`?

The value 60 is from the original RRF paper (Cormack et al., 2009). It controls how much weight goes to top-ranked documents vs lower-ranked ones. At k=60, rank 1 vs rank 2 difference is small enough that an excellent rank-2 from both lists beats a rank-1 from one list. This is empirically optimal for most fusion tasks.

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| RRF | Score-free, rank-only, zero tuning needed | Linear score combination | Must normalize FAISS cosine (-1 to 1) + BM25 TF-IDF (0 to 20) to same scale — brittle |
| RRF | Proven in TREC benchmarks, consistent | Borda count | Borda is rank-only like RRF but doesn't handle missing entries as gracefully |
| k=60 | Empirically validated constant | Tuned k | Tuning needs labeled relevance data we don't have |

#### What v2 should do

- **Weighted RRF**: give dense results 1.5× weight for semantic queries, BM25 1.5× for code/exact queries — classified by query type
- **DBSFusion**: Reciprocal Rank Fusion's cousin with slightly better theoretical properties

---

### 4.7 Reranking — Cross-Encoder

**File**: `app/reranker/cross_encoder.py`  
**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

#### How we built it

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

pairs = [[query, chunk.text] for chunk in candidates]
scores = model.predict(pairs, show_progress_bar=False)

scored = [(chunk, float(score)) for chunk, score in zip(candidates, scores)]
scored.sort(key=lambda x: x[1], reverse=True)
return scored[:top_n]
```

The Cross-Encoder reads both the query and document chunk *together* in a single BERT forward pass. This full joint attention is what makes it far more accurate than a bi-encoder similarity score — it can detect when a document *answers* the query vs merely *containing similar words*.

#### Bi-encoder vs Cross-Encoder — the key distinction

```
Bi-Encoder (FAISS):
  Query  → embed → q_vec
  Doc    → embed → d_vec
  Score  = cosine(q_vec, d_vec)    ← vectors computed independently
  
Cross-Encoder:
  [Query, Doc] → joint BERT → single relevance score
  ← full attention between every query token and every doc token
```

The Cross-Encoder can detect negation, qualifiers, and conditional relevance that a bi-encoder misses. But it's too slow to run on every chunk — hence the two-stage pipeline.

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` | 22M params, fast on CPU, good MS MARCO performance | `ms-marco-MiniLM-L-12-v2` | Twice the layers → twice the latency, marginal quality gain for clinical text |
| Cross-Encoder after pre-filter | Only 10 candidates → fast enough on CPU | Cross-Encoder on all chunks | 27 chunks = manageable now; at 10K chunks this would be ~5 seconds per query |
| Only top-3 to LLM | Prevents context window overflow, forces quality | Top-10 to LLM | LLMs degrade with very long contexts — "lost in the middle" problem |

#### What v2 should do

- **`ms-marco-electra-base`**: ELECTRA-based Cross-Encoder, significantly better quality for clinical Q&A
- **Clinical-specific reranker**: Fine-tune on MedMCQA or medical IR datasets for domain-specific relevance signals
- **GPU acceleration**: `device="cuda"` reduces reranking from ~200ms to ~20ms

---

### 4.8 Prompt Engineering

**File**: `app/prompts/templates.py`

#### How we built it

Two-part prompt: a **system instruction** (role + guardrails) and a **user template** (context + query).

```
SYSTEM:
  "You are a professional clinical AI assistant.
   Answer ONLY from the provided context.
   If not found: say 'I cannot find the answer in the provided documents.'
   Do not speculate or add external knowledge.
   If off-topic: say 'I am a clinical assistant...'"

USER:
  "Refer to the following clinical guidelines:
   --
   {context}           ← top-3 reranked chunk texts
   --
   User Query: {query}
   Clinical Response:"
```

The explicit fallback phrases (`"I cannot find..."`) are critical — without them, LLMs will confidently confabulate an answer using their training data, bypassing the RAG entirely.

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| Hard verbatim refusal strings | Forces deterministic OOS behavior | Soft "try your best" instruction | LLMs interpret soft instructions as permission to hallucinate |
| System + User split | Works with all chat-completion APIs | Single combined prompt | Completion-style APIs (Ollama some models) handle them differently |
| Brief system prompt | Fits within all model context windows | Extremely detailed instruction | Longer prompts reduce answer space for actual context |

#### What v2 should do

- **Chain-of-thought prompting**: *"Think step by step before answering"* — improves accuracy on multi-hop clinical questions
- **Few-shot examples**: Include 2–3 (query, ideal answer) examples in the system prompt for formatting consistency
- **Structured output**: Force JSON output with `response_format={"type": "json_object"}` for machine-readable answers with confidence and citations embedded

---

### 4.9 LLM Provider Layer

**Files**: `app/llm/base.py`, `openai_client.py`, `gemini_client.py`, `ollama_client.py`, `mock_client.py`

#### How we built it

Abstract base class pattern:

```python
class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_instruction: str = "") -> str: ...
```

Each provider implements `generate()` independently. The pipeline only ever calls `self.llm.generate()` — it has no knowledge of which provider is active.

**OpenAI/Groq client** uses `AsyncOpenAI` with an optional `base_url` override:
```python
client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
```
Setting `base_url=https://api.groq.com/openai/v1` routes to Groq's API using OpenAI's exact protocol — zero code difference.

**Gemini client** had a major problem: `generate_content_async` hangs indefinitely on Windows/FastAPI event loops. Fix: use `asyncio.to_thread()` to run the synchronous SDK call in a thread pool.

#### Provider Comparison

| Provider | Model Used | Cost | Privacy | Latency | Limit |
|---|---|---|---|---|---|
| **Groq** | llama-3.3-70b-versatile | **Free** | Cloud | ⚡ ~1-2s (LPU) | 6K req/day |
| OpenAI | gpt-4o-mini | $0.15/1M in | Cloud | ~2-4s | None (paid) |
| Gemini | gemini-1.5-flash | Free tier | Cloud | ~2-3s | 15 req/min |
| Ollama | llama3, mistral | **Free** | **Local** | ~15-30s CPU | None |
| Mock | — | Free | Local | **Instant** | None |

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| Abstract `BaseLLM` | Provider-agnostic pipeline | Hardcoded OpenAI calls | One provider change = rewrite pipeline |
| `base_url` override for Groq | Groq is free + fast + uses OpenAI protocol | Groq SDK | OpenAI SDK is already installed, no extra dependency |
| `asyncio.to_thread` for Gemini | Prevents FastAPI event loop blocking | `generate_content_async` | Google SDK hangs on Windows — confirmed bug |
| MockLLM | CI/CD without API keys, free test runs | Mocking with `unittest.mock` | Mock at class level breaks integration tests |

---

### 4.10 Safety Guardrails

**File**: `app/rag/pipeline.py` — top of `answer_query()`

#### How we built it

```python
crisis_keywords = [
    "i want to die", "how can i die", "suicide", "kill myself",
    "end my life", "self harm", "harm myself", "want to end it"
]
if any(kw in query.lower().strip() for kw in crisis_keywords):
    return { "answer": CRISIS_RESPONSE, "confidence_level": "NONE (CRISIS INTERCEPT)" }
```

The crisis response provides:
- Empathetic acknowledgment
- 988 Suicide & Crisis Lifeline (US)
- Crisis Text Line (text HOME to 741741)
- findahelpline.com for international users

#### Two-layer guardrail design

```
Layer 1 — Crisis intercept (keyword, pre-retrieval)
  ↓ (only if no crisis keyword)
Layer 2 — Scope check (LLM classification, post-retrieval-miss)
  → "Is this query related to healthcare?"
  → YES → general medical fallback with disclaimer
  → NO  → "I am a clinical assistant" refusal
```

Layer 2 uses the LLM to classify scope because paraphrased non-medical queries (e.g., *"who won the World Cup?"*) won't be caught by keywords and will simply return no matching chunks from FAISS.

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| Keyword check pre-retrieval | Zero latency, always fires | LLM-based crisis detection | Adds 1-2s latency to a response that must be instant |
| Simple string match | Auditable by non-ML engineers | Embedding similarity to crisis phrases | Probabilistic, may miss edge cases, harder to audit |
| Hard-coded crisis response | Consistent, testable | LLM-generated crisis response | LLM may vary the message, possibly less safe |

#### What v2 should do

- **OpenAI Moderation API** as secondary layer — catches paraphrased or adversarial crisis queries
- **Regex + unicode normalization** — catch leet-speak or character substitution evasion
- **Audit log** — every crisis trigger written to a separate `crisis_log.jsonl` for compliance

---

### 4.11 Confidence Scoring

**File**: `app/rag/pipeline.py` (score assignment) + `app/api/endpoints.py` (response) + `app/static/app.js` (UI badge)

#### How we built it

The Cross-Encoder returns raw logit scores (unbounded real numbers). We use the top-scoring chunk's score as the answer confidence proxy:

```python
top_score = reranked[0][1]   # best Cross-Encoder score

if top_score >= 1.0:   level = "HIGH"
elif top_score >= -1.0: level = "MEDIUM"
elif top_score >= -3.0: level = "LOW"
else:                  # filtered by threshold — fallback mode
```

The UI renders a colored badge:
- 🟢 HIGH — green, check icon
- 🟡 MEDIUM — yellow, info icon  
- 🔴 LOW — orange, warning triangle
- ⚫ NONE — grey, shield icon (fallback/crisis/OOS)

#### Why Cross-Encoder score as confidence proxy?

Cross-Encoder logits are calibrated relevance scores: +5 means highly relevant, -5 means irrelevant. They directly reflect how well the retrieved document answers the query — which is exactly what "confidence this answer is correct" means in a RAG context.

Compare to LLM self-reported confidence: LLMs say "I'm 95% confident" on hallucinated answers all the time. The retrieval score is an objective, external signal.

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| Cross-Encoder score as confidence | Already computed, no extra cost, objective | Separate confidence model | Extra inference step, extra model to load |
| Threshold filtering at -3.0 | Empirically validated on synthetic medical data | No threshold (always send context) | Low-relevance chunks pollute LLM context → hallucinations |
| Show score to user | Transparency, clinicians can make their own call | Hide score, only show level | Hiding information reduces trust in clinical AI systems |

---

### 4.12 Feedback Loop (RLHF)

**File**: `app/api/endpoints.py` → `POST /api/feedback`  
**Storage**: `data/feedback.jsonl`

#### How we built it

```python
feedback_data = {
    "timestamp": time.time(),
    "query": feedback_req.query,
    "answer": feedback_req.answer,
    "rating": feedback_req.rating,   # +1 or -1
    "reason": feedback_req.reason
}
# Append to JSONL file
with open("data/feedback.jsonl", "a") as f:
    f.write(json.dumps(feedback_data) + "\n")
```

Each feedback entry is one JSON line. This format is directly compatible with HuggingFace `datasets.load_dataset("json", ...)` for supervised fine-tuning.

#### RLHF roadmap built into the design

```
Phase 1 (now): Collect feedback.jsonl
Phase 2: Use positively-rated (query, answer) pairs as SFT training data
Phase 3: Use +1/-1 pairs to train a reward model
Phase 4: Run PPO/DPO fine-tuning on a smaller open-source model (e.g., Mistral-7B)
```

#### Tradeoffs

| Choice | Why | Rejected Alternative | Why Rejected |
|---|---|---|---|
| JSONL append | Simplest possible, zero dependencies | PostgreSQL | Overkill for current scale, adds ops overhead |
| `+1 / -1` rating | Maps directly to RLHF reward signal | Star ratings (1-5) | Ordinal ratings need normalization before RLHF use |
| Local file storage | Privacy — feedback stays on-premise | Cloud feedback platform | PHI in clinical feedback can't go to third-party services |

---

## 5. Low-Level Design (LLD) — Request Data Flow

### Document Upload Flow

```
POST /api/documents/upload
        │
        ▼
Validate file extension (.pdf)
        │
        ▼
Save to data/{filename}.pdf
        │
        ▼
PDFLoader.load(path)
  └─► fitz.open() → page-by-page text extraction
  └─► Returns LoadedDocument { pages: List[PageDocument] }
        │
        ▼
DocumentChunker.chunk_documents(pages)
  └─► _split_text() — recursive separator hierarchy
  └─► _merge_splits() — overlap reassembly
  └─► Returns List[DocumentChunk] (each with chunk_id, source, page_number)
        │
        ▼
FAISSVectorStore.add_chunks(chunks)
  └─► BioBERTEmbedder.encode(texts) → float32 [N, 768] array
  └─► FAISS index.add(vectors)
  └─► Update chunks_map { int → DocumentChunk }
  └─► vector_store.save() → index.faiss + metadata.json
        │
BM25Retriever.add_chunks(chunks)
  └─► Tokenize all chunks
  └─► BM25Okapi(tokenized_corpus) → rebuild index
  └─► bm25_retriever.save() → bm25.pkl
        │
        ▼
Return UploadResponse { chunks_created: N, status: "success" }
```

### Chat Query Flow

```
POST /api/chat { query: "..." }
        │
        ▼
Guard: vector_store.is_empty() → 400 if no docs indexed
        │
        ▼
RAGPipeline.answer_query(query)
        │
        ├─[Step 0]─► Crisis keyword scan
        │              → Match? → Return crisis response immediately
        │
        ├─[Step 1]─► FAISSVectorStore.similarity_search(query, k=15)
        │              └─► Embed query → 768-dim vector
        │              └─► index.search() → (distances, indices)
        │              └─► Map indices → chunks_map → List[(DocumentChunk, score)]
        │
        ├─[Step 2]─► BM25Retriever.search(query, k=15)
        │              └─► Tokenize query
        │              └─► bm25.get_scores() → scored chunks
        │              └─► Filter score > 0, sort desc → List[DocumentChunk]
        │
        ├─[Step 3]─► reciprocal_rank_fusion(dense, sparse, k=60, limit=10)
        │              └─► Score = Σ 1/(60 + rank + 1) across both lists
        │              └─► Sort by RRF score → top 10 unique DocumentChunks
        │
        ├─[Step 4]─► CrossEncoderReranker.rerank(query, candidates, top_n=3)
        │              └─► pairs = [[query, chunk.text] for chunk in candidates]
        │              └─► model.predict(pairs) → float scores
        │              └─► Sort desc → [(DocumentChunk, score), ...][:3]
        │
        ├─[Step 5]─► Threshold filter: score >= -3.0
        │              └─► Empty? → LLM scope classification → fallback / OOS refusal
        │
        ├─[Step 6]─► Build context string from top-3 chunks
        │              └─► "[Source: X, Page: N]\n{chunk.text}"
        │
        ├─[Step 7]─► Format prompt
        │              └─► USER_PROMPT_TEMPLATE.format(context=..., query=...)
        │
        ├─[Step 8]─► LLM.generate(prompt, system_instruction=SYSTEM_INSTRUCTION)
        │              └─► OpenAI/Groq: AsyncOpenAI.chat.completions.create()
        │              └─► Gemini: asyncio.to_thread(model.generate_content)
        │              └─► Ollama: httpx.AsyncClient POST /api/generate
        │
        ├─[Step 9]─► Assign confidence level from top reranker score
        │
        └─[Step 10]► Return {query, answer, citations, confidence_score, confidence_level}
                       │
                       ▼
                API serializes to ChatResponse JSON
                       │
                       ▼
               app.js renderBotAnswer()
               └─► Confidence badge (colored by level)
               └─► renderMarkdown(answer) → HTML
               └─► Citation chips (source + page)
               └─► Feedback thumbs up/down
```

---

## 6. Bugs We Hit & How We Fixed Them

| Bug | Root Cause | Fix |
|---|---|---|
| **Confidence always showing NONE** | `endpoints.py` built `ChatResponse(...)` without passing `confidence_score`/`confidence_level` fields | Added `confidence_score=result.get(...)` and `confidence_level=result.get(...)` to the constructor |
| **Markdown showing as raw text** | `app.js` used `textElement.innerText = answer` which treats `**bold**` as literal characters | Added `renderMarkdown()` function, switched to `innerHTML = renderMarkdown(answer)` |
| **Gemini hangs forever** | `google-generativeai` SDK's `generate_content_async` blocks the asyncio event loop on Windows | Wrapped sync `model.generate_content()` in `asyncio.to_thread()` |
| **Crisis response not triggering for mixed case** | Keyword check was exact string match, "I Want To Die" didn't match "i want to die" | Added `.lower().strip()` normalization before keyword scan |
| **BM25 not persisting between restarts** | BM25 pkl path wasn't checked on startup | Added `if (store_dir / "bm25.pkl").exists(): bm25_retriever.load(store_dir)` to lifespan |

---

## 7. Future Versions Roadmap

### v2 — Production Hardening (Next)

| Feature | Why | How |
|---|---|---|
| **SSE Streaming** | 10s spinner kills UX | `stream=True` in OpenAI call → `StreamingResponse` → `EventSource` in JS |
| **Rate Limiting** | Prevent API abuse | `slowapi` — 10 req/min chat, 5 req/min upload |
| **CORS middleware** | Required for cross-origin frontends | `CORSMiddleware` in `main.py` |
| **Background upload jobs** | Large PDFs block HTTP thread | FastAPI `BackgroundTasks` → return `202 + job_id` → poll `GET /api/jobs/{id}` |
| **Multi-turn memory** | "What about the dosage?" needs prior context | Session ID + last-5-turns in memory, prepended to prompt |
| **Structured logging** | Debugging in production | `structlog` JSON logs with `request_id`, `latency_ms`, `confidence_level` |
| **Docker** | One-command deployment | `Dockerfile` + `docker-compose.yml` |

### v3 — Model Upgrades

| Feature | Why | How |
|---|---|---|
| **MedCPT embeddings** | Better clinical IR than BioBERT | Swap embedding model in `config.py` |
| **SPLADE sparse retrieval** | Outperforms BM25 with learned expansion | Add `SPLADE` class alongside `BM25Retriever` |
| **Streaming Cross-Encoder** | Reduce reranking latency | Batch prediction with async thread pool |
| **Fine-tuned reranker** | ms-marco trained on web queries, not clinical | Fine-tune on MedMCQA dataset |
| **ColBERT** | Late-interaction model — best of bi-encoder + cross-encoder | `RAGatouille` library integration |

### v4 — Multi-User & HIPAA

| Feature | Why | How |
|---|---|---|
| **User authentication** | Isolate document access per clinician | JWT tokens + `python-jose` |
| **Per-user FAISS indexes** | Prevent cross-user data leakage | Separate FAISS index per `user_id` |
| **Audit log** | HIPAA compliance | Every query + response logged with user_id + timestamp |
| **Encryption at rest** | PHI protection | Encrypt FAISS index + PDFs with `cryptography` |
| **PostgreSQL feedback** | Scalable RLHF data collection | Replace `feedback.jsonl` with async SQLAlchemy writes |

### v5 — RLHF Fine-Tuning

| Feature | Why | How |
|---|---|---|
| **SFT dataset export** | Turn collected feedback into training data | `GET /api/export/sft` → JSONL in HuggingFace format |
| **Reward model training** | Learn from +1/-1 ratings | Train BERT classifier on (query, answer, rating) triplets |
| **DPO fine-tuning** | Align model to clinician preferences | Direct Preference Optimization on Mistral-7B using feedback pairs |
| **A/B testing** | Compare fine-tuned vs base model live | Route 50% traffic to fine-tuned model, compare feedback ratings |

---

*MediFetch — built bottom-up, every component chosen deliberately, every tradeoff documented.*
