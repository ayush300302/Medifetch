"""
RAG Pipeline — Coordinates retrieval, reranking, and generation.
"""

import logging
from typing import Dict, Any, List

from app.loaders.schemas import DocumentChunk
from app.retriever.vector_store import FAISSVectorStore
from app.retriever.bm25 import BM25Retriever
from app.reranker.base import BaseReranker
from app.llm.base import BaseLLM
from app.prompts.templates import SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE
from app.utils.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Coordinates the retrieval, reranking, and generation pipeline.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_retriever: BM25Retriever,
        reranker: BaseReranker,
        llm: BaseLLM,
    ):
        """
        Initializes the pipeline with required components.

        Args:
            vector_store: Instantiated FAISSVectorStore.
            bm25_retriever: Instantiated BM25Retriever.
            reranker: Instantiated BaseReranker (e.g. CrossEncoderReranker).
            llm: Instantiated BaseLLM connector.
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.llm = llm

    def reciprocal_rank_fusion(
        self,
        dense_results: List[tuple],
        sparse_results: List[DocumentChunk],
        k: int = 60,
        limit: int = 10,
    ) -> List[DocumentChunk]:
        """
        Fuses dense vector results and sparse term results using Reciprocal Rank Fusion.
        """
        rrf_scores = {}
        chunk_lookup = {}

        # 1. Accumulate scores from dense search
        for rank, (chunk, _) in enumerate(dense_results):
            chunk_id = chunk.chunk_id
            chunk_lookup[chunk_id] = chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))

        # 2. Accumulate scores from sparse keyword search
        for rank, chunk in enumerate(sparse_results):
            chunk_id = chunk.chunk_id
            chunk_lookup[chunk_id] = chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))

        # 3. Sort chunk IDs based on RRF scores descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        logger.debug(
            "RRF fused %d dense and %d sparse candidates down to top %d.",
            len(dense_results), len(sparse_results), min(limit, len(sorted_chunk_ids))
        )
        return [chunk_lookup[cid] for cid in sorted_chunk_ids[:limit]]

    async def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Answers a user query using the full RAG pipeline (FAISS retrieval -> Cross-Encoder reranking -> LLM call).

        Args:
            query: Clinical or user search query.

        Returns:
            Dict containing:
              - 'query': the original query string
              - 'answer': the generated clinical text
              - 'citations': list of source dictionaries (source file, page, score)
        """
        logger.info("Executing RAG pipeline for query: '%s'", query)

        # ── Crisis & Self-Harm Safety Guardrail ──
        crisis_keywords = [
            "i want to die", "how can i die", "suicide", "kill myself", 
            "end my life", "self harm", "harm myself", "want to end it"
        ]
        query_lower = query.lower().strip()
        if any(keyword in query_lower for keyword in crisis_keywords):
            logger.warning("Crisis query detected: '%s'. Triggering crisis response.", query)
            return {
                "query": query,
                "answer": (
                    "Please know that you are not alone, and there is support available. "
                    "If you are feeling overwhelmed, in pain, or having thoughts of self-harm, "
                    "please reach out for help immediately. You can speak with someone who cares "
                    "and wants to support you.\n\n"
                    "📞 **Suicide & Crisis Lifeline**: Call or text **988** (Available 24/7, free, and confidential in the US & Canada).\n"
                    "💬 **Crisis Text Line**: Text **HOME** to **741741**.\n"
                    "🌍 **International Resources**: If you are outside the US, please contact your local emergency services or visit [findahelpline.com](https://findahelpline.com/) to find support in your country.\n\n"
                    "Please stay safe, take a deep breath, and connect with a healthcare professional or crisis service."
                ),
                "citations": [],
                "confidence_score": -10.0,
                "confidence_level": "NONE (CRISIS INTERCEPT)",
            }

        # 1. Hybrid Search (Dense FAISS + Sparse BM25)
        # Retrieve up to 15 candidates from each to ensure high recall
        dense_candidates = self.vector_store.similarity_search(query, k=15)
        sparse_candidates = self.bm25_retriever.search(query, k=15)
        logger.debug("Retrieval candidate counts - FAISS: %d, BM25: %d", len(dense_candidates), len(sparse_candidates))

        # 2. Reciprocal Rank Fusion (RRF)
        # Keep top 10 unique candidates to send to Cross-Encoder
        candidates = self.reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60, limit=10)

        # 3. Rerank candidates using Cross-Encoder
        top_n = settings.TOP_K_RERANK
        reranked_candidates = self.reranker.rerank(query, candidates, top_n=top_n)
        
        # Enforce confidence threshold filter
        reranked = [item for item in reranked_candidates if item[1] >= settings.RERANK_SCORE_THRESHOLD]
        logger.info(
            "Rerank filter retained %d/%d chunks above threshold %.2f", 
            len(reranked), len(reranked_candidates), settings.RERANK_SCORE_THRESHOLD
        )

        if not reranked:
            # Retrieve classification from LLM to identify out-of-scope queries
            classification_prompt = (
                f"Identify if the following user query is related to healthcare, medicine, clinical guidelines, "
                f"or human biology. Reply with exactly one word: 'YES' or 'NO'.\n\n"
                f"Query: {query}\n\n"
                f"Response:"
            )
            try:
                is_medical_str = await self.llm.generate(classification_prompt)
                is_medical = "yes" in is_medical_str.lower().strip()
            except Exception as exc:
                logger.error("Failed to classify query scope: %s. Defaulting to YES.", exc)
                is_medical = True # Sane fallback on error

            if not is_medical:
                logger.info("Query '%s' classified as out-of-scope.", query)
                return {
                    "query": query,
                    "answer": (
                        "I am a clinical assistant. I can only assist with healthcare queries "
                        "and medical guidelines based on your uploaded documents."
                    ),
                    "citations": [],
                    "confidence_score": -10.0,
                    "confidence_level": "NONE (OUT-OF-SCOPE)",
                }

            logger.info("Query '%s' going into general medical fallback mode.", query)
            general_prompt = (
                f"You are a clinical AI assistant. The user's query could not be matched with any specific "
                f"clinical guidelines in the uploaded documents.\n\n"
                f"Please answer their medical query using your general medical and clinical knowledge. "
                f"You MUST start your response with this exact disclaimer:\n"
                f"\"[General Medical Information - Not Guideline Grounded] \"\n"
                f"In your response, advise the user that this is general information, explain potential general causes "
                f"or considerations, and emphasize that they should consult a healthcare professional. "
                f"Keep it precise, objective, and helpful.\n\n"
                f"User Query: {query}\n\n"
                f"Clinical Response:"
            )
            try:
                answer = await self.llm.generate(general_prompt)
                return {
                    "query": query,
                    "answer": answer.strip(),
                    "citations": [],
                    "confidence_score": -10.0,
                    "confidence_level": "NONE (FALLBACK)",
                }
            except Exception as exc:
                logger.error("General medical fallback generation failed: %s", exc)
                return {
                    "query": query,
                    "answer": f"Error during general medical fallback: {exc}",
                    "citations": [],
                    "confidence_score": -10.0,
                    "confidence_level": "NONE (ERROR)",
                }



        # 3. Build context & citations
        context_blocks = []
        citations = []

        for chunk, score in reranked:
            context_blocks.append(
                f"[Source: {chunk.source}, Page: {chunk.page_number}]\n{chunk.text}"
            )
            citations.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "score": score,
                "text": chunk.text,
            })


        context = "\n\n".join(context_blocks)

        # 4. Prompt construction & LLM Generation
        prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)
        
        logger.info("Generating LLM response...")
        try:
            answer = await self.llm.generate(
                prompt, system_instruction=SYSTEM_INSTRUCTION
            )
        except Exception as exc:
            logger.error("LLM Generation failed: %s", exc)
            return {
                "query": query,
                "answer": f"Error during generation: {exc}",
                "citations": citations,
                "confidence_score": float(reranked[0][1]) if reranked else -10.0,
                "confidence_level": "NONE (GENERATION ERROR)",
            }
 
        top_score = reranked[0][1] if reranked else -10.0
        if top_score >= 1.0:
            level = "HIGH"
        elif top_score >= -1.0:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "query": query,
            "answer": answer.strip(),
            "citations": citations,
            "confidence_score": float(top_score),
            "confidence_level": level,
        }
