"""
app/core/retrieval/reranker.py

Step 5 — Cross-Encoder Reranking.

BGE-Reranker-Large scores every (query, chunk_text) pair independently.
More accurate than bi-encoder cosine similarity because it processes the
query and chunk text together in one forward pass — attending to both.

Input:  query string + top-20 ScoredChunks from RRF merge (Step 8 / KG boost Step 9)
Output: top-5 ScoredChunks with rerank_score set, sorted by score descending

Equation boost:
  When query_type == "equation", equation chunks receive a +2.0 score bonus
  applied BEFORE top-5 selection. This ensures equation chunks always surface
  to the top for equation queries — the cross-encoder alone may rank them lower
  because equation text (LaTeX) is hard for the model to parse semantically.

CrossEncoder is synchronous — runs in a thread pool executor so it doesn't
block the async event loop. Model is loaded once and cached as a singleton.

Model: BAAI/bge-reranker-large (~1.3GB, must be downloaded from HuggingFace)
"""

from __future__ import annotations

import asyncio
from typing import Optional

from app.config.settings import get_settings
from app.models.chunk import ScoredChunk
from app.models.query import QueryType
from app.utils.logger import get_logger, StageTimer

log      = get_logger(__name__)
settings = get_settings()


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-ENCODER SINGLETON
# Heavy model — loaded once, never reloaded.
# ══════════════════════════════════════════════════════════════════════════════

_cross_encoder   = None
_reranker_lock   = asyncio.Lock()


async def _get_cross_encoder():
    """
    Lazily load the BGE cross-encoder reranker model.
    Thread-safe: uses asyncio.Lock with double-check pattern.
    First call downloads the model (~1.3GB) if not cached.
    """
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    async with _reranker_lock:
        if _cross_encoder is not None:
            return _cross_encoder

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            log.error(
                "sentence_transformers_not_installed",
                fix="pip install sentence-transformers",
            )
            raise

        log.info(
            "loading_reranker_model",
            model=settings.reranker_model,
            device=settings.embedding_device,
        )

        loop = asyncio.get_event_loop()
        _cross_encoder = await loop.run_in_executor(
            None,
            lambda: CrossEncoder(
                settings.reranker_model,
                device=settings.embedding_device,
                max_length=512,       # BGE-Reranker-Large max context
                cache_dir=settings.hf_home,
            ),
        )

        log.info("reranker_model_loaded", model=settings.reranker_model)

    return _cross_encoder


# ══════════════════════════════════════════════════════════════════════════════
# RERANKER
# ══════════════════════════════════════════════════════════════════════════════

class Reranker:
    """
    Cross-encoder reranker.
    Singleton — get via get_reranker().
    """

    async def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        query_type: QueryType,
    ) -> list[ScoredChunk]:
        """
        Score all (query, chunk_text) pairs and return top-k reranked.

        Steps:
          1. Build (query, text) pairs for every chunk
          2. Run CrossEncoder.predict() in thread pool
          3. Apply equation boost (+2.0) if query_type == "equation"
          4. Sort by final score descending
          5. Return top reranker_top_k_output chunks (default 5)

        Args:
            query:      The user's original question (not prefixed).
            chunks:     Top-k ScoredChunks from RRF / KG boost (default 20).
            query_type: From RouterOutput — triggers equation boost.

        Returns:
            Top-5 ScoredChunks with rerank_score and score updated.
        """
        if not chunks:
            return []

        with StageTimer("reranker", log, input_chunks=len(chunks), query_type=query_type):

            # ── Step 1: Build (query, text) pairs ────────────────────────────
            pairs = [(query, sc.chunk.text) for sc in chunks]

            # ── Step 2: Score pairs in thread pool ────────────────────────────
            cross_encoder = await _get_cross_encoder()
            loop = asyncio.get_event_loop()

            raw_scores: list[float] = await loop.run_in_executor(
                None,
                lambda: cross_encoder.predict(
                    pairs,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).tolist(),
            )

            # ── Step 3: Apply equation boost ──────────────────────────────────
            is_equation_query = (query_type == "equation")
            boosted_scores: list[float] = []

            for i, sc in enumerate(chunks):
                score = raw_scores[i]
                if is_equation_query and sc.chunk.chunk_type == "equation":
                    score += settings.reranker_equation_boost
                    log.debug(
                        "equation_boost_applied",
                        chunk_id=sc.chunk.chunk_id,
                        raw_score=round(raw_scores[i], 4),
                        boosted_score=round(score, 4),
                    )
                boosted_scores.append(score)

            # ── Step 4: Sort by boosted score descending ──────────────────────
            scored_pairs = sorted(
                zip(chunks, boosted_scores),
                key=lambda x: x[1],
                reverse=True,
            )

            # ── Step 5: Take top-k and update ScoredChunk scores ──────────────
            top_k   = settings.reranker_top_k_output
            results = []

            for sc, rerank_score in scored_pairs[:top_k]:
                updated = sc.apply_rerank_score(rerank_score)
                results.append(updated)

        log.info(
            "reranking_complete",
            input_chunks=len(chunks),
            output_chunks=len(results),
            top_score=round(results[0].score, 4) if results else 0,
            query_type=query_type,
            equation_boost_active=is_equation_query,
        )

        return results


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_reranker_instance: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """
    Return the shared Reranker singleton.

    Usage in pipeline.py:
        from app.core.retrieval.reranker import get_reranker
        reranker = get_reranker()
        top5 = await reranker.rerank(query, top20_chunks, query_type)
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance