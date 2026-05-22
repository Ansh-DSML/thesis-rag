"""
app/core/retrieval/hybrid_retriever.py

Steps 7A + 7B + 8 — Hybrid Retrieval with RRF Merge.

Step 7A — Vector search (Qdrant HNSW):
  Cosine similarity ANN search with metadata filter applied server-side.
  Returns top-20 ScoredChunks with full metadata payloads.

Step 7B — BM25 search (rank_bm25):
  Keyword search over the pre-built BM25 index loaded from bm25_index.pkl.
  Applies metadata filter manually post-retrieval (no server-side filtering).
  Runs in a thread pool (rank_bm25 is synchronous).

Both run in parallel via asyncio.gather — total latency = max(7A, 7B).

Step 8 — RRF merge:
  Merges the two result lists using Reciprocal Rank Fusion:
    score = 0.6 × 1/(60 + vector_rank) + 0.4 × 1/(60 + bm25_rank)
  Deduplicates by chunk_id. Returns up to 40 candidates sorted by RRF score.
  These pass forward to Step 9 (KG boost) and Step 10 (reranker).

BM25 index is loaded once at first retrieval and cached in memory.
Chunk metadata (for BM25 result enrichment) is also loaded at first retrieval.

Usage:
    from app.core.retrieval.hybrid_retriever import get_hybrid_retriever
    retriever = get_hybrid_retriever()
    chunks = await retriever.retrieve(
        query=query,
        query_vector=query_vector,
        router_output=router_output,
    )
"""

from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from app.config.settings import get_settings
from app.core.retrieval.metadata_filter import (
    apply_filter_to_chunk,
    build_qdrant_filter,
    describe_filter,
)
from app.models.chunk import ChunkMetadata, ScoredChunk
from app.models.query import RouterOutput
from app.utils.logger import get_logger, StageTimer

log      = get_logger(__name__)
settings = get_settings()


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP DATA — loaded once, cached in module-level variables
# ══════════════════════════════════════════════════════════════════════════════

_bm25_index       = None       # BM25Okapi instance
_bm25_chunk_ids: list[str] = []
_chunk_meta_store: dict[str, ChunkMetadata] = {}  # chunk_id → ChunkMetadata
_data_lock = asyncio.Lock()
_data_loaded = False


async def _load_retrieval_data() -> None:
    """
    Load BM25 index + chunk metadata on first retrieval.
    Protected by asyncio.Lock — only runs once even under concurrency.
    """
    global _bm25_index, _bm25_chunk_ids, _chunk_meta_store, _data_loaded

    async with _data_lock:
        if _data_loaded:
            return

        # ── Load BM25 index ───────────────────────────────────────────────────
        bm25_path = Path(settings.bm25_index_path)
        if not bm25_path.exists():
            log.error(
                "bm25_index_not_found",
                path=str(bm25_path),
                fix="Run: python scripts/run_ingestion.py",
            )
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}")

        log.info("loading_bm25_index", path=str(bm25_path))
        loop = asyncio.get_event_loop()
        index_data = await loop.run_in_executor(
            None,
            lambda: pickle.loads(bm25_path.read_bytes()),
        )
        _bm25_index     = index_data["bm25"]
        _bm25_chunk_ids = index_data["chunk_ids"]
        log.info("bm25_index_loaded", chunks=len(_bm25_chunk_ids))

        # ── Load chunk metadata for BM25 result enrichment ───────────────────
        # chunks_with_metadata.json is produced by run_ingestion.py Stage 8.
        # Used to build ChunkMetadata for BM25 hits (BM25 doesn't store metadata).
        chunks_json_path = Path(settings.bm25_index_path).parent / "chunks_with_metadata.json"
        if chunks_json_path.exists():
            log.info("loading_chunk_metadata", path=str(chunks_json_path))
            raw = await loop.run_in_executor(
                None,
                lambda: json.loads(chunks_json_path.read_text(encoding="utf-8")),
            )
            for item in raw:
                try:
                    meta = ChunkMetadata.model_validate(item)
                    _chunk_meta_store[meta.chunk_id] = meta
                except Exception as exc:
                    log.warning("chunk_metadata_parse_error", error=str(exc))
            log.info("chunk_metadata_loaded", count=len(_chunk_meta_store))
        else:
            log.warning(
                "chunks_metadata_not_found",
                path=str(chunks_json_path),
                note="BM25 results will be metadata-poor until this file exists.",
            )

        _data_loaded = True


# ══════════════════════════════════════════════════════════════════════════════
# QDRANT ASYNC CLIENT — singleton
# ══════════════════════════════════════════════════════════════════════════════

_qdrant_client = None


def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    try:
        from qdrant_client import AsyncQdrantClient
    except ImportError:
        log.error("qdrant_client_not_installed", fix="pip install qdrant-client")
        raise

    _qdrant_client = AsyncQdrantClient(
        url=settings.qdrant_url_with_port,
        api_key=settings.qdrant_api_key,
        timeout=30,
    )
    log.info("qdrant_async_client_created", url=settings.qdrant_url_with_port)
    return _qdrant_client


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Runs vector search and BM25 search in parallel, merges with RRF.
    Singleton — get via get_hybrid_retriever().
    """

    async def retrieve(
        self,
        query: str,
        query_vector: np.ndarray,
        router_output: RouterOutput,
    ) -> list[ScoredChunk]:
        """
        Full hybrid retrieval: vector + BM25 in parallel, RRF merge.

        Args:
            query:         Raw query string (for BM25 tokenisation).
            query_vector:  1024-dim float32 vector from HyDEExpander.
            router_output: RouterOutput for metadata filter construction.

        Returns:
            Up to 40 ScoredChunks sorted by RRF score descending.
            These pass to Step 9 (KG boost) and Step 10 (reranker).
        """
        await _load_retrieval_data()

        qdrant_filter  = build_qdrant_filter(router_output)
        filter_summary = describe_filter(router_output)
        log.info("retrieval_start", filter=filter_summary)

        with StageTimer("hybrid_retrieval", log, filter=filter_summary):
            # Steps 7A and 7B run in parallel
            vector_task = asyncio.create_task(
                self._vector_search(query_vector, qdrant_filter)
            )
            bm25_task = asyncio.create_task(
                self._bm25_search(query, router_output)
            )

            vector_results, bm25_results = await asyncio.gather(
                vector_task, bm25_task, return_exceptions=False
            )

        log.info(
            "retrieval_raw_results",
            vector_hits=len(vector_results),
            bm25_hits=len(bm25_results),
        )

        # Step 8 — RRF merge
        merged = self._rrf_merge(vector_results, bm25_results)

        log.info(
            "rrf_merge_complete",
            merged_count=len(merged),
            top_score=round(merged[0].score, 4) if merged else 0,
        )
        return merged

    # ── Step 7A: Vector search ────────────────────────────────────────────────

    async def _vector_search(
        self,
        query_vector: np.ndarray,
        qdrant_filter: Optional[object],
    ) -> list[ScoredChunk]:
        """
        Cosine similarity ANN search on Qdrant Cloud.
        Returns top-k ScoredChunks with full metadata payloads.
        """
        client = _get_qdrant_client()

        with StageTimer("vector_search", log):
            try:
                results = await client.search(
                    collection_name=settings.qdrant_collection_name,
                    query_vector=query_vector.tolist(),
                    query_filter=qdrant_filter,
                    limit=settings.qdrant_top_k,
                    with_payload=True,
                    with_vectors=False,   # don't return stored vectors (saves bandwidth)
                )
            except Exception as exc:
                log.error("vector_search_failed", error=str(exc))
                return []

        scored_chunks = []
        for i, hit in enumerate(results):
            try:
                sc = ScoredChunk.from_qdrant_result(
                    point_id=hit.id,
                    score=hit.score,
                    payload=hit.payload,
                )
                sc = sc.model_copy(update={"vector_rank": i + 1})
                scored_chunks.append(sc)
            except Exception as exc:
                log.warning(
                    "vector_result_parse_error",
                    point_id=hit.id,
                    error=str(exc),
                )

        log.debug("vector_search_done", hits=len(scored_chunks))
        return scored_chunks

    # ── Step 7B: BM25 search ──────────────────────────────────────────────────

    async def _bm25_search(
        self,
        query: str,
        router_output: RouterOutput,
    ) -> list[ScoredChunk]:
        """
        BM25 keyword search over the pre-built index.
        Runs in a thread pool (rank_bm25 is synchronous).
        Applies metadata filter post-retrieval.
        Returns top BM25_TOP_K ScoredChunks.
        """
        if _bm25_index is None:
            log.warning("bm25_index_not_loaded")
            return []

        with StageTimer("bm25_search", log):
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._bm25_score_and_filter(query, router_output),
            )

        log.debug("bm25_search_done", hits=len(results))
        return results

    def _bm25_score_and_filter(
        self,
        query: str,
        router_output: RouterOutput,
    ) -> list[ScoredChunk]:
        """
        Synchronous BM25 scoring + metadata post-filtering.
        Runs in thread pool so it doesn't block the event loop.
        """
        # Tokenise query (simple split — consistent with ingestion)
        query_tokens = query.lower().split()
        if not query_tokens:
            return []

        # Get BM25 scores for all corpus documents
        scores: np.ndarray = _bm25_index.get_scores(query_tokens)

        # Get top-K indices by score (before filtering)
        top_k_raw = min(settings.bm25_top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k_raw)[-top_k_raw:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[ScoredChunk] = []
        rank = 0

        for idx in top_indices:
            if scores[idx] <= 0.0:
                break  # remaining scores are zero — not worth including

            chunk_id = _bm25_chunk_ids[idx]
            meta     = _chunk_meta_store.get(chunk_id)

            if meta is None:
                # Chunk ID in BM25 index but not in metadata store
                # (can happen if ingestion was partially re-run)
                log.warning("bm25_chunk_meta_missing", chunk_id=chunk_id)
                continue

            # Apply metadata filter post-retrieval
            if not apply_filter_to_chunk(meta, router_output):
                continue

            rank += 1
            sc = ScoredChunk.from_bm25_result(
                chunk_metadata=meta,
                bm25_score=float(scores[idx]),
                rank=rank,
            )
            results.append(sc)

            if rank >= settings.qdrant_top_k:   # same top-k limit as vector
                break

        return results

    # ── Step 8: RRF merge ─────────────────────────────────────────────────────

    def _rrf_merge(
        self,
        vector_results: list[ScoredChunk],
        bm25_results: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        """
        Reciprocal Rank Fusion merge.

        Formula: score = α × 1/(k + vector_rank) + (1-α) × 1/(k + bm25_rank)
          α = settings.rrf_vector_weight = 0.6
          k = settings.rrf_k             = 60

        Deduplicates by chunk_id — if a chunk appears in both result sets,
        it gets contributions from both ranks (naturally boosted).

        Chunks appearing in only one result set still receive a valid score
        from that branch (the other branch contributes 0).

        Returns merged list sorted by RRF score descending.
        """
        alpha = settings.rrf_vector_weight
        beta  = settings.rrf_bm25_weight
        k     = settings.rrf_k

        # Map chunk_id → ScoredChunk (prefer vector result for metadata richness)
        chunk_store: dict[str, ScoredChunk] = {}
        rrf_scores:  dict[str, float]       = {}
        vector_ranks: dict[str, int]         = {}
        bm25_ranks:   dict[str, int]         = {}

        # Accumulate vector contributions
        for rank, sc in enumerate(vector_results, start=1):
            cid = sc.chunk.chunk_id
            chunk_store[cid]  = sc
            rrf_scores[cid]   = rrf_scores.get(cid, 0.0) + alpha * (1.0 / (k + rank))
            vector_ranks[cid] = rank

        # Accumulate BM25 contributions
        for rank, sc in enumerate(bm25_results, start=1):
            cid = sc.chunk.chunk_id
            if cid not in chunk_store:
                chunk_store[cid] = sc   # BM25-only chunk
            rrf_scores[cid]  = rrf_scores.get(cid, 0.0) + beta * (1.0 / (k + rank))
            bm25_ranks[cid]  = rank

        # Build final ScoredChunk list with RRF scores
        merged: list[ScoredChunk] = []
        for cid, sc in chunk_store.items():
            rrf_score = rrf_scores[cid]
            # Determine retrieval source
            has_vector = cid in vector_ranks
            has_bm25   = cid in bm25_ranks
            if has_vector and has_bm25:
                source = "hybrid"
            elif has_vector:
                source = "vector"
            else:
                source = "bm25"

            updated_sc = sc.model_copy(update={
                "score":            rrf_score,
                "rrf_score":        rrf_score,
                "vector_score":     sc.vector_score,
                "bm25_score":       sc.bm25_score,
                "vector_rank":      vector_ranks.get(cid),
                "bm25_rank":        bm25_ranks.get(cid),
                "retrieval_source": source,
            })
            merged.append(updated_sc)

        # Sort by RRF score descending
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_retriever_instance: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """
    Return the shared HybridRetriever singleton.

    Usage in pipeline.py:
        from app.core.retrieval.hybrid_retriever import get_hybrid_retriever
        retriever = get_hybrid_retriever()
        chunks = await retriever.retrieve(query, query_vector, router_output)
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance