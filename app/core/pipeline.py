"""
app/core/pipeline.py

RAGPipeline — wires all 14 pipeline steps end to end.

Step  1 : Redis cache check
Step  2 : Query router (Prompt 1)
Step  3 : Knowledge graph lookup (if use_graph)
Step  4 : HyDE expansion (Prompt 2, skipped for equation)
Step  5 : Query embedding (BGE-large + optional HyDE average)
Step  6 : Metadata filter construction
Step  7A: Vector search (Qdrant HNSW) ─┐ parallel
Step  7B: BM25 search (rank_bm25)      ─┘
Step  8 : RRF merge
Step  9 : KG score boost (if use_graph)
Step 10 : Cross-encoder reranking (BGE-Reranker-Large)
Step 11 : Context compression (LLMLingua-2, if > 3000 tokens)
Step 12 : Chat memory injection
Step 13 : Answer generation (Prompt 4, Gemini)
Step 14 : Cache write + memory save + return QueryResponse

Usage:
    pipeline = RAGPipeline()
    response = await pipeline.run(request)          # full pipeline
    results  = await pipeline.retrieve_only(request) # steps 1–10 only (for /search)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from app.config.settings import get_settings
from app.core.cache.redis_cache import get_cache
from app.core.generation.answer_generator import AnswerGenerator
from app.core.retrieval.compressor import get_compressor
from app.core.retrieval.hybrid_retriever import get_hybrid_retriever
from app.core.retrieval.hyde import get_hyde_expander
from app.core.retrieval.metadata_filter import build_qdrant_filter
from app.core.retrieval.query_router import get_query_router
from app.core.retrieval.reranker import get_reranker
from app.models.chunk import ScoredChunk
from app.models.query import QueryRequest, QueryResponse, RouterOutput, SearchResult, SearchResponse
from app.utils.logger import get_logger, StageTimer, set_trace_id, bind_request_context

log      = get_logger(__name__)
settings = get_settings()


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH LOADER (lazy singleton)
# ══════════════════════════════════════════════════════════════════════════════

_kg_graph      = None
_kg_loaded     = False


def _load_kg_graph():
    """Load the NetworkX KG from graph.json. Returns None if not found."""
    global _kg_graph, _kg_loaded
    if _kg_loaded:
        return _kg_graph

    kg_path = Path(settings.kg_output_path)
    if not kg_path.exists():
        log.warning(
            "kg_graph_not_found",
            path=str(kg_path),
            note="KG lookup disabled. Run: python scripts/build_knowledge_graph.py",
        )
        _kg_loaded = True
        return None

    try:
        import networkx as nx
        data = json.loads(kg_path.read_text(encoding="utf-8"))
        _kg_graph = nx.node_link_graph(data, directed=True)
        log.info(
            "kg_graph_loaded",
            nodes=_kg_graph.number_of_nodes(),
            edges=_kg_graph.number_of_edges(),
        )
    except Exception as exc:
        log.error("kg_graph_load_failed", error=str(exc))
        _kg_graph = None

    _kg_loaded = True
    return _kg_graph


# ══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Orchestrates all 14 pipeline steps.
    Singleton — get via get_pipeline() in dependencies.py.
    All component clients are singletons so no extra cost per request.
    """

    def __init__(self) -> None:
        self._router     = get_query_router()
        self._expander   = get_hyde_expander()
        self._retriever  = get_hybrid_retriever()
        self._reranker   = get_reranker()
        self._compressor = get_compressor()
        self._generator  = AnswerGenerator()

    # ══════════════════════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    async def run(self, request: QueryRequest) -> QueryResponse:
        """
        Execute the full 14-step RAG pipeline and return a QueryResponse.

        Every step is logged with timing. Any unhandled exception bubbles up
        to the FastAPI exception handler — the pipeline does not swallow errors.
        """
        import uuid
        trace_id = set_trace_id()
        bind_request_context(
            trace_id=trace_id,
            session_id=request.session_id,
            query=request.query,
        )
        total_start = time.perf_counter()

        log.info(
            "pipeline_start",
            query_preview=request.query[:80],
            session_id=request.session_id,
        )

        # ── Step 0: Static cache check (pre-warmed suggested queries) ────────
        cache = await get_cache()
        static_data = await cache.get_static_query(request.query)
        if static_data:
            response = QueryResponse.model_validate(static_data)
            response = response.model_copy(update={
                "cache_hit": True,
                "session_id": request.session_id,   # patch to match current session
                "query": request.query,
                "latency_ms": round((time.perf_counter() - total_start) * 1000, 2),
            })
            log.info(
                "pipeline_static_cache_hit",
                latency_ms=round((time.perf_counter() - total_start) * 1000, 2),
                query_preview=request.query[:40],
            )
            return response

        # ── Step 1: Redis cache check (session-bound) ─────────────────────────
        cached_data = await cache.get_query(request.query, request.session_id)
        if cached_data:
            response = QueryResponse.model_validate(cached_data)
            response = response.model_copy(update={"cache_hit": True})
            log.info(
                "pipeline_cache_hit",
                latency_ms=round((time.perf_counter() - total_start) * 1000, 2),
            )
            return response

        # ── Step 2: Query router ──────────────────────────────────────────────
        router_output: RouterOutput = await self._router.route(
            query=request.query,
            session_id=request.session_id,
        )
        # Apply any user-level overrides from the request
        router_output = self._router.apply_request_overrides(
            router_output=router_output,
            source_filter=request.source_filter,
            chapter_filter=request.chapter_filter,
            use_graph=request.use_graph,
        )

        # Force HyDE for all non-equation queries
        if router_output.query_type != "equation":
            router_output = router_output.model_copy(update={"use_hyde": True})

        # Force KG lookup whenever entities extracted
        if router_output.entities:
            router_output = router_output.model_copy(update={"use_graph": True})

        # ── Step 3: Knowledge graph lookup ────────────────────────────────────
        kg_chunk_ids: set[str] = set()
        if router_output.use_graph and settings.use_knowledge_graph:
            kg_chunk_ids = self._kg_lookup(router_output.entities)
            log.info("kg_lookup_complete", matched_chunks=len(kg_chunk_ids))

        # ── Steps 4 + 5: HyDE expansion + query embedding ────────────────────
        query_vector = await self._expander.get_query_vector(
            query=request.query,
            query_type=router_output.query_type,
            use_hyde=router_output.use_hyde,
        )

        # ── Step 6: Metadata filter ───────────────────────────────────────────
        metadata_filter = build_qdrant_filter(router_output)

        # ── Steps 7A + 7B + 8: Hybrid retrieval + RRF ────────────────────────
        rrf_chunks: list[ScoredChunk] = await self._retriever.retrieve(
            query=request.query,
            query_vector=query_vector,
            router_output=router_output,
        )

        if not rrf_chunks:
            log.warning("pipeline_no_chunks_retrieved", query=request.query[:60])
            return self._empty_response(request, router_output, total_start)

        # ── Step 9: KG score boost ────────────────────────────────────────────
        if kg_chunk_ids:
            rrf_chunks = self._apply_kg_boost(rrf_chunks, kg_chunk_ids)

        # ── Step 10: Cross-encoder reranking ──────────────────────────────────
        top_chunks: list[ScoredChunk] = await self._reranker.rerank(
            query=request.query,
            chunks=rrf_chunks,
            query_type=router_output.query_type,
        )

        # ── Step 11: Context compression ──────────────────────────────────────
        compressed_context, was_compressed = await self._compressor.compress(
            chunks=top_chunks,
            query=request.query,
            query_type=router_output.query_type,
        )

        # ── Steps 12 + 13: Memory injection + answer generation ───────────────
        answer, citations, token_count = await self._generator.generate(
            query=request.query,
            query_type=router_output.query_type,
            chunks=top_chunks,
            session_id=request.session_id,
            router_output=router_output,
            compressed_context=compressed_context if was_compressed else None,
        )

        # ── Build QueryResponse ───────────────────────────────────────────────
        latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
        response = QueryResponse(
            answer=answer,
            citations=citations,
            session_id=request.session_id,
            query=request.query,
            query_type=router_output.query_type,
            cache_hit=False,
            kg_used=bool(kg_chunk_ids),
            hyde_used=router_output.use_hyde,
            compression_applied=was_compressed,
            latency_ms=latency_ms,
            token_usage=token_count,
        )

        # ── Step 14: Cache write + memory save ────────────────────────────────
        await cache.set_query(
            query=request.query,
            session_id=request.session_id,
            response=response.model_dump(mode="json"),
        )
        await self._generator.save_turn(
            session_id=request.session_id,
            query=request.query,
            answer=answer,
        )

        log.info(
            "pipeline_complete",
            latency_ms=latency_ms,
            citations=len(citations),
            compressed=was_compressed,
            kg_used=bool(kg_chunk_ids),
            tokens=token_count,
        )
        return response

    # ══════════════════════════════════════════════════════════════════════════
    # RETRIEVAL-ONLY  (for POST /search — no generation)
    # ══════════════════════════════════════════════════════════════════════════

    async def retrieve_only(self, request: QueryRequest) -> SearchResponse:
        """
        Execute steps 1–10 only (no generation, no cache write).
        Used by POST /search for debugging retrieval quality.
        """
        start = time.perf_counter()

        router_output = await self._router.route(request.query, request.session_id)
        router_output = self._router.apply_request_overrides(
            router_output, request.source_filter, request.chapter_filter, request.use_graph
        )

        # Force HyDE for all non-equation queries
        if router_output.query_type != "equation":
            router_output = router_output.model_copy(update={"use_hyde": True})

        # Force KG lookup whenever entities extracted
        if router_output.entities:
            router_output = router_output.model_copy(update={"use_graph": True})

        query_vector = await self._expander.get_query_vector(
            query=request.query,
            query_type=router_output.query_type,
            use_hyde=router_output.use_hyde,
        )

        rrf_chunks = await self._retriever.retrieve(
            query=request.query,
            query_vector=query_vector,
            router_output=router_output,
        )

        if router_output.use_graph:
            kg_chunk_ids = self._kg_lookup(router_output.entities)
            if kg_chunk_ids:
                rrf_chunks = self._apply_kg_boost(rrf_chunks, kg_chunk_ids)

        top_chunks = await self._reranker.rerank(
            query=request.query,
            chunks=rrf_chunks,
            query_type=router_output.query_type,
        )

        results = [
            SearchResult(
                chunk_id=sc.chunk.chunk_id,
                score=round(sc.score, 4),
                source_type=sc.chunk.source_type,
                display_source=sc.chunk.display_source,
                section_heading=sc.chunk.section_heading,
                text_snippet=sc.chunk.text[:300] + ("…" if len(sc.chunk.text) > 300 else ""),
                retrieval_source=sc.retrieval_source,
                kg_boosted=sc.kg_boosted,
                vector_score=round(sc.vector_score, 4) if sc.vector_score else None,
                bm25_score=round(sc.bm25_score, 4) if sc.bm25_score else None,
                rerank_score=round(sc.rerank_score, 4) if sc.rerank_score else None,
            )
            for sc in top_chunks
        ]

        return SearchResponse(
            query=request.query,
            results=results,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            total_found=len(results),
            query_type=router_output.query_type,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _kg_lookup(self, entities: list[str]) -> set[str]:
        """
        Traverse the NetworkX KG to find chunk_ids linked to the given entities.
        Returns an empty set if the KG is not loaded or no matches found.
        """
        graph = _load_kg_graph()
        if graph is None or not entities:
            return set()

        chunk_ids: set[str] = set()
        for entity in entities:
            # Try exact match and normalised (lowercase) match
            for node_key in (f"entity::{entity}", f"entity::{entity.lower()}"):
                if not graph.has_node(node_key):
                    continue
                for _, target, data in graph.out_edges(node_key, data=True):
                    if data.get("relation") == "APPEARS_IN" and target.startswith("chunk::"):
                        chunk_ids.add(target.replace("chunk::", ""))

        log.debug("kg_lookup", entities=entities, matched=len(chunk_ids))
        return chunk_ids

    def _apply_kg_boost(
        self,
        chunks: list[ScoredChunk],
        kg_chunk_ids: set[str],
    ) -> list[ScoredChunk]:
        """
        Add KG boost score to chunks whose chunk_id appears in the KG result.
        Re-sorts the list by updated score. Returns the re-sorted list.
        """
        boosted = []
        for sc in chunks:
            if sc.chunk.chunk_id in kg_chunk_ids:
                sc = sc.apply_kg_boost(settings.kg_boost_score)
            boosted.append(sc)

        boosted.sort(key=lambda x: x.score, reverse=True)
        kg_boosted_count = sum(1 for sc in boosted if sc.kg_boosted)
        log.debug("kg_boost_applied", boosted_chunks=kg_boosted_count)
        return boosted

    def _empty_response(
        self,
        request: QueryRequest,
        router_output: RouterOutput,
        start: float,
    ) -> QueryResponse:
        """Return a graceful empty response when no chunks are retrieved."""
        return QueryResponse(
            answer=(
                "I could not find relevant content in the thesis or research "
                "papers to answer this question. Please try rephrasing."
            ),
            citations=[],
            session_id=request.session_id,
            query=request.query,
            query_type=router_output.query_type,
            cache_hit=False,
            kg_used=False,
            hyde_used=router_output.use_hyde,        # ← ADD THIS
            compression_applied=False,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Return the shared RAGPipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance