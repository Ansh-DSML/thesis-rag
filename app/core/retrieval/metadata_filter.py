"""
app/core/retrieval/metadata_filter.py

Step 6 — Metadata Filter Construction.

Converts RouterOutput into a Qdrant Filter object applied server-side
before any vector math. This narrows the ANN search space to only
relevant chunks — faster and more precise than post-filtering.

Also provides apply_filter_to_chunk() for BM25 post-retrieval filtering,
since BM25 has no server-side filter capability.

Filter logic:
  source_type   → "thesis" | "paper"      (from router source_filter)
  chapter_num   → 1–6                     (from router chapter_filter)
  chunk_type    → "equation"              (only for equation query type)
  entities      → MatchAny across         (only for specific query types
                  entities field           with named entities present)

For synthesis and comparative queries: NO entity filter is applied
(too broad — would miss relevant context). KG boost (step 9) handles
entity relevance for these types.

Returns None when no filters apply → Qdrant searches entire collection.
"""

from __future__ import annotations

from typing import Optional

from app.models.chunk import ChunkMetadata
from app.models.query import RouterOutput, QueryType
from app.utils.logger import get_logger

log = get_logger(__name__)


# ── Query types where entity-based Qdrant filtering is appropriate ────────────
# Synthesis and comparative are excluded — they need broad retrieval.
_ENTITY_FILTER_TYPES = {"factual", "methodology", "algorithm", "equation"}


def build_qdrant_filter(
    router_output: RouterOutput,
) -> Optional[object]:
    """
    Build a Qdrant Filter from RouterOutput.

    Applied server-side before ANN search in Qdrant — reduces the search
    space before any vector math runs.

    Returns None if no constraints apply (search entire collection).

    Args:
        router_output: Full RouterOutput from the query router.

    Returns:
        qdrant_client.models.Filter instance, or None.
    """
    try:
        from qdrant_client.models import (
            Filter,
            FieldCondition,
            MatchValue,
            MatchAny,
        )
    except ImportError:
        log.error("qdrant_client_not_installed", fix="pip install qdrant-client")
        raise

    must_conditions = []

    # ── 1. Source type filter (thesis vs paper) ───────────────────────────────
    if router_output.source_filter:
        must_conditions.append(
            FieldCondition(
                key="source_type",
                match=MatchValue(value=router_output.source_filter),
            )
        )
        log.debug("filter_source_type", value=router_output.source_filter)

    # ── 2. Chapter filter (thesis chapters 1–6) ───────────────────────────────
    if router_output.chapter_filter is not None:
        must_conditions.append(
            FieldCondition(
                key="chapter_num",
                match=MatchValue(value=router_output.chapter_filter),
            )
        )
        log.debug("filter_chapter", value=router_output.chapter_filter)

    # ── 3. Chunk type filter (equation queries → equation chunks only) ─────────
    if router_output.query_type == "equation":
        must_conditions.append(
            FieldCondition(
                key="chunk_type",
                match=MatchValue(value="equation"),
            )
        )
        log.debug("filter_chunk_type", value="equation")

    # ── 4. Entity filter (factual / methodology / algorithm queries only) ──────
    # Applied as a MUST condition: at least one entity must appear in the
    # chunk's entity list. Skipped for synthesis/comparative to avoid
    # over-filtering broad queries.
    if (
        router_output.entities
        and router_output.query_type in _ENTITY_FILTER_TYPES
        and router_output.query_type != "equation"  # already filtered by chunk_type
    ):
        # Normalise entities: lowercase for broader matching
        normalised = [e.lower() for e in router_output.entities if len(e) > 1]
        original   = [e for e in router_output.entities if len(e) > 1]
        all_forms  = list(set(normalised + original))

        if all_forms:
            must_conditions.append(
                FieldCondition(
                    key="entities",
                    match=MatchAny(any=all_forms),
                )
            )
            log.debug("filter_entities", count=len(all_forms), sample=all_forms[:3])

    # ── Return filter or None ─────────────────────────────────────────────────
    if not must_conditions:
        log.debug("no_metadata_filter_applied")
        return None

    return Filter(must=must_conditions)


def apply_filter_to_chunk(
    chunk: ChunkMetadata,
    router_output: RouterOutput,
) -> bool:
    """
    Apply the same filter logic to a single ChunkMetadata object.

    Used for BM25 post-retrieval filtering — BM25 has no server-side
    filter so we filter results manually after retrieval.

    Returns True if the chunk passes all filter conditions (should be kept).
    Returns False if any hard condition fails (should be dropped).

    Args:
        chunk:         ChunkMetadata of the BM25 result to evaluate.
        router_output: RouterOutput containing filter criteria.

    Returns:
        bool — True = keep, False = discard.
    """
    # ── 1. Source type ────────────────────────────────────────────────────────
    if router_output.source_filter:
        if chunk.source_type != router_output.source_filter:
            return False

    # ── 2. Chapter filter ─────────────────────────────────────────────────────
    if router_output.chapter_filter is not None:
        if chunk.chapter_num != router_output.chapter_filter:
            return False

    # ── 3. Chunk type (equation queries) ──────────────────────────────────────
    if router_output.query_type == "equation":
        if chunk.chunk_type != "equation":
            return False

    # ── 4. Entity filter ──────────────────────────────────────────────────────
    if (
        router_output.entities
        and router_output.query_type in _ENTITY_FILTER_TYPES
        and router_output.query_type != "equation"
    ):
        original   = {e for e in router_output.entities if len(e) > 1}
        normalised = {e.lower() for e in original}
        all_forms  = original | normalised

        chunk_entities_lower = {e.lower() for e in chunk.entities}
        chunk_entities_orig  = set(chunk.entities)
        chunk_all = chunk_entities_orig | chunk_entities_lower

        # At least one router entity must appear in the chunk's entity list
        if not (all_forms & chunk_all):
            return False

    return True


def describe_filter(router_output: RouterOutput) -> str:
    """
    Return a human-readable description of the active filters.
    Used in logging for debugging retrieval.
    """
    parts = []
    if router_output.source_filter:
        parts.append(f"source={router_output.source_filter}")
    if router_output.chapter_filter:
        parts.append(f"chapter={router_output.chapter_filter}")
    if router_output.query_type == "equation":
        parts.append("chunk_type=equation")
    if router_output.entities and router_output.query_type in _ENTITY_FILTER_TYPES:
        parts.append(f"entities={router_output.entities[:3]}")
    return ", ".join(parts) if parts else "none"