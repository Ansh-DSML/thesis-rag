"""
app/core/retrieval/metadata_filter.py

Step 6 — Metadata Filter Construction.

Converts RouterOutput into a Qdrant Filter applied server-side before ANN search.

DESIGN DECISION — hard constraints only:
  Entity-based filtering has been intentionally removed from Qdrant pre-filtering.
  Reason: spaCy NER tags entities as ORG/PERSON/GPE etc — not domain terms like
  "PSO", "bio algorithms", "feature selection". Filtering by these causes false
  negatives (0 retrieval results) for the majority of domain queries.

  Entity relevance is handled DOWNSTREAM by:
    Step 9  — KG score boost (+0.15 for entity-linked chunks)
    Step 10 — Cross-encoder reranker (scores query+chunk pairs directly)

Hard constraints applied here (safe, no false negatives):
  source_type  → "thesis" | "paper"   (only when router is very confident)
  chapter_num  → 1–6                  (only when chapter explicitly referenced)
  chunk_type   → "equation"           (only for equation queries)

Returns None when no constraints apply — Qdrant searches entire collection.
"""

from __future__ import annotations

from typing import Optional

from app.models.chunk import ChunkMetadata
from app.models.query import RouterOutput
from app.utils.logger import get_logger

log = get_logger(__name__)


def build_qdrant_filter(router_output: RouterOutput) -> Optional[object]:
    """
    Build a Qdrant Filter from RouterOutput using hard constraints only.

    Returns None if no constraints apply (search entire collection).
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
    except ImportError:
        log.error("qdrant_client_not_installed", fix="pip install qdrant-client")
        raise

    must_conditions = []

    # ── 1. Source type — only apply when router is explicit ──────────────────
    if router_output.source_filter:
        must_conditions.append(
            FieldCondition(
                key="source_type",
                match=MatchValue(value=router_output.source_filter),
            )
        )
        log.debug("filter_source_type", value=router_output.source_filter)

    # ── 2. Chapter — only when a specific chapter is referenced ──────────────
    if router_output.chapter_filter is not None:
        must_conditions.append(
            FieldCondition(
                key="chapter_num",
                match=MatchValue(value=router_output.chapter_filter),
            )
        )
        log.debug("filter_chapter", value=router_output.chapter_filter)

    # ── 3. Chunk type — equation queries only retrieve equation chunks ────────
    if router_output.query_type == "equation":
        must_conditions.append(
            FieldCondition(
                key="chunk_type",
                match=MatchValue(value="equation"),
            )
        )
        log.debug("filter_chunk_type", value="equation")

    # NOTE: No entity-based filtering.
    # Entity relevance is handled by KG boost (Step 9) and reranker (Step 10).
    # Pre-filtering by entities causes false negatives for most domain queries.

    if not must_conditions:
        log.debug("no_metadata_filter_applied")
        return None

    return Filter(must=must_conditions)


def apply_filter_to_chunk(
    chunk: ChunkMetadata,
    router_output: RouterOutput,
) -> bool:
    """
    Apply the same hard-constraint logic to a ChunkMetadata for BM25 post-filtering.

    Returns True = keep, False = discard.
    """
    # Source type
    if router_output.source_filter:
        if chunk.source_type != router_output.source_filter:
            return False

    # Chapter
    if router_output.chapter_filter is not None:
        if chunk.chapter_num != router_output.chapter_filter:
            return False

    # Chunk type for equation queries
    if router_output.query_type == "equation":
        if chunk.chunk_type != "equation":
            return False

    return True


def describe_filter(router_output: RouterOutput) -> str:
    """Human-readable filter summary for logging."""
    parts = []
    if router_output.source_filter:
        parts.append(f"source={router_output.source_filter}")
    if router_output.chapter_filter:
        parts.append(f"chapter={router_output.chapter_filter}")
    if router_output.query_type == "equation":
        parts.append("chunk_type=equation")
    return ", ".join(parts) if parts else "none"