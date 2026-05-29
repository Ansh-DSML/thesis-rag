"""
app/models/query.py

All Pydantic models for the query pipeline — request in, response out,
and every intermediate structured object between them.

Import map:
  QueryRequest   ← API layer receives this from the user
  RouterOutput   ← Prompt 1 (query router) returns this
  Citation       ← Built from ScoredChunk after answer generation
  QueryResponse  ← API layer returns this to the user
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

from app.models.chunk import QueryType, SourceType


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """
    Payload the user (or frontend) sends to POST /chat.

    session_id ties this request to a Redis chat memory key so the
    pipeline can inject the last k conversation turns into Prompt 4.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's natural-language question.",
    )
    session_id: str = Field(
        ...,
        description=(
            "Unique per-user session identifier. "
            "Used as Redis key for chat memory and query cache."
        ),
    )

    # Optional overrides — let the user narrow retrieval if they know what they want
    source_filter: Optional[SourceType] = Field(
        default=None,
        description="Restrict retrieval to 'thesis' or 'paper' chunks only.",
    )
    chapter_filter: Optional[int] = Field(
        default=None,
        ge=1,
        le=6,
        description="Restrict retrieval to a specific thesis chapter (1–6).",
    )
    use_graph: Optional[bool] = Field(
        default=None,
        description=(
            "Override KG lookup decision from router. "
            "None = let the router decide."
        ),
    )

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        return str(v).strip()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER OUTPUT  (Prompt 1 structured response)
# ══════════════════════════════════════════════════════════════════════════════

class RouterOutput(BaseModel):
    """
    Structured JSON output from Prompt 1 (Query Router LLM call).

    The router classifies the query type, extracts domain entities,
    and sets control flags that drive every downstream decision:
      - use_graph  → whether to run KG lookup (Step 3)
      - use_hyde   → whether to run HyDE expansion (Prompt 2)
      - source_filter / chapter_filter → metadata pre-filter for Qdrant

    LLM is instructed to return ONLY this JSON schema, no extra text.
    """

    query_type: QueryType = Field(
        ...,
        description=(
            "Query classification: factual | comparative | synthesis | "
            "methodology | equation | algorithm"
        ),
    )
    entities: list[str] = Field(
        default_factory=list,
        description=(
            "Domain entities extracted from the query. "
            "E.g. ['PSO', 'diabetic retinopathy', 'focal loss']. "
            "Used for KG lookup and metadata filtering."
        ),
    )
    use_graph: bool = Field(
        default=False,
        description=(
            "True if the query is entity-centric and KG lookup should run. "
            "False for broad synthesis or factual lookups."
        ),
    )
    use_hyde: bool = Field(
        default=True,
        description=(
            "True if HyDE expansion should run. "
            "Always False for equation_lookup (exact symbolic match needed)."
        ),
    )
    source_filter: Optional[SourceType] = Field(
        default=None,
        description=(
            "If the query clearly targets thesis or papers only, set this. "
            "None = search both."
        ),
    )
    chapter_filter: Optional[int] = Field(
        default=None,
        ge=1,
        le=6,
        description=(
            "If the query references a specific thesis chapter, set this. "
            "None = search all chapters."
        ),
    )

    @field_validator("use_hyde", mode="after")
    @classmethod
    def _hyde_off_for_equations(cls, v: bool, info) -> bool:
        """HyDE is always disabled for equation queries — guard at model level."""
        query_type = info.data.get("query_type")
        if query_type == "equation":
            return False
        return v

    @classmethod
    def fallback(cls, query: str) -> RouterOutput:
        """
        Safe default when the LLM returns malformed JSON or times out.
        Treats the query as factual with no filters — safe middle ground.
        """
        return cls(
            query_type="factual",
            entities=[],
            use_graph=False,
            use_hyde=True,
            source_filter=None,
            chapter_filter=None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CITATION
# ══════════════════════════════════════════════════════════════════════════════

class Citation(BaseModel):
    """
    A single source reference injected into the generated answer.

    Built from a ScoredChunk after the reranker selects the top-5 chunks.
    Returned to the user alongside the answer so they can verify claims.
    """

    chunk_id: str = Field(..., description="Stable chunk identifier from ingestion.")
    citation_key: str = Field(
        ...,
        description="Short key used inline in the answer: '[Chapter 3]' or '[Zhang 2022]'.",
    )
    source_type: SourceType
    display_source: str = Field(
        ...,
        description="Human-readable source: 'Chapter 3: Methodology' or paper title.",
    )

    # Thesis-specific
    chapter_num: Optional[int]  = None
    chapter_name: Optional[str] = None

    # Paper-specific
    paper_title: Optional[str]       = None
    paper_authors: list[str]         = Field(default_factory=list)
    paper_year: Optional[int]        = None
    paper_journal: Optional[str]     = None
    paper_doi: Optional[str]         = None

    # Retrieval context
    section_heading: Optional[str]  = None
    page: Optional[int]             = None
    relevance_score: float          = Field(
        default=0.0,
        description="Final reranker score — higher = more relevant.",
    )
    text_snippet: str = Field(
        default="",
        description="First 200 chars of the chunk — shown as evidence in UI.",
    )

    @classmethod
    def from_scored_chunk(cls, sc) -> Citation:
        """
        Build a Citation from a ScoredChunk (imported lazily to avoid circular import).
        sc: ScoredChunk instance from app.models.chunk
        """
        chunk = sc.chunk
        snippet = chunk.text[:200].rstrip()
        if len(chunk.text) > 200:
            snippet += "…"

        return cls(
            chunk_id=chunk.chunk_id,
            citation_key=chunk.citation_key,
            source_type=chunk.source_type,
            display_source=chunk.display_source,
            chapter_num=chunk.chapter_num,
            chapter_name=chunk.chapter_name,
            paper_title=chunk.paper_title,
            paper_authors=chunk.paper_authors,
            paper_year=chunk.paper_year,
            paper_journal=chunk.paper_journal,
            paper_doi=chunk.paper_doi,
            section_heading=chunk.section_heading,
            page=chunk.page_start,
            relevance_score=round(sc.score, 4),
            text_snippet=snippet,
        )


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

class QueryResponse(BaseModel):
    """
    Full response returned to the user from POST /chat.

    Contains the generated answer, all citations, timing information,
    and cache metadata. The frontend can use citations to build a
    source-reference panel alongside the answer.
    """

    # Core output
    answer: str = Field(..., description="Generated answer from Prompt 4.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Ordered list of sources used in the answer (top-5 after reranking).",
    )

    # Request echo — useful for async clients
    session_id: str
    query: str

    # Pipeline metadata
    query_type: QueryType = Field(
        default="factual",
        description="Query type classified by Prompt 1.",
    )
    cache_hit: bool = Field(
        default=False,
        description="True if this response was served from Redis cache.",
    )
    kg_used: bool = Field(
        default=False,
        description="True if the knowledge graph lookup contributed to retrieval.",
    )
    hyde_used: bool = Field(
        default=False,
        description="True if HyDE expansion was applied.",
    )
    compression_applied: bool = Field(
        default=False,
        description="True if LLMLingua-2 compressed the context before generation.",
    )

    # Performance
    latency_ms: float = Field(
        default=0.0,
        description="Total end-to-end latency in milliseconds.",
    )
    token_usage: int = Field(
        default=0,
        description="Approximate tokens consumed by the LLM call (Prompt 4).",
    )

    @property
    def citation_count(self) -> int:
        return len(self.citations)

    @property
    def has_thesis_sources(self) -> bool:
        return any(c.source_type == "thesis" for c in self.citations)

    @property
    def has_paper_sources(self) -> bool:
        return any(c.source_type == "paper" for c in self.citations)


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH RESPONSE  (POST /search — raw retrieval without generation)
# ══════════════════════════════════════════════════════════════════════════════

class SearchResult(BaseModel):
    """
    Returned by POST /search for debugging retrieval quality
    without paying for LLM generation.
    """
    chunk_id: str
    score: float
    source_type: SourceType
    display_source: str
    section_heading: Optional[str] = None
    text_snippet: str
    retrieval_source: str   # "vector" | "bm25" | "hybrid"
    kg_boosted: bool = False
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None


class SearchResponse(BaseModel):
    """Response for POST /search (retrieval-only, no generation)."""
    query: str
    results: list[SearchResult]
    latency_ms: float
    total_found: int
    query_type: Optional[QueryType] = None