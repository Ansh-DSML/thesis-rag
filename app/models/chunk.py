"""
app/models/chunk.py

Pydantic models for chunks throughout the retrieval pipeline.

ChunkMetadata  — Full metadata schema. Must exactly match the Qdrant payload
                 structure written by run_ingestion.py. Any field added here
                 must also be present in the Qdrant payload (or be Optional).

ScoredChunk    — Wraps ChunkMetadata with retrieval scores. This is the
                 object that travels through every retrieval stage:
                 vector search → BM25 → RRF merge → KG boost → reranker → compressor

Both models are immutable (frozen=False but use model_copy for updates)
to make score accumulation explicit and traceable.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import field_validator

from pydantic import BaseModel, Field, computed_field, model_validator


# ── Query types (defined here so chunk.py has no circular import) ─────────────
QueryType = Literal[
    "factual",       # Specific fact lookup — e.g. "What is the accuracy of X?"
    "comparative",   # Comparing two things — e.g. "PSO vs GA for feature selection"
    "synthesis",     # Broad topic summary — e.g. "Summarise findings on DR detection"
    "methodology",   # How something was done — e.g. "Explain the preprocessing pipeline"
    "equation",      # Mathematical content — e.g. "What is the loss function formula?"
    "algorithm",     # Algorithmic content — e.g. "Describe the HyDE retrieval steps"
]

# ── Chunk type ────────────────────────────────────────────────────────────────
ChunkType = Literal["text", "equation", "table"]

# ── Source type ───────────────────────────────────────────────────────────────
SourceType = Literal["thesis", "paper"]


class ChunkMetadata(BaseModel):
    """
    Full metadata for a single chunk.

    Ingestion writes this into Qdrant as a payload dict.
    Retrieval reads it back via from_qdrant_payload().

    Field names here must match the keys in the Qdrant payload exactly.
    If you rename a field, re-run ingestion or write a migration.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    chunk_id: str = Field(..., description="SHA1 of chunk text — stable identifier")
    doc_id: str   = Field(..., description="SHA1 of source filename")

    # ── Content ───────────────────────────────────────────────────────────────
    text: str
    chunk_type: ChunkType = Field(
        default="text",
        description="text | equation | table. Equations and tables skip LLMLingua-2.",
    )
    source_type: SourceType

    # ── Thesis-specific ───────────────────────────────────────────────────────
    chapter_num: Optional[int]      = None   # 1–6
    chapter_name: Optional[str]     = None   # "Introduction", "Methodology", etc.
    chapter_filename: Optional[str] = None   # "Chapter1_Introduction.pdf"

    # ── Paper-specific ────────────────────────────────────────────────────────
    paper_title: Optional[str]          = None
    paper_authors: list[str]            = Field(default_factory=list)
    paper_year: Optional[int]           = None
    paper_journal: Optional[str]        = None
    paper_doi: Optional[str]            = None
    paper_abstract: Optional[str]       = None
    paper_filename: Optional[str]       = None

    # ── Structural ────────────────────────────────────────────────────────────
    section_heading: Optional[str]  = None
    page_start: Optional[int]       = None
    token_count: int                = 0
    word_count: int                 = 0

    # ── Entities extracted by spaCy ───────────────────────────────────────────
    entities: list[str]              = Field(default_factory=list)
    entity_labels: dict[str, str]    = Field(
        default_factory=dict,
        description="entity_text → spaCy label (ORG, PERSON, GPE, …)",
    )

    # ── Domain-specific filter fields ─────────────────────────────────────────
    # Used by metadata_filter.py to build pre-retrieval Qdrant filters.
    # Populated from entities during ingestion (or left empty if not extracted).
    diseases: list[str]       = Field(
        default_factory=list,
        description="Disease/condition names found in this chunk.",
    )
    algorithms: list[str]     = Field(
        default_factory=list,
        description="ML / CS algorithm names found in this chunk.",
    )
    loss_functions: list[str] = Field(
        default_factory=list,
        description="Loss function names (MSE, focal loss, etc.) in this chunk.",
    )

    # ── Computed helpers (not stored in Qdrant) ───────────────────────────────

    @field_validator("paper_authors", mode="before")
    @classmethod
    def _coerce_paper_authors(cls, v):
        """Convert None to empty list — Grobid returns None when authors not found."""
        if v is None:
            return []
        return v

    @computed_field
    @property
    def display_source(self) -> str:
        """Human-readable source label used in answer citations."""
        if self.source_type == "thesis":
            ch = f"Chapter {self.chapter_num}" if self.chapter_num else "Thesis"
            name = f": {self.chapter_name}" if self.chapter_name else ""
            return f"{ch}{name}"
        return self.paper_title or self.paper_filename or "Unknown paper"

    @computed_field
    @property
    def citation_key(self) -> str:
        """
        Short citation key injected into generated answers.
        Thesis:  [Chapter 3]
        Paper:   [Zhang 2022]  (last name of first author + year)
        """
        if self.source_type == "thesis":
            return f"[Chapter {self.chapter_num}]"
        if self.paper_authors and self.paper_year:
            last_name = self.paper_authors[0].strip().split()[-1]
            return f"[{last_name} {self.paper_year}]"
        if self.paper_title:
            short = self.paper_title[:30].rstrip()
            return f"[{short}]"
        return "[Unknown]"

    @computed_field
    @property
    def is_equation(self) -> bool:
        return self.chunk_type == "equation"

    @computed_field
    @property
    def is_table(self) -> bool:
        return self.chunk_type == "table"

    @computed_field
    @property
    def should_skip_compression(self) -> bool:
        """
        Equations and tables are always bypassed by LLMLingua-2 and
        re-injected verbatim at the front of the compressed context.
        """
        return self.chunk_type in ("equation", "table")

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_qdrant_payload(cls, payload: dict) -> ChunkMetadata:
        """
        Deserialize a raw Qdrant point payload dict into ChunkMetadata.

        Filters out None values so Optional fields keep their defaults.
        Unknown keys are silently dropped (extra='ignore' via model_config).
        """
        clean = {k: v for k, v in payload.items() if v is not None}
        return cls.model_validate(clean)

    def to_qdrant_payload(self) -> dict:
        """
        Serialize to a flat dict for Qdrant upsert.
        Excludes computed_fields (they are not stored — re-computed on read).
        """
        return self.model_dump(
            exclude={"display_source", "citation_key", "is_equation",
                     "is_table", "should_skip_compression"},
        )

    model_config = {"extra": "ignore"}


class ScoredChunk(BaseModel):
    """
    A chunk annotated with retrieval scores from each pipeline stage.

    The score field is the *current best* score at any point in the pipeline:
      - After vector search   : score = cosine similarity
      - After RRF merge       : score = rrf_score
      - After KG boost        : score = rrf_score + kg_boost (if applied)
      - After reranking       : score = rerank_score
    """

    chunk: ChunkMetadata

    # ── Score trace ───────────────────────────────────────────────────────────
    score: float         = 0.0    # Current best score — always up to date
    vector_score: Optional[float] = None   # Raw Qdrant cosine similarity
    bm25_score: Optional[float]   = None   # Raw BM25 score
    rrf_score: Optional[float]    = None   # After reciprocal rank fusion
    rerank_score: Optional[float] = None   # Cross-encoder score (final authority)

    # ── Provenance ────────────────────────────────────────────────────────────
    retrieval_source: Literal["vector", "bm25", "hybrid", "kg"] = "hybrid"
    kg_boosted: bool     = False
    vector_rank: Optional[int] = None    # Rank in vector search results
    bm25_rank: Optional[int]   = None    # Rank in BM25 results

    # ── Pipeline helpers ──────────────────────────────────────────────────────

    def apply_kg_boost(self, boost: float) -> ScoredChunk:
        """
        Return a new ScoredChunk with KG score boost applied.
        Does not mutate — returns a copy (immutability for thread safety).
        """
        return self.model_copy(update={
            "score": self.score + boost,
            "kg_boosted": True,
        })

    def apply_rerank_score(self, rerank_score: float) -> ScoredChunk:
        """Return a new ScoredChunk with cross-encoder score as the final score."""
        return self.model_copy(update={
            "rerank_score": rerank_score,
            "score": rerank_score,
        })

    @property
    def is_equation(self) -> bool:
        return self.chunk.chunk_type == "equation"

    @property
    def is_table(self) -> bool:
        return self.chunk.chunk_type == "table"

    @property
    def should_skip_compression(self) -> bool:
        return self.chunk.should_skip_compression

    @property
    def text(self) -> str:
        """Convenience accessor — avoids .chunk.text everywhere."""
        return self.chunk.text

    @property
    def citation_key(self) -> str:
        return self.chunk.citation_key

    @property
    def display_source(self) -> str:
        return self.chunk.display_source

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_qdrant_result(
        cls,
        point_id: int,
        score: float,
        payload: dict,
    ) -> ScoredChunk:
        """Build a ScoredChunk directly from a Qdrant search result."""
        metadata = ChunkMetadata.from_qdrant_payload(payload)
        return cls(
            chunk=metadata,
            score=score,
            vector_score=score,
            retrieval_source="vector",
        )

    @classmethod
    def from_bm25_result(
        cls,
        chunk_metadata: ChunkMetadata,
        bm25_score: float,
        rank: int,
    ) -> ScoredChunk:
        """Build a ScoredChunk from a BM25 retrieval result."""
        return cls(
            chunk=chunk_metadata,
            score=bm25_score,
            bm25_score=bm25_score,
            bm25_rank=rank,
            retrieval_source="bm25",
        )

    def __repr__(self) -> str:
        return (
            f"ScoredChunk("
            f"score={self.score:.4f}, "
            f"source={self.chunk.source_type}, "
            f"citation={self.citation_key!r}, "
            f"text={self.chunk.text[:60]!r}…"
            f")"
        )