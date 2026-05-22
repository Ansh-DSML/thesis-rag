"""
app/config/settings.py

Single source of truth for every configuration value in the system.
All other files do:
    from app.config.settings import get_settings
    settings = get_settings()

Reads from .env automatically via pydantic-settings.
Values are cached after first load (lru_cache) — zero overhead on repeated calls.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # HF_HOME and hf_home both work
        extra="ignore",         # silently ignore unknown .env keys
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 1
    api_secret_key: str = ""

    @field_validator("app_env", mode="before")
    @classmethod
    def _normalise_app_env(cls, v: str) -> str:
        """Handle typos like 'developmentA' that sneak in via copy-paste."""
        v = str(v).strip()
        if v.startswith("development"):
            return "development"
        if v.startswith("production"):
            return "production"
        if v.startswith("test"):
            return "test"
        return "development"

    # ── LLM — Gemini ──────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    llm_model_main: str = "gemini-2.5-pro"      # Prompt 4 — answer generation
    llm_model_router: str = "gemini-2.5-pro"    # Prompt 1 — query routing (fast/cheap)
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.0                # Low = factual, deterministic
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 3                    # Exponential backoff retry count

    # ── HuggingFace / Embeddings ───────────────────────────────────────────────
    hf_token: str = ""
    hf_home: str = "D:\\thesis-rag\\.hf_cache"  # Updated path from ingestion fix
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 8
    embedding_device: str = "cpu"

    # BGE-large requires this prefix on QUERIES only (not on documents)
    # Documents were indexed without any prefix — keep it that way
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # ── Reranker (BGE cross-encoder) ───────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_top_k_input: int = 20      # Chunks fed into cross-encoder
    reranker_top_k_output: int = 5      # Chunks returned after reranking
    # Equation chunks get a +2.0 bonus on equation_lookup queries
    reranker_equation_boost: float = 2.0

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_use_cloud: bool = True
    qdrant_cloud_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "thesis_rag"
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 200
    qdrant_hnsw_ef: int = 128           # ef at query time — accuracy vs latency
    qdrant_top_k: int = 20              # Vector ANN candidates before RRF

    # ── BM25 ──────────────────────────────────────────────────────────────────
    bm25_index_path: str = "data/processed/bm25_index.pkl"
    bm25_top_k: int = 500               # BM25 candidates before RRF merge

    # ── Hybrid retrieval — RRF ────────────────────────────────────────────────
    # RRF score = rrf_vector_weight × 1/(rrf_k + vector_rank)
    #           + rrf_bm25_weight  × 1/(rrf_k + bm25_rank)
    rrf_vector_weight: float = 0.6
    rrf_bm25_weight: float = 0.4
    rrf_k: int = 60                     # RRF smoothing constant (60 is standard)

    # ── Knowledge graph ───────────────────────────────────────────────────────
    kg_output_path: str = "app/core/knowledge_graph/graph.json"
    use_knowledge_graph: bool = True
    kg_boost_score: float = 0.15        # Added to RRF score for KG-linked chunks

    # ── HyDE (Hypothetical Document Expansion) ────────────────────────────────
    # Final query vector = alpha × query_emb + (1-alpha) × hyde_emb
    hyde_alpha: float = 0.5             # 0.5 = equal mix; 1.0 = query only
    # HyDE is skipped for equation_lookup (exact symbolic match needed)
    hyde_skip_query_types: list[str] = Field(
        default=["equation_lookup"],
        description="Query types where HyDE is not applied.",
    )

    # ── Context compression — LLMLingua-2 ─────────────────────────────────────
    compression_ratio: float = 0.5              # Target: keep 50% of tokens
    compression_token_threshold: int = 3000     # Only compress if context > 3000 tokens
    # equation and table chunks are always bypassed (re-injected verbatim)

    # ── Redis Cloud ───────────────────────────────────────────────────────────
    redis_cloud_host: str = ""
    redis_cloud_port: int = 12658
    redis_cloud_password: str = ""
    cache_ttl_query: int = 86400        # 24 hours  — query result cache
    cache_ttl_embedding: int = 604800   # 7 days    — embedding cache
    cache_ttl_session: int = 86400      # 24 hours  — chat session memory

    # ── Chat memory ───────────────────────────────────────────────────────────
    chat_memory_k: int = 5              # Last k turn-pairs injected into prompt

    # ── Grobid (only needed for ingestion, not retrieval) ─────────────────────
    grobid_host: str = "localhost"
    grobid_port: int = 8070
    grobid_timeout_seconds: int = 120
    grobid_max_concurrent_requests: int = 2

    # ── Computed properties (derived — not in .env) ───────────────────────────

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @computed_field
    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @computed_field
    @property
    def grobid_base_url(self) -> str:
        return f"http://{self.grobid_host}:{self.grobid_port}"

    @computed_field
    @property
    def redis_url(self) -> str:
        """
        Redis Cloud uses TLS — rediss:// (double-s).
        ssl_cert_reqs=none disables cert verification (Redis Cloud self-signed).
        """
        return (
            f"redis://:{self.redis_cloud_password}"
            f"@{self.redis_cloud_host}:{self.redis_cloud_port}/0"
            f"?ssl_cert_reqs=none"
        )

    @computed_field
    @property
    def qdrant_url_with_port(self) -> str:
        """Ensure :6333 is present on the Qdrant Cloud URL."""
        url = self.qdrant_cloud_url.rstrip("/")
        if not url.endswith(":6333"):
            url = f"{url}:6333"
        return url

    def validate_required_secrets(self) -> None:
        """
        Call this at application startup to fail fast if secrets are missing.
        Raises ValueError with a clear message listing all missing fields.
        """
        missing = []
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if not self.qdrant_api_key:
            missing.append("QDRANT_API_KEY")
        if not self.qdrant_cloud_url:
            missing.append("QDRANT_CLOUD_URL")
        if not self.redis_cloud_password:
            missing.append("REDIS_CLOUD_PASSWORD")
        if not self.api_secret_key:
            missing.append("API_SECRET_KEY")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Check your .env file."
            )


@lru_cache()
def get_settings() -> Settings:
    """
    Returns the singleton Settings instance.
    Cached after first call — safe to call from any module without perf cost.

    Usage:
        from app.config.settings import get_settings
        settings = get_settings()
    """
    return Settings()