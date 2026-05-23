"""
app/core/retrieval/hyde.py

Steps 4 + 5 — HyDE Expansion and Query Embedding.

This file owns everything related to the query vector:
  Step 4: Generate hypothetical paragraph via Prompt 2 (skipped for equations)
  Step 5: Embed query (always) + embed HyDE paragraph (if ran) + average them

Final vector formula:
  vector = α × embed(query) + (1-α) × embed(hyde_paragraph)    [if HyDE ran]
  vector = embed(query)                                          [if HyDE skipped]
  vector is L2-normalised before use in Qdrant.

BGE-large query encoding rule:
  Queries MUST use the instruction prefix:
    "Represent this sentence for searching relevant passages: {query}"
  Documents were indexed WITHOUT this prefix.
  Applying prefix to queries is what makes BGE-large accurate.

Caching:
  The final query vector is cached in Redis (7d TTL) keyed by
  SHA256(prefix + query + hyde_flag + query_type).
  On a cache hit the entire Steps 4+5 are skipped.

Usage (called from pipeline.py):
    from app.core.retrieval.hyde import get_hyde_expander
    expander = get_hyde_expander()
    query_vector = await expander.get_query_vector(
        query=query,
        query_type=router_output.query_type,
        use_hyde=router_output.use_hyde,
    )
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Optional

import numpy as np

from app.config.settings import get_settings
from app.core.generation.llm_client import get_llm_client
from app.core.generation.prompts.hyde_prompt import build_hyde_prompt, should_run_hyde
from app.models.query import QueryType
from app.utils.logger import get_logger, StageTimer

log      = get_logger(__name__)
settings = get_settings()


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL SINGLETON
# Loaded once at first use — ~1.3GB, expensive to load repeatedly.
# ══════════════════════════════════════════════════════════════════════════════

_embed_model = None
_embed_lock  = asyncio.Lock()


async def get_embed_model():
    """
    Return the shared SentenceTransformer embedding model.
    Thread-safe lazy initialisation using asyncio.Lock.
    First call downloads the model if not cached locally.
    Subsequent calls return immediately (cached in module variable).
    """
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    async with _embed_lock:
        # Double-check inside lock (another coroutine may have loaded it)
        if _embed_model is not None:
            return _embed_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.error(
                "sentence_transformers_not_installed",
                fix="pip install sentence-transformers",
            )
            raise

        log.info(
            "loading_embedding_model",
            model=settings.embedding_model,
            device=settings.embedding_device,
        )

        # Load in thread pool — blocks but doesn't freeze the event loop
        loop = asyncio.get_event_loop()
        _embed_model = await loop.run_in_executor(
            None,
            lambda: __import__("sentence_transformers").SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device,
                cache_folder=settings.hf_home,
            ),
        )

        log.info(
            "embedding_model_loaded",
            model=settings.embedding_model,
            device=settings.embedding_device,
            dim=settings.embedding_dimension,
        )

    return _embed_model


# ══════════════════════════════════════════════════════════════════════════════
# HYDE EXPANDER
# ══════════════════════════════════════════════════════════════════════════════

class HyDEExpander:
    """
    Owns query embedding and optional HyDE expansion.

    Singleton — get via get_hyde_expander().
    """

    def __init__(self) -> None:
        self._llm = get_llm_client()

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_query_vector(
        self,
        query: str,
        query_type: QueryType,
        use_hyde: bool,
    ) -> np.ndarray:
        """
        Return the final 1024-dim query vector for Qdrant search.

        If HyDE runs:
            vector = normalise(α × embed(query) + (1-α) × embed(hyde_para))
        If HyDE skipped:
            vector = embed(query_with_prefix)

        Results are cached in Redis. Cache key encodes query + hyde decision
        so a cached query-only vector is not returned for a HyDE-enabled call.

        Args:
            query:      The user's raw question.
            query_type: From RouterOutput — controls HyDE template selection.
            use_hyde:   From RouterOutput — whether to run HyDE.

        Returns:
            1024-dim float32 numpy array, L2-normalised.
        """
        run_hyde = should_run_hyde(query_type=query_type, use_hyde_flag=use_hyde)
        cache_key = self._cache_key(query, query_type, run_hyde)

        # ── Cache check ───────────────────────────────────────────────────────
        cached = await self._get_cached_vector(cache_key)
        if cached is not None:
            log.debug("query_vector_cache_hit", query_type=query_type, hyde=run_hyde)
            return cached

        with StageTimer("query_embedding", log, query_type=query_type, hyde=run_hyde):

            # ── Always: embed the raw query ───────────────────────────────────
            prefixed_query = settings.embedding_query_prefix + query
            query_emb = await self._embed(prefixed_query)

            if run_hyde:
                # ── Step 4: Generate hypothetical paragraph ───────────────────
                hyde_text = await self._generate_hyde_paragraph(query, query_type)

                if hyde_text:
                    # ── Step 5: Embed HyDE paragraph + average ────────────────
                    prefixed_hyde = settings.embedding_query_prefix + hyde_text
                    hyde_emb = await self._embed(prefixed_hyde)

                    alpha = settings.hyde_alpha
                    combined = alpha * query_emb + (1.0 - alpha) * hyde_emb
                    final_vector = self._normalise(combined)

                    log.debug(
                        "hyde_vector_combined",
                        alpha=alpha,
                        hyde_chars=len(hyde_text),
                    )
                else:
                    # HyDE generation failed — fall back to query-only vector
                    log.warning("hyde_generation_empty_fallback")
                    final_vector = query_emb
            else:
                final_vector = query_emb

        # ── Cache result ──────────────────────────────────────────────────────
        await self._set_cached_vector(cache_key, final_vector)

        return final_vector

    async def embed_text_for_indexing(self, text: str) -> np.ndarray:
        """
        Embed a document chunk WITHOUT the query prefix.
        Used if any component needs to re-embed a chunk at retrieval time.
        (Normally not needed — chunks are pre-embedded in Qdrant.)
        """
        return await self._embed(text)  # no prefix for documents

    # ── Private: HyDE paragraph generation ───────────────────────────────────

    async def _generate_hyde_paragraph(
        self,
        query: str,
        query_type: QueryType,
    ) -> str:
        """
        Call Gemini (main role) with Prompt 2 to generate a hypothetical
        expert answer paragraph (100–150 words).

        Returns empty string on failure so the caller can fall back gracefully.
        """
        with StageTimer("hyde_generation", log, query_type=query_type):
            prompt = build_hyde_prompt(query=query, query_type=query_type)
            try:
                response = await self._llm.complete(
                    prompt=prompt,
                    role="main",
                    max_tokens=600,     # 100–150 words + some buffer
                )
                if response.is_empty:
                    log.warning("hyde_llm_empty_response", query_preview=query[:60])
                    return ""

                log.debug(
                    "hyde_paragraph_generated",
                    chars=len(response.text),
                    tokens=response.token_count,
                )
                return response.text

            except Exception as exc:
                log.error(
                    "hyde_generation_failed",
                    error=str(exc),
                    query_preview=query[:60],
                )
                return ""

    # ── Private: Embedding ────────────────────────────────────────────────────

    async def _embed(self, text: str) -> np.ndarray:
        """
        Encode a single text string using BGE-large.
        Runs in a thread pool executor so it doesn't block the event loop.
        Returns a 1024-dim float32 numpy array, L2-normalised.
        """
        model = await get_embed_model()
        loop  = asyncio.get_event_loop()

        emb: np.ndarray = await loop.run_in_executor(
            None,
            lambda: model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )

        # Validate dimension
        if emb.shape[0] != settings.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"got {emb.shape[0]}, expected {settings.embedding_dimension}. "
                f"Check EMBEDDING_MODEL in .env."
            )

        return emb.astype(np.float32)

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        """L2-normalise a vector. Returns a unit vector."""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return vector
        return (vector / norm).astype(np.float32)

    # ── Private: Cache helpers ────────────────────────────────────────────────

    @staticmethod
    def _cache_key(query: str, query_type: str, run_hyde: bool) -> str:
        """
        Deterministic cache key for this (query, type, hyde) combination.
        SHA256 ensures fixed length and no Redis key issues.
        """
        raw = f"qvec::{query.strip()}::type={query_type}::hyde={run_hyde}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def _get_cached_vector(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached query vector from Redis."""
        try:
            from app.core.cache.redis_cache import get_cache
            cache = await get_cache()
            return await cache.get_embedding(key)
        except Exception as exc:
            log.warning("query_vector_cache_get_failed", error=str(exc))
            return None

    async def _set_cached_vector(self, key: str, vector: np.ndarray) -> None:
        """Store query vector in Redis."""
        try:
            from app.core.cache.redis_cache import get_cache
            cache = await get_cache()
            await cache.set_embedding(key, vector)
        except Exception as exc:
            log.warning("query_vector_cache_set_failed", error=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_expander_instance: Optional[HyDEExpander] = None


def get_hyde_expander() -> HyDEExpander:
    """
    Return the shared HyDEExpander singleton.

    Usage in pipeline.py:
        from app.core.retrieval.hyde import get_hyde_expander
        expander = get_hyde_expander()
        query_vector = await expander.get_query_vector(query, query_type, use_hyde)
    """
    global _expander_instance
    if _expander_instance is None:
        _expander_instance = HyDEExpander()
    return _expander_instance