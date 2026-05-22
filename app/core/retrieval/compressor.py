"""
app/core/retrieval/compressor.py

Step 6 — Context Compression (LLMLingua-2).

Compresses the text of top-5 reranked chunks from ~3000+ tokens down to ~50%.
This reduces LLM input cost and improves answer focus by removing low-information
tokens while preserving the content the model needs most.

Bypass rule (critical):
  Equation chunks (chunk_type == "equation") are ALWAYS bypassed —
  never passed to LLMLingua-2. They are re-injected verbatim at the
  FRONT of the final context string.

  Table chunks (chunk_type == "table") follow the same rule.

  Reason: LLMLingua-2 cannot reliably preserve LaTeX notation, table
  cell values, or mathematical structure during compression.

Threshold:
  Compression only runs if total context token count > 3000 tokens.
  Below this threshold, the context is returned as-is (no latency cost).

Output format:
  [VERBATIM — equation/table chunks first]
  [COMPRESSED — text chunks after]

Graceful degradation:
  If LLMLingua-2 is not installed or fails, the full uncompressed context
  is returned. The pipeline never crashes due to compression failure.

Install:
  pip install llmlingua

Model:
  microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
  (~300MB — much smaller than the embedding/reranker models)
"""

from __future__ import annotations

import asyncio
from typing import Optional

from app.config.settings import get_settings
from app.core.generation.prompts.compression_prompt import (
    get_compression_config,
    should_compress,
)
from app.models.chunk import ScoredChunk
from app.models.query import QueryType
from app.utils.logger import get_logger, StageTimer
from app.utils.text_utils import count_tokens

log      = get_logger(__name__)
settings = get_settings()

# LLMLingua-2 model name (smaller bert-base variant — good balance of speed/quality)
_LLMLINGUA2_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


# ══════════════════════════════════════════════════════════════════════════════
# LLMLINGUA-2 SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_compressor_model  = None
_compressor_lock   = asyncio.Lock()
_compressor_failed = False    # if loading failed, skip compression gracefully


async def _get_compressor():
    """
    Lazily load LLMLingua-2 compressor model.
    Returns None if llmlingua is not installed or loading fails —
    callers fall back to returning uncompressed context.
    """
    global _compressor_model, _compressor_failed

    if _compressor_model is not None:
        return _compressor_model
    if _compressor_failed:
        return None

    async with _compressor_lock:
        if _compressor_model is not None:
            return _compressor_model
        if _compressor_failed:
            return None

        try:
            from llmlingua import PromptCompressor
        except ImportError:
            log.warning(
                "llmlingua_not_installed",
                note="Compression disabled. Install with: pip install llmlingua",
            )
            _compressor_failed = True
            return None

        log.info("loading_llmlingua2", model=_LLMLINGUA2_MODEL)
        loop = asyncio.get_event_loop()

        try:
            _compressor_model = await loop.run_in_executor(
                None,
                lambda: PromptCompressor(
                    model_name=_LLMLINGUA2_MODEL,
                    use_llmlingua2=True,
                    device_map=settings.embedding_device,
                ),
            )
            log.info("llmlingua2_loaded", model=_LLMLINGUA2_MODEL)
        except Exception as exc:
            log.error("llmlingua2_load_failed", error=str(exc))
            _compressor_failed = True
            return None

    return _compressor_model


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _format_chunk_verbatim(sc: ScoredChunk, idx: int) -> str:
    """
    Format a bypassed chunk (equation / table) for verbatim injection.
    Labelled clearly so the LLM knows this content is exact.
    """
    chunk = sc.chunk
    label = (
        f"[VERBATIM {chunk.chunk_type.upper()} | "
        f"Citation: {chunk.citation_key} | "
        f"Source: {chunk.display_source}]"
    )
    return f"{label}\n{chunk.text}"


def _format_chunk_for_compression(sc: ScoredChunk, idx: int) -> str:
    """
    Format a text chunk for input to LLMLingua-2.
    Plain text with a lightweight header the compressor can work with.
    """
    chunk = sc.chunk
    header = (
        f"[Source: {chunk.display_source} | "
        f"Citation: {chunk.citation_key}]"
    )
    return f"{header}\n{chunk.text}"


def _build_full_context(chunks: list[ScoredChunk]) -> str:
    """
    Build the full uncompressed context string from top-k chunks.
    Used when compression is skipped (below threshold or unavailable).
    """
    parts = []
    for i, sc in enumerate(chunks, start=1):
        chunk = sc.chunk
        label = (
            f"[PASSAGE {i} | "
            f"Citation: {chunk.citation_key} | "
            f"Source: {chunk.display_source}]"
        )
        parts.append(f"{label}\n{chunk.text}")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# COMPRESSOR
# ══════════════════════════════════════════════════════════════════════════════

class ContextCompressor:
    """
    LLMLingua-2 context compressor with equation/table bypass.
    Singleton — get via get_compressor().
    """

    async def compress(
        self,
        chunks: list[ScoredChunk],
        query: str,
        query_type: QueryType,
    ) -> tuple[str, bool]:
        """
        Compress the context built from top-k reranked chunks.

        Algorithm:
          1. Separate chunks into: bypass_chunks (equation/table) + text_chunks
          2. Build full context string and count tokens
          3. If tokens <= threshold: return full context, skip compression
          4. Compress text_chunks with LLMLingua-2
          5. Prepend bypass_chunks verbatim before compressed text
          6. Return combined context string

        Args:
            chunks:     Top-k ScoredChunks from reranker (usually 5).
            query:      Original user query (unused here, for future use).
            query_type: Controls compression config (rate, force_tokens).

        Returns:
            Tuple of:
              - context_str (str): Final context ready for Prompt 4.
              - compressed (bool): True if LLMLingua-2 ran, False if skipped.
        """
        if not chunks:
            return "", False

        with StageTimer("context_compression", log, query_type=query_type):

            # ── Step 1: Separate bypass vs text chunks ────────────────────────
            bypass_chunks = [sc for sc in chunks if sc.should_skip_compression]
            text_chunks   = [sc for sc in chunks if not sc.should_skip_compression]

            log.debug(
                "compressor_split",
                bypass=len(bypass_chunks),
                text=len(text_chunks),
            )

            # ── Step 2: Build full context + count tokens ─────────────────────
            full_context = _build_full_context(chunks)
            total_tokens = count_tokens(full_context)

            # ── Step 3: Check threshold ───────────────────────────────────────
            threshold = settings.compression_token_threshold
            if not should_compress(total_tokens, threshold):
                log.info(
                    "compression_skipped",
                    reason="below_threshold",
                    tokens=total_tokens,
                    threshold=threshold,
                )
                return full_context, False

            if not text_chunks:
                # All chunks are equations/tables — nothing to compress
                log.info("compression_skipped", reason="all_bypass_chunks")
                verbatim_parts = [
                    _format_chunk_verbatim(sc, i)
                    for i, sc in enumerate(bypass_chunks, 1)
                ]
                return "\n\n".join(verbatim_parts), False

            # ── Step 4: Compress text chunks with LLMLingua-2 ─────────────────
            compressor = await _get_compressor()

            if compressor is None:
                # LLMLingua-2 not available — return full uncompressed context
                log.warning(
                    "compression_skipped",
                    reason="llmlingua2_unavailable",
                    tokens=total_tokens,
                )
                return full_context, False

            comp_config = get_compression_config(query_type)
            text_passages = [
                _format_chunk_for_compression(sc, i)
                for i, sc in enumerate(text_chunks, 1)
            ]

            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: compressor.compress_prompt(
                        context=text_passages,
                        rate=comp_config["rate"],
                        force_tokens=comp_config.get("force_tokens", []),
                        condition_in_question=comp_config.get(
                            "condition_in_question", ""
                        ),
                        drop_consecutive=comp_config.get("drop_consecutive", True),
                    ),
                )
                compressed_text = result.get("compressed_prompt", "")
                orig_tokens  = result.get("origin_tokens", total_tokens)
                final_tokens = result.get("compressed_tokens", count_tokens(compressed_text))

                log.info(
                    "compression_complete",
                    original_tokens=orig_tokens,
                    compressed_tokens=final_tokens,
                    ratio=round(final_tokens / max(orig_tokens, 1), 2),
                )

            except Exception as exc:
                log.error(
                    "llmlingua2_compression_failed",
                    error=str(exc),
                    fallback="returning_full_context",
                )
                return full_context, False

            # ── Step 5: Prepend bypass chunks verbatim ────────────────────────
            final_parts: list[str] = []

            if bypass_chunks:
                verbatim_header = (
                    "━━━ VERBATIM EQUATIONS / TABLES "
                    "(not compressed — reproduced exactly) ━━━"
                )
                final_parts.append(verbatim_header)
                for i, sc in enumerate(bypass_chunks, 1):
                    final_parts.append(_format_chunk_verbatim(sc, i))

            if compressed_text.strip():
                if bypass_chunks:
                    final_parts.append(
                        "━━━ COMPRESSED TEXT PASSAGES ━━━"
                    )
                final_parts.append(compressed_text.strip())

            return "\n\n".join(final_parts), True


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_compressor_instance: Optional[ContextCompressor] = None


def get_compressor() -> ContextCompressor:
    """
    Return the shared ContextCompressor singleton.

    Usage in pipeline.py:
        from app.core.retrieval.compressor import get_compressor
        compressor = get_compressor()
        context, was_compressed = await compressor.compress(
            chunks=top5_chunks,
            query=query,
            query_type=query_type,
        )
    """
    global _compressor_instance
    if _compressor_instance is None:
        _compressor_instance = ContextCompressor()
    return _compressor_instance