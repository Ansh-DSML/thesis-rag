"""
app/core/generation/answer_generator.py

Answer Generator — Prompt 4 orchestrator.

Responsibilities:
  1. Receive top-k reranked ScoredChunks from the retrieval pipeline
  2. Format chunks into a readable context string
  3. Retrieve formatted chat history from ChatMemory
  4. Select the right answer sub-template via query_type
  5. Build the complete Prompt 4 string
  6. Call Gemini via LLMClient
  7. Parse inline citation keys from the generated answer
  8. Build and return Citation objects matched to source chunks

This is the LAST step before returning QueryResponse to the user.

Usage (called from app/core/pipeline.py):
    from app.core.generation.answer_generator import AnswerGenerator

    generator = AnswerGenerator()
    answer, citations, token_count = await generator.generate(
        query=request.query,
        query_type=router_output.query_type,
        chunks=reranked_chunks,
        session_id=request.session_id,
    )
"""

from __future__ import annotations

import re
import time
from typing import Optional

from app.config.settings import get_settings
from app.core.generation.llm_client import get_llm_client, LLMResponse
from app.core.generation.prompts.answer_prompts import (
    build_answer_prompt,
    format_chunks_for_context,
)
from app.models.chunk import ScoredChunk
from app.models.query import Citation, QueryType, RouterOutput
from app.utils.logger import get_logger, StageTimer
from app.utils.text_utils import count_tokens

log      = get_logger(__name__)
settings = get_settings()

# ── Citation key patterns found in generated text ─────────────────────────────
# Matches: [Chapter 3], [Chapter 3], [Zhang 2022], [Smith et al. 2021]
_CITATION_PATTERN = re.compile(
    r"\["
    r"(?:"
    r"Chapter\s+\d+"               # [Chapter N]
    r"|"
    r"[A-Z][a-zA-Z\s\-]+\s+\d{4}" # [Author(s) Year]
    r")"
    r"\]"
)


# ══════════════════════════════════════════════════════════════════════════════
# CITATION PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _extract_citation_keys(text: str) -> list[str]:
    """
    Extract all inline citation keys from a generated answer string.
    Returns deduplicated list preserving first-occurrence order.
    """
    found = _CITATION_PATTERN.findall(text)
    seen: set[str] = set()
    unique: list[str] = []
    for key in found:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _match_citations_to_chunks(
    cited_keys: list[str],
    chunks: list[ScoredChunk],
) -> list[Citation]:
    """
    For each citation key found in the answer, find the corresponding
    ScoredChunk and build a Citation object.

    Matching strategy:
      1. Exact match on chunk.citation_key
      2. If no exact match, try normalised comparison (case-insensitive,
         strip spaces)
      3. Unmatched citation keys are logged as warnings but not dropped —
         a minimal Citation is still created so the user sees the reference.

    Returns Citations ordered by their first appearance in the answer.
    """
    # Build lookup from citation_key → ScoredChunk
    chunk_by_key: dict[str, ScoredChunk] = {}
    for sc in chunks:
        key = sc.chunk.citation_key
        chunk_by_key[key] = sc
        # Also index by normalised key for fuzzy matching
        chunk_by_key[key.lower().replace(" ", "")] = sc

    citations: list[Citation] = []
    for key in cited_keys:
        sc = chunk_by_key.get(key) or chunk_by_key.get(
            key.lower().replace(" ", "")
        )
        if sc:
            citations.append(Citation.from_scored_chunk(sc))
        else:
            log.warning(
                "citation_key_unmatched",
                key=key,
                available_keys=list(chunk_by_key.keys())[:10],
            )
            # Still include it — the answer text references it
            citations.append(
                Citation(
                    chunk_id="unknown",
                    citation_key=key,
                    source_type="thesis",        # best guess
                    display_source=key,
                )
            )

    return citations


# ══════════════════════════════════════════════════════════════════════════════
# ANSWER GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class AnswerGenerator:
    """
    Orchestrates Prompt 4: formats context, builds prompt,
    calls Gemini, parses citations, returns structured output.
    """

    def __init__(self) -> None:
        self._llm = get_llm_client()

    async def generate(
        self,
        query: str,
        query_type: QueryType,
        chunks: list[ScoredChunk],
        session_id: str,
        router_output: Optional[RouterOutput] = None,
    ) -> tuple[str, list[Citation], int]:
        """
        Generate an answer for the given query using reranked chunks.

        Args:
            query:         The user's original question.
            query_type:    Classified type from RouterOutput.
            chunks:        Top-k ScoredChunks after reranking (max 5).
            session_id:    Session ID for chat memory retrieval.
            router_output: Full RouterOutput (optional, for logging).

        Returns:
            Tuple of:
              - answer (str)          — generated answer text
              - citations (list)      — Citation objects for sources used
              - token_count (int)     — total LLM tokens consumed
        """
        with StageTimer("answer_generation", log, query_type=query_type):
            # ── Step 1: Guard — handle empty chunks ───────────────────────────
            if not chunks:
                log.warning("answer_generator_no_chunks", query=query[:60])
                return (
                    "I could not find relevant information in the thesis or "
                    "research papers to answer this question.",
                    [],
                    0,
                )

            # ── Step 2: Format chunks into context string ─────────────────────
            formatted_context, citation_keys = format_chunks_for_context(chunks)

            # ── Step 3: Retrieve chat history ─────────────────────────────────
            chat_history = ""
            try:
                from app.core.memory.chat_memory import get_memory
                memory = await get_memory(session_id)
                chat_history = await memory.get_formatted(max_tokens=600)
            except Exception as exc:
                log.warning(
                    "chat_history_retrieval_failed",
                    session_id=session_id,
                    error=str(exc),
                )

            # ── Step 4: Build Prompt 4 ────────────────────────────────────────
            prompt = build_answer_prompt(
                query=query,
                query_type=query_type,
                formatted_context=formatted_context,
                citation_keys=citation_keys,
                chat_history=chat_history,
            )

            log.debug(
                "answer_prompt_ready",
                prompt_tokens_approx=count_tokens(prompt),
                chunks_used=len(chunks),
                has_history=bool(chat_history.strip()),
            )

            # ── Step 5: Call Gemini ───────────────────────────────────────────
            llm_response: LLMResponse = await self._llm.complete(
                prompt=prompt,
                role="main",
            )

            if llm_response.is_empty:
                log.error(
                    "llm_empty_response",
                    query=query[:60],
                    query_type=query_type,
                )
                return (
                    "The language model returned an empty response. "
                    "Please try rephrasing your question.",
                    [],
                    llm_response.token_count,
                )

            answer_text = llm_response.text

            # ── Step 6: Parse citations from the answer ───────────────────────
            cited_keys = _extract_citation_keys(answer_text)

            if not cited_keys:
                # Model answered without citing — use all provided chunk keys
                log.warning(
                    "answer_no_citations_found",
                    query_type=query_type,
                    query_preview=query[:60],
                )
                cited_keys = citation_keys[:3]   # fallback: cite top-3 chunks

            citations = _match_citations_to_chunks(cited_keys, chunks)

            log.info(
                "answer_generated",
                query_type=query_type,
                answer_chars=len(answer_text),
                citations_found=len(citations),
                token_count=llm_response.token_count,
            )

            return answer_text, citations, llm_response.token_count

    async def save_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
    ) -> None:
        """
        Save this Q&A turn to Redis chat memory after generation.
        Strips citation JSON from answer before saving — keeps only the
        inline-cited text (e.g. "PSO achieved 94.2% [Chapter 4]").

        Called by the main pipeline (pipeline.py) after response is built,
        not inside generate() — allows the pipeline to save regardless of
        whether generation succeeded or failed gracefully.
        """
        try:
            from app.core.memory.chat_memory import get_memory
            memory = await get_memory(session_id)
            await memory.add_turn(human_msg=query, ai_msg=answer)
        except Exception as exc:
            log.warning(
                "chat_memory_save_failed",
                session_id=session_id,
                error=str(exc),
            )