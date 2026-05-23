"""
app/core/retrieval/query_router.py

Step 2 — Query Router (runs on every cache miss).

Calls Prompt 1 via Gemini (router role = fast model).
Returns RouterOutput which controls every downstream decision:
  use_graph      → whether to run KG lookup (Step 3)
  use_hyde       → whether to run HyDE expansion (Step 4)
  source_filter  → Qdrant metadata pre-filter
  chapter_filter → Qdrant metadata pre-filter
  entities       → fed into KG lookup + metadata filter

Input:  raw query string + session_id (to fetch last conversation turn)
Output: RouterOutput (Pydantic model, never raises — fallback on failure)
"""

from __future__ import annotations

from app.config.settings import get_settings
from app.core.generation.llm_client import get_llm_client
from app.core.generation.prompts.router_prompt import (
    build_router_prompt,
    parse_router_output,
)
from app.models.query import RouterOutput
from app.utils.logger import get_logger, StageTimer

log      = get_logger(__name__)
settings = get_settings()


class QueryRouter:
    """
    Classifies the user query and extracts routing metadata.

    Instantiate once and reuse (singleton pattern via get_query_router()).
    The underlying LLM client is already a singleton so no extra cost.
    """

    def __init__(self) -> None:
        self._llm = get_llm_client()

    async def route(
        self,
        query: str,
        session_id: str,
    ) -> RouterOutput:
        """
        Run Prompt 1 and return a RouterOutput.

        Steps:
          1. Fetch last conversation turn from Redis (for follow-up context)
          2. Build router prompt string
          3. Call Gemini (router role — fast)
          4. Parse JSON response → RouterOutput
          5. Apply user-level overrides from QueryRequest (handled in pipeline)

        Never raises — returns RouterOutput.fallback() on any failure.

        Args:
            query:      The user's raw question.
            session_id: Used to fetch the last conversation turn so the
                        router can classify follow-up queries correctly.

        Returns:
            RouterOutput with query_type, entities, and routing flags set.
        """
        with StageTimer("query_router", log, session_id=session_id):

            # ── Step 1: Get last conversation turn ────────────────────────────
            last_turn = await self._get_last_turn(session_id)

            # ── Step 2: Build router prompt ───────────────────────────────────
            prompt = build_router_prompt(query=query, last_turn=last_turn)

            # ── Step 3: Call Gemini (router role) ─────────────────────────────
            try:
                llm_response = await self._llm.complete(
                    prompt=prompt,
                    role="router",
                    max_tokens=500,     # RouterOutput JSON is small
                    disable_thinking=True,
                )
                raw_text = llm_response.text
            except Exception as exc:
                log.error(
                    "query_router_llm_failed",
                    error=str(exc),
                    query_preview=query[:60],
                )
                return RouterOutput.fallback(query)

            # ── Step 4: Parse JSON → RouterOutput ─────────────────────────────
            router_output = parse_router_output(
                raw_response=raw_text,
                query=query,
            )

            log.info(
                "query_routed",
                query_type=router_output.query_type,
                entities=router_output.entities,
                use_graph=router_output.use_graph,
                use_hyde=router_output.use_hyde,
                source_filter=router_output.source_filter,
                chapter_filter=router_output.chapter_filter,
            )

            return router_output

    def apply_request_overrides(
        self,
        router_output: RouterOutput,
        source_filter: str | None,
        chapter_filter: int | None,
        use_graph: bool | None,
    ) -> RouterOutput:
        """
        Apply user-level overrides from QueryRequest on top of router output.

        Users can explicitly set source_filter, chapter_filter, or use_graph
        in their request. These override the router's automatic decisions.

        Called from pipeline.py after route() returns.
        Returns a new RouterOutput with overrides applied.
        """
        overrides = {}

        if source_filter is not None:
            overrides["source_filter"] = source_filter
            log.debug("router_override_source", source_filter=source_filter)

        if chapter_filter is not None:
            overrides["chapter_filter"] = chapter_filter
            log.debug("router_override_chapter", chapter_filter=chapter_filter)

        if use_graph is not None:
            overrides["use_graph"] = use_graph
            log.debug("router_override_use_graph", use_graph=use_graph)

        if not overrides:
            return router_output

        return router_output.model_copy(update=overrides)

    async def _get_last_turn(self, session_id: str) -> dict | None:
        """
        Fetch the most recent conversation turn from Redis.
        Returns None for new sessions or on any Redis error.
        """
        try:
            from app.core.memory.chat_memory import get_memory
            memory = await get_memory(session_id)
            return await memory.get_last_turn()
        except Exception as exc:
            log.warning(
                "router_last_turn_fetch_failed",
                session_id=session_id,
                error=str(exc),
            )
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────

_router_instance: QueryRouter | None = None


def get_query_router() -> QueryRouter:
    """
    Return the shared QueryRouter singleton.

    Usage in pipeline.py:
        from app.core.retrieval.query_router import get_query_router
        router = get_query_router()
        router_output = await router.route(query, session_id)
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = QueryRouter()
    return _router_instance