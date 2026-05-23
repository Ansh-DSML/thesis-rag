# ══════════════════════════════════════════════════════════════════════════════
# FILE: app/api/routes/chat.py
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_pipeline, get_cache
from app.core.cache.redis_cache import RedisCache
from app.core.memory.chat_memory import get_memory
from app.core.pipeline import RAGPipeline
from app.models.query import QueryRequest, QueryResponse
from app.utils.logger import get_logger

chat_router = APIRouter(prefix="/chat", tags=["chat"])
_chat_log   = get_logger("routes.chat")


@chat_router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question about the thesis",
    description=(
        "Runs the full 14-step RAG pipeline: cache check → routing → "
        "retrieval → reranking → compression → answer generation. "
        "Returns a structured answer with inline citations."
    ),
)
async def post_chat(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """
    POST /chat

    Body:
        query      : str  — the question
        session_id : str  — unique per-user session (ties to Redis memory)
        source_filter  : "thesis" | "paper" | null   (optional)
        chapter_filter : 1–6 | null                  (optional)
        use_graph      : bool | null                  (optional)

    Returns QueryResponse with answer, citations, latency_ms, cache_hit.
    """
    try:
        response = await pipeline.run(request)
        return response
    except Exception as exc:
        _chat_log.error(
            "chat_endpoint_error",
            error=str(exc),
            session_id=request.session_id,
            query_preview=request.query[:60],
        )
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(exc)}",
        )


@chat_router.get(
    "/history/{session_id}",
    summary="Get conversation history for a session",
    description="Returns the last k conversation turns stored in Redis for this session.",
)
async def get_chat_history(session_id: str) -> dict:
    """
    GET /chat/history/{session_id}

    Returns:
        {
          "session_id": "...",
          "turns": [{"human": "...", "ai": "...", "ts": "..."}],
          "count": 3,
          "ttl_remaining_seconds": 84000
        }
    """
    try:
        memory = await get_memory(session_id)
        turns  = await memory.get_turns()
        ttl    = await memory.ttl_remaining()
        return {
            "session_id":           session_id,
            "turns":                turns,
            "count":                len(turns),
            "ttl_remaining_seconds": ttl,
        }
    except Exception as exc:
        _chat_log.error("chat_history_error", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@chat_router.delete(
    "/history/{session_id}",
    summary="Clear conversation history for a session",
)
async def delete_chat_history(session_id: str) -> dict:
    """DELETE /chat/history/{session_id} — clears Redis memory for this session."""
    memory  = await get_memory(session_id)
    cleared = await memory.clear()
    return {"session_id": session_id, "cleared": cleared}
