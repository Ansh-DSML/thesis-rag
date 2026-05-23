# ══════════════════════════════════════════════════════════════════════════════
# FILE: app/api/routes/search.py
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import APIRouter, Depends
from app.api.dependencies import get_pipeline
from app.core.pipeline import RAGPipeline
from app.models.query import QueryRequest, SearchResponse
from app.utils.logger import get_logger

search_router = APIRouter(prefix="/search", tags=["search"])
_search_log   = get_logger("routes.search")


@search_router.post(
    "",
    response_model=SearchResponse,
    summary="Raw retrieval without answer generation",
    description=(
        "Runs pipeline steps 1–10 only: routing → embedding → retrieval → "
        "reranking. Returns scored chunks with metadata. "
        "Use this to debug retrieval quality without paying for LLM generation."
    ),
)
async def post_search(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> SearchResponse:
    """
    POST /search

    Same request body as POST /chat.
    Returns SearchResponse with scored chunks instead of a generated answer.

    Useful for:
      - Checking which chunks are retrieved for a given query
      - Debugging metadata filter effects
      - Comparing vector vs BM25 contributions (retrieval_source field)
      - Verifying KG boost is applying correctly (kg_boosted field)
    """
    try:
        return await pipeline.retrieve_only(request)
    except Exception as exc:
        _search_log.error("search_endpoint_error", error=str(exc))
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(exc))