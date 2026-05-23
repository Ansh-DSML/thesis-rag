# ══════════════════════════════════════════════════════════════════════════════
# FILE: app/api/routes/health.py
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import APIRouter
from app.utils.logger import get_logger

health_router = APIRouter(tags=["health"])
_health_log   = get_logger("routes.health")


@health_router.get(
    "/health",
    summary="Liveness probe",
    description="Returns 200 if the app process is alive. No external checks.",
)
async def health() -> dict:
    """GET /health — liveness probe for Docker/Kubernetes."""
    return {"status": "ok"}


@health_router.get(
    "/ready",
    summary="Readiness probe",
    description="Checks that Qdrant and Redis are reachable before accepting traffic.",
)
async def ready() -> dict:
    """
    GET /ready — readiness probe.

    Returns:
        200 {"status": "ready", "qdrant": true, "redis": true}
        503 if any dependency is unreachable.
    """
    checks = {"qdrant": False, "redis": False}

    # Check Redis
    try:
        from app.core.cache.redis_cache import get_cache
        cache = await get_cache()
        checks["redis"] = await cache.health_check()
    except Exception as exc:
        _health_log.error("readiness_redis_failed", error=str(exc))

    # Check Qdrant
    try:
        from app.core.retrieval.hybrid_retriever import _get_qdrant_client
        from app.config.settings import get_settings
        client = _get_qdrant_client()
        collections = await client.get_collections()
        settings = get_settings()
        collection_names = [c.name for c in collections.collections]
        checks["qdrant"] = settings.qdrant_collection_name in collection_names
    except Exception as exc:
        _health_log.error("readiness_qdrant_failed", error=str(exc))

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_ready else "not_ready", **checks},
    )
