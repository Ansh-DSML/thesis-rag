"""
app/api/dependencies.py

FastAPI Depends() providers for dependency injection.

Every route that needs a component imports its provider from here.
Singletons are initialised on first call and reused across all requests.

Usage in routes:
    from app.api.dependencies import get_pipeline, get_cache, get_settings

    @router.post("/chat")
    async def chat(
        request: QueryRequest,
        pipeline: RAGPipeline = Depends(get_pipeline),
    ):
        return await pipeline.run(request)
"""

from __future__ import annotations

from app.config.settings import Settings, get_settings as _get_settings
from app.core.cache.redis_cache import RedisCache, get_cache as _get_cache
from app.core.pipeline import RAGPipeline, get_pipeline as _get_pipeline


async def get_pipeline() -> RAGPipeline:
    """Provide the RAGPipeline singleton to route handlers."""
    return _get_pipeline()


async def get_cache() -> RedisCache:
    """Provide the RedisCache singleton to route handlers."""
    return await _get_cache()


def get_settings() -> Settings:
    """Provide the Settings singleton to route handlers."""
    return _get_settings()