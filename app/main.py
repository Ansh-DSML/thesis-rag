"""
app/main.py

FastAPI application entry point. Write this last — it wires everything together.

Startup sequence (lifespan):
  1. Validate all required secrets (.env completeness check)
  2. Connect to Redis Cloud (fail fast if unreachable)
  3. Load BM25 index + chunk metadata into memory
  4. Load KG graph into memory
  5. Warm embedding model (BGE-large) — avoids cold start on first request
  6. Initialise RAGPipeline singleton

Shutdown sequence:
  1. Close Redis connection pool

Middleware chain (outermost → innermost):
  RequestLoggingMiddleware  — trace_id injection + request/response logging
  RateLimitMiddleware       — 60 req/min per user_id (Redis-backed)
  APIKeyMiddleware          — protects /ingest routes
  CORSMiddleware            — cross-origin headers

Run in development:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Run in production:
  uvicorn app.main:app --workers 1 --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware.auth import APIKeyMiddleware
from app.api.middleware.logging import RequestLoggingMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.api.routes.chat import chat_router
from app.api.routes.health import health_router
from app.api.routes.search import search_router
from app.config.settings import get_settings
from app.utils.logger import get_logger
from app.api.routes.ingest import ingest_router 

log      = get_logger(__name__)
settings = get_settings()


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN — startup and shutdown
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise all singletons and warm models.
    Shutdown: close external connections cleanly.
    """
    log.info("startup_begin", env=settings.app_env)

    # ── 1. Validate required secrets ─────────────────────────────────────────
    try:
        settings.validate_required_secrets()
        log.info("secrets_validated")
    except ValueError as exc:
        log.error("secrets_validation_failed", error=str(exc))
        raise RuntimeError(str(exc))

    # ── 2. Connect to Redis ───────────────────────────────────────────────────
    from app.core.cache.redis_cache import get_cache
    try:
        cache = await get_cache()
        log.info("redis_ready")
    except Exception as exc:
        log.error("redis_startup_failed", error=str(exc))
        raise

    # ── 3. Load BM25 index + chunk metadata ───────────────────────────────────
    try:
        from app.core.retrieval.hybrid_retriever import _load_retrieval_data
        await _load_retrieval_data()
        log.info("retrieval_data_ready")
    except Exception as exc:
        log.error("retrieval_data_startup_failed", error=str(exc))
        raise

    # ── 4. Load KG graph ──────────────────────────────────────────────────────
    try:
        from app.core.pipeline import _load_kg_graph
        _load_kg_graph()
        log.info("kg_graph_ready")
    except Exception as exc:
        # KG is optional — warn but don't crash startup
        log.warning("kg_graph_startup_failed", error=str(exc))

    # ── 5. Warm embedding model (avoids cold start on first request) ───────────
    try:
        from app.core.retrieval.hyde import get_embed_model
        await get_embed_model()
        log.info("embedding_model_warm")
    except Exception as exc:
        log.error("embedding_model_warm_failed", error=str(exc))
        raise

    # ── 6. Initialise RAGPipeline singleton ───────────────────────────────────
    from app.core.pipeline import get_pipeline
    get_pipeline()
    log.info("pipeline_ready")

    log.info(
        "startup_complete",
        host=settings.app_host,
        port=settings.app_port,
        env=settings.app_env,
    )

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("shutdown_begin")
    from app.core.cache.redis_cache import close_cache
    await close_cache()
    log.info("shutdown_complete")


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Thesis RAG API",
    description=(
        "Retrieval-Augmented Generation API for a doctoral thesis. "
        "Answers questions using hybrid vector + BM25 retrieval, "
        "cross-encoder reranking, and Gemini for answer generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Development: allow all origins.
# Production:  restrict to your frontend domain.
_allowed_origins = (
    ["*"]
    if settings.is_development
    else ["https://yourdomain.com"]  # ← update for production
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ── Custom middleware (added last = runs first in request chain) ───────────────
# Order: RequestLogging → RateLimit → APIKey → route handler
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(search_router)
app.include_router(ingest_router)

# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    log.error(
        "unhandled_exception",
        path=str(request.url.path),
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again.",
            "error_type": type(exc).__name__,
        },
    )