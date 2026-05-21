"""
app/utils/logger.py

Structured logging setup using structlog.

Behaviour:
  development  → pretty coloured console output (human-readable)
  production   → JSON lines to stdout (machine-parseable, for log aggregators)

Every log record automatically includes:
  - timestamp
  - log level
  - trace_id  (set per-request via set_trace_id(); propagates through the call stack)
  - module / function name

Usage:
  from app.utils.logger import get_logger, set_trace_id
  log = get_logger(__name__)

  # At the start of each request (in FastAPI middleware):
  set_trace_id("abc123")

  # In any module:
  log.info("retrieval_complete", chunks_found=5, latency_ms=42.1)
  log.error("qdrant_timeout", error=str(e))
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

# ── Trace ID context variable ─────────────────────────────────────────────────
# Set once per request in middleware; automatically included in every log call
# made during that request — even across async tasks in the same context.
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def set_trace_id(trace_id: str | None = None) -> str:
    """
    Set the trace ID for the current async context.
    If no trace_id is given, a new UUID4 is generated.
    Returns the trace_id that was set.
    """
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    """Return the current trace ID, or empty string if not set."""
    return _trace_id_var.get()


# ── Shared processors (run for every log event) ───────────────────────────────

def _add_trace_id(logger, method_name, event_dict: dict) -> dict:
    """Inject the current request's trace_id into every log record."""
    tid = _trace_id_var.get()
    if tid:
        event_dict["trace_id"] = tid
    return event_dict


def _add_log_level(logger, method_name, event_dict: dict) -> dict:
    """Add a clean 'level' field (structlog uses method_name which can be 'msg')."""
    event_dict["level"] = method_name.upper()
    return event_dict


# ── Logger factory ────────────────────────────────────────────────────────────

_configured = False


def _configure_structlog(is_production: bool) -> None:
    global _configured
    if _configured:
        return

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_trace_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if is_production:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    _configured = True


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Returns a structlog BoundLogger for a module.

    Configures structlog on first call (reads APP_ENV from environment).
    Safe to call at module import time.

    Usage:
        log = get_logger(__name__)
        log.info("stage_complete", stage=3, chunks=2755)
        log.warning("grobid_timeout", file="paper.pdf", timeout=120)
        log.error("qdrant_error", error=str(e), collection="thesis_rag")
    """
    import os
    is_prod = os.getenv("APP_ENV", "development").startswith("production")
    _configure_structlog(is_production=is_prod)
    return structlog.get_logger(name)


# ── Request-scoped context helpers ────────────────────────────────────────────

def bind_request_context(
    trace_id: str,
    session_id: str,
    query: str | None = None,
) -> None:
    """
    Bind per-request fields to structlog's contextvars so they appear in
    every log line for the duration of this request without passing them around.

    Call this at the start of each request (after set_trace_id).

    structlog.contextvars are automatically cleared between requests when
    using async frameworks — no manual cleanup needed.
    """
    structlog.contextvars.clear_contextvars()
    ctx: dict = {"trace_id": trace_id, "session_id": session_id}
    if query:
        # Log first 80 chars of query — enough for debugging without bloating logs
        ctx["query_preview"] = query[:80]
    structlog.contextvars.bind_contextvars(**ctx)


def clear_request_context() -> None:
    """Clear per-request context. Call in middleware after response is sent."""
    structlog.contextvars.clear_contextvars()


# ── Convenience: pipeline stage timer ────────────────────────────────────────

class StageTimer:
    """
    Context manager that logs stage start/end with elapsed time.

    Usage:
        with StageTimer("vector_search", log, chunks=20):
            results = await qdrant.search(...)
        # logs: stage_complete stage=vector_search latency_ms=43.2 chunks=20
    """

    def __init__(self, stage_name: str, logger, **extra):
        self._name = stage_name
        self._log = logger
        self._extra = extra
        self._start: float = 0.0

    def __enter__(self) -> StageTimer:
        import time
        self._start = time.perf_counter()
        self._log.debug("stage_start", stage=self._name, **self._extra)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        if exc_type is None:
            self._log.info(
                "stage_complete",
                stage=self._name,
                latency_ms=round(elapsed_ms, 2),
                **self._extra,
            )
        else:
            self._log.error(
                "stage_failed",
                stage=self._name,
                latency_ms=round(elapsed_ms, 2),
                error=str(exc_val),
                **self._extra,
            )
        return False  # don't suppress exceptions