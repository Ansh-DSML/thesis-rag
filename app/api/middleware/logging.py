# ══════════════════════════════════════════════════════════════════════════════
# FILE: app/api/middleware/logging.py
# ══════════════════════════════════════════════════════════════════════════════
"""
Request/response structured logging middleware.

Sets a unique trace_id on every request.
Logs: method, path, user_id, status_code, latency_ms.
Clears structlog context after each request (no bleed between requests).
"""

import time as _time_log

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from app.utils.logger import (
    get_logger,
    set_trace_id,
    bind_request_context,
    clear_request_context,
)

_req_log = get_logger("middleware.logging")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Injects trace_id into every request context.
    Logs structured request/response records for every API call.
    """

    async def dispatch(self, request: Request, call_next):
        trace_id = set_trace_id()
        user_id  = request.headers.get("X-User-ID", "anonymous")
        session_id = request.headers.get("X-Session-ID", "")

        bind_request_context(
            trace_id=trace_id,
            session_id=session_id,
        )

        start = _time_log.perf_counter()

        _req_log.info(
            "request_start",
            method=request.method,
            path=request.url.path,
            user_id=user_id,
            trace_id=trace_id,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = round((_time_log.perf_counter() - start) * 1000, 2)
            _req_log.error(
                "request_error",
                method=request.method,
                path=request.url.path,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise
        finally:
            clear_request_context()

        latency_ms = round((_time_log.perf_counter() - start) * 1000, 2)
        _req_log.info(
            "request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
            user_id=user_id,
            trace_id=trace_id,
        )

        # Inject trace_id into response headers for client-side debugging
        response.headers["X-Trace-ID"] = trace_id
        return response