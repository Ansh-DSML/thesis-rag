# ══════════════════════════════════════════════════════════════════════════════
# FILE: app/api/middleware/rate_limit.py
# ══════════════════════════════════════════════════════════════════════════════
"""
Per-user sliding window rate limiter backed by Redis.

Key  : ratelimit:{user_id}:{current_minute}
Value: request count (integer)
TTL  : 90 seconds (gives full 60s window + 30s buffer)

Limit: 60 requests per minute per user_id (configurable).
User identity: X-User-ID header, or falls back to client IP.
"""

import time as _time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.utils.logger import get_logger

_rl_log = get_logger("middleware.rate_limit")

RATE_LIMIT_RPM   = 60     # requests per minute per user
RATE_WINDOW_SECS = 60     # window size in seconds
_SKIP_PATHS      = {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter using Redis counters.
    Skips health/docs routes to avoid false throttling.
    Returns 429 Too Many Requests when limit is exceeded.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        user_id = (
            request.headers.get("X-User-ID")
            or request.client.host
            or "anonymous"
        )

        try:
            is_allowed, current_count = await self._check_rate_limit(user_id)
        except Exception as exc:
            # Redis error — allow request rather than blocking users
            _rl_log.warning("rate_limit_redis_error", error=str(exc))
            return await call_next(request)

        if not is_allowed:
            _rl_log.warning(
                "rate_limit_exceeded",
                user_id=user_id,
                count=current_count,
                limit=RATE_LIMIT_RPM,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: {RATE_LIMIT_RPM} requests/minute. "
                        f"Current count: {current_count}. Please wait and retry."
                    )
                },
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
        return response

    @staticmethod
    async def _check_rate_limit(user_id: str) -> tuple[bool, int]:
        """
        Increment the sliding window counter and return (allowed, count).
        Uses Redis INCR + EXPIRE for atomic counter management.
        """
        from app.core.cache.redis_cache import get_cache

        cache   = await get_cache()
        client  = cache._client
        minute  = int(_time.time() // RATE_WINDOW_SECS)
        key     = f"ratelimit:{user_id}:{minute}"

        count = await client.incr(key)
        if count == 1:
            # First request in this window — set TTL
            await client.expire(key, RATE_WINDOW_SECS + 30)

        return count <= RATE_LIMIT_RPM, count
