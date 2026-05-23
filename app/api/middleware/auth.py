# ══════════════════════════════════════════════════════════════════════════════
# FILE: app/api/middleware/auth.py
# ══════════════════════════════════════════════════════════════════════════════
"""
API key authentication middleware.
Protects POST /ingest/* routes with X-API-Key header check.
All other routes are public (auth handled at router level for /chat).
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.config.settings import get_settings
from app.utils.logger import get_logger

_auth_log = get_logger("middleware.auth")
_settings = get_settings()

# Routes that require API key authentication
_PROTECTED_PREFIXES = ["/ingest"]


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Check X-API-Key header for protected routes.
    Returns 403 Forbidden if the key is missing or incorrect.
    Passes through all other routes without checking.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Only protect routes matching _PROTECTED_PREFIXES
        needs_auth = any(path.startswith(p) for p in _PROTECTED_PREFIXES)
        if not needs_auth:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if not api_key or api_key != _settings.api_secret_key:
            _auth_log.warning(
                "api_key_rejected",
                path=path,
                provided_key=api_key[:6] + "…" if api_key else "(none)",
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid or missing API key."},
            )

        return await call_next(request)
