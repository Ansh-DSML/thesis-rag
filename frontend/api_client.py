# frontend/api_client.py
# ─────────────────────────────────────────────────────────────────────────────
# All HTTP communication with the FastAPI backend lives here.
# app.py NEVER calls httpx directly — always go through this module.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
import time
from typing import Optional

import httpx

from config import cfg


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS  (plain classes — no Pydantic needed in the frontend)
# ═══════════════════════════════════════════════════════════════════════════════

class ChatResponse:
    """Normalised wrapper around the JSON returned by POST /chat."""

    def __init__(self, raw: dict):
        # Core answer — support both "answer" and "response" key names
        self.answer:      str            = raw.get("answer") or raw.get("response") or ""

        # Metadata from backend pipeline
        self.query_type:  Optional[str]  = raw.get("query_type")        # Gemini-2.5-Flash router
        self.cache_hit:   bool           = bool(raw.get("cache_hit", False))   # Redis hit
        self.kg_used:     bool           = bool(raw.get("kg_used", False))     # KG retrieval used
        self.hyde_used:   bool           = bool(raw.get("hyde_used", False))   # HyDE used
        self.latency_ms:  Optional[float]= raw.get("latency_ms")
        self.token_usage: int            = raw.get("token_usage", 0)
        self.citations:   list           = raw.get("citations", [])
        self.error:       Optional[str]  = None

        # Parse [Chapter X] references inline in the answer text
        self.chapter_refs: list[int] = _parse_chapter_refs(self.answer)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def succeeded(self) -> bool:
        return bool(self.answer.strip()) and self.error is None

    @property
    def latency_str(self) -> str:
        """Human-readable latency: '4.2s' or '1m 12s'."""
        if self.latency_ms is None:
            return ""
        s = self.latency_ms / 1000
        return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m {int(s % 60)}s"


class HealthResponse:
    """Normalised wrapper around GET /health."""

    def __init__(self, raw: dict, reachable: bool):
        self.reachable: bool = reachable
        self.status:    str  = raw.get("status", "unknown") if reachable else "unreachable"
        self.qdrant_ok: bool = raw.get("qdrant", False)       # Qdrant Cloud
        self.redis_ok:  bool = raw.get("redis", False)        # Redis Cloud
        self.model_ok:  bool = raw.get("models_loaded", False)

    @property
    def all_ok(self) -> bool:
        return self.reachable and self.status == "ok"

    @property
    def status_emoji(self) -> str:
        if not self.reachable:
            return "🔴"
        if self.all_ok:
            return "🟢"
        return "🟡"


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_chapter_refs(text: str) -> list[int]:
    """
    Extract chapter numbers from inline [Chapter X] or [Ch. X] citations.
    Returns a deduplicated list in order of first appearance.
    """
    pattern = r"\[(?:Chapter|Ch\.?)\s*(\d+)[^\]]*\]"
    matches  = re.findall(pattern, text, re.IGNORECASE)
    seen: list[int] = []
    for m in matches:
        n = int(m)
        if n not in seen:
            seen.append(n)
    return seen


def _make_client() -> httpx.Client:
    """Create an httpx client with the configured inference timeout."""
    return httpx.Client(timeout=cfg.api_timeout)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API CALLS
# ═══════════════════════════════════════════════════════════════════════════════

def check_health() -> HealthResponse:
    """
    GET /health — verify the backend is reachable and all services are healthy.

    Checks:
      • FastAPI process is up
      • Qdrant Cloud connection (vector DB)
      • Redis Cloud connection (cache + session memory)
      • Embedding + reranker models loaded

    Returns HealthResponse(reachable=False) on any connection error.
    Never raises — safe to call unconditionally at startup.
    """
    try:
        with httpx.Client(timeout=8) as client:
            resp = client.get(f"{cfg.api_url}/health")
        if resp.status_code == 200:
            return HealthResponse(resp.json(), reachable=True)
        return HealthResponse({}, reachable=False)
    except Exception:
        return HealthResponse({}, reachable=False)


def send_message(query: str, session_id: str) -> ChatResponse:
    """
    POST /chat — run the full RAG pipeline and return the answer.

    Pipeline on the backend:
      Gemini-2.5-Flash router → HyDE → hybrid retrieval (Qdrant + BM25 + KG)
      → BGE reranker → LLMLingua-2 compression → Llama-3.3-70b (Groq) answer

    Retries up to cfg.api_retries times on timeout/network errors.
    HTTP 4xx/5xx errors are not retried (they won't self-heal).
    Returns ChatResponse with .error set if all attempts fail.
    """
    payload    = {"query": query.strip(), "session_id": session_id}
    last_error = ""

    for attempt in range(cfg.api_retries + 1):
        try:
            t0 = time.perf_counter()
            with _make_client() as client:
                resp = client.post(f"{cfg.api_url}/chat", json=payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            resp.raise_for_status()
            data = resp.json()

            # Use client-measured round-trip if backend didn't return latency_ms
            if "latency_ms" not in data:
                data["latency_ms"] = elapsed_ms

            return ChatResponse(data)

        except httpx.HTTPStatusError as e:
            # Server returned 4xx or 5xx — no point retrying
            last_error = (
                f"Server error {e.response.status_code}: "
                f"{e.response.text[:200]}"
            )
            break

        except httpx.TimeoutException:
            last_error = (
                f"Request timed out after {cfg.api_timeout}s. "
                "BGE models or Groq may still be loading — please retry in 30 s."
            )
            if attempt < cfg.api_retries:
                time.sleep(2)   # brief pause before retry

        except httpx.ConnectError:
            last_error = (
                f"Cannot reach backend at {cfg.api_url}. "
                "Verify the FastAPI service is running on EC2 and port 8000 is open."
            )
            break  # connection refused won't fix itself on retry

        except Exception as e:
            last_error = str(e)
            break

    # All attempts failed — return an empty ChatResponse with the error set
    cr        = ChatResponse({})
    cr.error  = last_error
    return cr


def get_chat_history(session_id: str) -> list[dict]:
    """
    GET /chat/history/{session_id} — load previous turns from Redis.

    Used to restore context when the user refreshes the page.
    Returns an empty list on any error — chat still works without history.
    """
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{cfg.api_url}/chat/history/{session_id}")
        if resp.status_code == 200:
            data = resp.json()
            # Backend may return {"messages": [...]} or directly [...]
            return data.get("messages", data) if isinstance(data, dict) else data
    except Exception:
        pass
    return []


def clear_session(session_id: str) -> bool:
    """
    DELETE /chat/history/{session_id} — wipe Redis session memory.

    Called when the user clicks "New conversation" in the sidebar.
    Returns True on success, False on any error.
    """
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.delete(f"{cfg.api_url}/chat/history/{session_id}")
        return resp.status_code in (200, 204)
    except Exception:
        return False