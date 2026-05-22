"""
app/core/memory/chat_memory.py

Per-session conversation memory backed by Redis.

Stores the last k=5 human/AI turn pairs per session.
Retrieved and formatted for injection into Prompt 4 so the model
can resolve references like "that equation you mentioned" or
"compare it to the previous paper".

Storage format in Redis:
  Key   : memory:{session_id}
  Value : JSON list of turn dicts, newest last
  TTL   : 24 hours (reset on every new turn)

Turn dict structure:
  {
    "human": "What feature selection method was used?",
    "ai":    "PSO was used... [Chapter 3]",
    "ts":    "2026-05-21T05:46:25+00:00"
  }

Usage:
    from app.core.memory.chat_memory import get_memory

    memory = await get_memory(session_id)
    history = await memory.get_formatted()   # str ready for prompt
    await memory.add_turn(human_msg=query, ai_msg=answer)
    await memory.clear()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from app.config.settings import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


# ── Key builder ───────────────────────────────────────────────────────────────

def _memory_key(session_id: str) -> str:
    return f"memory:{session_id}"


# ══════════════════════════════════════════════════════════════════════════════
# CHAT MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class ChatMemory:
    """
    Manages conversation history for one session in Redis.

    Instantiate per-request with the session_id from QueryRequest.
    Requires the RedisCache singleton — use get_memory() factory below.
    """

    def __init__(self, session_id: str, cache=None) -> None:
        self.session_id = session_id
        self._key = _memory_key(session_id)
        self._cache = cache
        self._k = settings.chat_memory_k
        self._ttl = settings.cache_ttl_session

    # ── Internal: Redis access ────────────────────────────────────────────────

    async def _get_client(self):
        """Return the raw Redis async client."""
        if self._cache is None:
            from app.core.cache.redis_cache import get_cache
            self._cache = await get_cache()
        return self._cache._client

    async def _load_turns(self) -> list[dict]:
        """
        Load all stored turns from Redis.
        Returns [] on miss, error, or corrupted data — never raises.
        """
        try:
            client = await self._get_client()
            raw = await client.get(self._key)
            if raw is None:
                return []
            turns = json.loads(raw.decode("utf-8"))
            if not isinstance(turns, list):
                log.warning("chat_memory_corrupt", session_id=self.session_id)
                return []
            return turns
        except Exception as exc:
            log.warning(
                "chat_memory_load_failed",
                session_id=self.session_id,
                error=str(exc),
            )
            return []

    async def _save_turns(self, turns: list[dict]) -> bool:
        """
        Persist turns list to Redis, resetting TTL.
        Always trims to the last k turns before saving.
        """
        if len(turns) > self._k:
            turns = turns[-self._k:]
        try:
            client = await self._get_client()
            raw = json.dumps(turns, ensure_ascii=False).encode("utf-8")
            await client.setex(self._key, self._ttl, raw)
            return True
        except Exception as exc:
            log.warning(
                "chat_memory_save_failed",
                session_id=self.session_id,
                error=str(exc),
            )
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_turns(self) -> list[dict]:
        """
        Return the last k turn dicts for this session.
        Each dict: {'human': str, 'ai': str, 'ts': str}.
        Returns [] for a new session.
        """
        return await self._load_turns()

    async def add_turn(self, human_msg: str, ai_msg: str) -> bool:
        """
        Append a new human/AI turn and reset TTL.
        Call this AFTER generation completes.

        Args:
            human_msg: The user's raw query string.
            ai_msg:    The generated answer (with inline citation keys
                       like [Chapter 3] but without full citation JSON).
        Returns:
            True on success, False if Redis is unavailable.
        """
        turns = await self._load_turns()
        turn = {
            "human": human_msg.strip(),
            "ai": ai_msg.strip(),
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        turns.append(turn)
        success = await self._save_turns(turns)
        if success:
            log.debug(
                "chat_turn_saved",
                session_id=self.session_id,
                turn_count=min(len(turns), self._k),
            )
        return success

    async def get_formatted(self, max_tokens: int = 800) -> str:
        """
        Return conversation history as a formatted string for Prompt 4 injection.

        Format (chronological, most recent last):
            [Turn 1]
            User: What feature selection method was used?
            Assistant: PSO was used for feature selection... [Chapter 3]

            [Turn 2]
            User: What was the PSO population size?
            Assistant: The population size was set to 30... [Chapter 3]

        Args:
            max_tokens: Soft token budget. Older turns are dropped first
                        if history exceeds this limit.
        Returns:
            Formatted string, or "" for new sessions.
        """
        from app.utils.text_utils import count_tokens

        turns = await self._load_turns()
        if not turns:
            return ""

        # Walk turns newest-first to respect token budget,
        # then reverse back to chronological order for the prompt.
        blocks: list[str] = []
        total_tokens = 0

        for i, turn in enumerate(reversed(turns)):
            turn_number = len(turns) - i
            block = (
                f"[Turn {turn_number}]\n"
                f"User: {turn['human']}\n"
                f"Assistant: {turn['ai']}"
            )
            block_tokens = count_tokens(block)
            if total_tokens + block_tokens > max_tokens:
                log.debug(
                    "chat_memory_truncated",
                    session_id=self.session_id,
                    kept=len(blocks),
                    total=len(turns),
                )
                break
            blocks.append(block)
            total_tokens += block_tokens

        if not blocks:
            return ""

        blocks.reverse()
        return "\n\n".join(blocks)

    async def get_last_turn(self) -> Optional[dict]:
        """
        Return the most recent turn dict, or None for new sessions.
        Useful for follow-up detection in the query router.
        """
        turns = await self._load_turns()
        return turns[-1] if turns else None

    async def is_followup(self, query: str) -> bool:
        """
        Heuristic: True if this query looks like a follow-up to the
        previous turn based on pronoun/reference signals.

        Used by the query router to inject memory context even when
        the query isn't entity-centric.
        """
        followup_signals = [
            "it", "that", "this", "those", "they", "them",
            "the same", "mentioned", "above", "previous", "earlier",
            "you said", "you mentioned", "what about", "and also",
            "how about", "compare it", "explain that",
        ]
        last = await self.get_last_turn()
        if last is None:
            return False
        q_lower = query.lower()
        return any(signal in q_lower for signal in followup_signals)

    async def turn_count(self) -> int:
        """Return number of stored turns for this session."""
        return len(await self._load_turns())

    async def ttl_remaining(self) -> int:
        """
        Seconds until this session expires in Redis.
        Returns -1 if key does not exist.
        """
        try:
            client = await self._get_client()
            return await client.ttl(self._key)
        except Exception:
            return -1

    async def clear(self) -> bool:
        """
        Delete this session's entire conversation history.
        Call on explicit logout or session reset.
        """
        try:
            client = await self._get_client()
            await client.delete(self._key)
            log.info("chat_memory_cleared", session_id=self.session_id)
            return True
        except Exception as exc:
            log.warning(
                "chat_memory_clear_failed",
                session_id=self.session_id,
                error=str(exc),
            )
            return False

    def __repr__(self) -> str:
        return f"ChatMemory(session_id={self.session_id!r}, k={self._k})"


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

async def get_memory(session_id: str) -> ChatMemory:
    """
    Return a ChatMemory instance with Redis cache already injected.

    This is what you import everywhere in the pipeline:
        memory = await get_memory(request.session_id)
        history_str = await memory.get_formatted()
        await memory.add_turn(query, answer)
    """
    from app.core.cache.redis_cache import get_cache
    cache = await get_cache()
    return ChatMemory(session_id=session_id, cache=cache)


async def clear_session(session_id: str) -> bool:
    """
    Delete all memory for a session. Convenience wrapper for the API layer.
    Call from a session-reset endpoint if you add one.
    """
    memory = await get_memory(session_id)
    return await memory.clear()