"""
app/core/cache/redis_cache.py

Async Redis cache layer — sits at the very start and end of every request.

Two cache namespaces:
  cache:query:{sha256}   — Full QueryResponse JSON, TTL=24h
  cache:emb:{sha256}     — Query embedding (1024-dim float32), TTL=7d

Design principles:
  - Graceful degradation: if Redis is unreachable, log a warning and
    continue without caching. The pipeline never crashes due to cache failure.
  - Async-first: all operations are async — no blocking calls in the request path.
  - Connection pooling: one shared pool across all requests (singleton via get_cache()).
  - Encoding: embeddings stored as raw bytes (compact); responses stored as JSON.

Usage:
    from app.core.cache.redis_cache import get_cache
    cache = await get_cache()

    # Query cache
    hit = await cache.get_query(query, session_id)
    if hit:
        return hit   # QueryResponse dict, skip entire pipeline

    # ... run pipeline ...

    await cache.set_query(query, session_id, response)

    # Embedding cache
    emb = await cache.get_embedding(query_text)
    if emb is None:
        emb = model.encode(query_text)
        await cache.set_embedding(query_text, emb)
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
from typing import Optional

import numpy as np

from app.config.settings import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

# ── Key builders ─────────────────────────────────────────────────────────────

def _query_cache_key(query: str, session_id: str) -> str:
    """
    Deterministic cache key for a query+session pair.
    SHA256 ensures fixed length and avoids Redis key-size issues.
    Includes session_id so different users' identical queries don't
    cross-contaminate (different chat histories = different context).
    """
    raw = f"{query.strip().lower()}::{session_id}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"cache:query:{digest}"


def _embedding_cache_key(text: str) -> str:
    """
    Cache key for a text embedding.
    Does NOT include session_id — the same text always produces the same embedding.
    """
    digest = hashlib.sha256(text.strip().encode()).hexdigest()
    return f"cache:emb:{digest}"


# ── Embedding serialization ───────────────────────────────────────────────────

def _serialize_embedding(embedding: np.ndarray) -> bytes:
    """
    Serialize a float32 numpy array to bytes for Redis storage.
    1024-dim float32 = 4096 bytes raw → ~5.5KB as base64.
    Much more compact than JSON (which would be ~15KB).
    """
    arr = embedding.astype(np.float32)
    # Prefix with dimension so we can validate on read
    dim = arr.shape[0]
    header = struct.pack(">I", dim)   # 4-byte big-endian uint32
    return header + arr.tobytes()


def _deserialize_embedding(data: bytes) -> np.ndarray:
    """Deserialize bytes back to a float32 numpy array."""
    dim = struct.unpack(">I", data[:4])[0]
    arr = np.frombuffer(data[4:], dtype=np.float32)
    if arr.shape[0] != dim:
        raise ValueError(f"Embedding dimension mismatch: expected {dim}, got {arr.shape[0]}")
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# REDIS CACHE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RedisCache:
    """
    Async Redis cache client.
    Instantiated once at startup via get_cache().
    All methods degrade gracefully on connection errors.
    """

    def __init__(self, client) -> None:
        self._client = client
        self._available = True   # flips to False on repeated connection failures

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _get_raw(self, key: str) -> Optional[bytes]:
        """Get raw bytes from Redis. Returns None on miss or error."""
        try:
            value = await self._client.get(key)
            return value
        except Exception as exc:
            log.warning("redis_get_failed", key=key, error=str(exc))
            return None

    async def _set_raw(self, key: str, value: bytes, ttl: int) -> bool:
        """Set raw bytes in Redis with TTL. Returns True on success."""
        try:
            await self._client.setex(key, ttl, value)
            return True
        except Exception as exc:
            log.warning("redis_set_failed", key=key, error=str(exc))
            return False

    async def _delete(self, key: str) -> bool:
        try:
            await self._client.delete(key)
            return True
        except Exception as exc:
            log.warning("redis_delete_failed", key=key, error=str(exc))
            return False

    # ── Query cache ───────────────────────────────────────────────────────────

    async def get_query(
        self,
        query: str,
        session_id: str,
    ) -> Optional[dict]:
        """
        Look up a cached QueryResponse for this (query, session_id) pair.
        Returns the response dict if found, None on miss.

        The caller should check:
            hit = await cache.get_query(query, session_id)
            if hit:
                return QueryResponse.model_validate(hit)
        """
        key = _query_cache_key(query, session_id)
        raw = await self._get_raw(key)
        if raw is None:
            log.debug("query_cache_miss", key=key[:20])
            return None
        try:
            result = json.loads(raw.decode("utf-8"))
            log.info("query_cache_hit", key=key[:20])
            return result
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("query_cache_corrupt", key=key[:20], error=str(exc))
            await self._delete(key)
            return None

    async def set_query(
        self,
        query: str,
        session_id: str,
        response: dict,
    ) -> bool:
        """
        Cache a QueryResponse dict for 24 hours.
        Pass response as a dict (use QueryResponse.model_dump()).
        """
        key = _query_cache_key(query, session_id)
        try:
            raw = json.dumps(response, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            log.warning("query_cache_serialize_failed", error=str(exc))
            return False
        success = await self._set_raw(key, raw, ttl=settings.cache_ttl_query)
        if success:
            log.debug("query_cached", key=key[:20], ttl=settings.cache_ttl_query)
        return success

    async def invalidate_query(self, query: str, session_id: str) -> bool:
        """Remove a specific query from cache (e.g. after data update)."""
        key = _query_cache_key(query, session_id)
        return await self._delete(key)

    # ── Embedding cache ───────────────────────────────────────────────────────

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Look up a cached embedding for this text.
        Returns a 1024-dim float32 numpy array on hit, None on miss.

        Usage in query_router / hyde:
            emb = await cache.get_embedding(query)
            if emb is None:
                emb = embed_model.encode(query)
                await cache.set_embedding(query, emb)
        """
        key = _embedding_cache_key(text)
        raw = await self._get_raw(key)
        if raw is None:
            return None
        try:
            arr = _deserialize_embedding(raw)
            log.debug("embedding_cache_hit", dim=arr.shape[0])
            return arr
        except Exception as exc:
            log.warning("embedding_cache_corrupt", error=str(exc))
            await self._delete(key)
            return None

    async def set_embedding(self, text: str, embedding: np.ndarray) -> bool:
        """Cache a query embedding for 7 days."""
        key = _embedding_cache_key(text)
        try:
            raw = _serialize_embedding(embedding)
        except Exception as exc:
            log.warning("embedding_serialize_failed", error=str(exc))
            return False
        return await self._set_raw(key, raw, ttl=settings.cache_ttl_embedding)

    # ── Utilities ─────────────────────────────────────────────────────────────

    async def flush_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        E.g. flush_pattern("cache:query:*") clears entire query cache.
        USE WITH CARE in production.
        Returns the number of keys deleted.
        """
        try:
            keys = []
            async for key in self._client.scan_iter(match=pattern, count=100):
                keys.append(key)
            if keys:
                deleted = await self._client.delete(*keys)
                log.info("cache_flushed", pattern=pattern, deleted=deleted)
                return deleted
            return 0
        except Exception as exc:
            log.error("flush_pattern_failed", pattern=pattern, error=str(exc))
            return 0

    async def flush_all_queries(self) -> int:
        """Clear the entire query cache namespace."""
        return await self.flush_pattern("cache:query:*")

    async def flush_all_embeddings(self) -> int:
        """Clear the entire embedding cache namespace."""
        return await self.flush_pattern("cache:emb:*")

    async def health_check(self) -> bool:
        """Ping Redis. Returns True if reachable."""
        try:
            await self._client.ping()
            return True
        except Exception as exc:
            log.error("redis_health_check_failed", error=str(exc))
            return False

    async def close(self) -> None:
        """Close the Redis connection pool gracefully."""
        try:
            await self._client.aclose()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON — one connection pool shared across all requests
# ══════════════════════════════════════════════════════════════════════════════

_cache_instance: Optional[RedisCache] = None


async def get_cache() -> RedisCache:
    """
    Return the shared RedisCache singleton.
    Creates the connection pool on first call.

    Import and call this from FastAPI dependencies or pipeline components:
        cache = await get_cache()
    """
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    try:
        import redis.asyncio as aioredis
    except ImportError:
        log.error("redis_not_installed", fix="pip install redis")
        raise

    try:
        client = aioredis.Redis(
            host=settings.redis_cloud_host,
            port=settings.redis_cloud_port,
            password=settings.redis_cloud_password,
            ssl=False,      # Redis Cloud uses self-signed certs
            decode_responses=False,  # we handle encoding ourselves
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
            max_connections=20,      # pool size — enough for production workers
        )

        # Verify connection immediately so startup fails fast if Redis is down
        await client.ping()
        log.info(
            "redis_connected",
            host=settings.redis_cloud_host,
            port=settings.redis_cloud_port,
        )

        _cache_instance = RedisCache(client)

    except Exception as exc:
        log.error(
            "redis_connection_failed",
            host=settings.redis_cloud_host,
            port=settings.redis_cloud_port,
            error=str(exc),
        )
        raise

    return _cache_instance


async def close_cache() -> None:
    """Close the Redis connection pool. Call on application shutdown."""
    global _cache_instance
    if _cache_instance is not None:
        await _cache_instance.close()
        _cache_instance = None
        log.info("redis_connection_closed")