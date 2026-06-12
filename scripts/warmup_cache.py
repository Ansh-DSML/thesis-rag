"""
scripts/warmup_cache.py

Pre-populate Redis static cache with answers for all suggested questions.
Run this ONCE on your EC2 instance after deployment to warm up the cache.

Usage:
    cd /path/to/thesis-rag
    python scripts/warmup_cache.py

What it does:
  1. Reads all suggested questions (same list as the Streamlit frontend)
  2. Runs each through the full 14-step RAG pipeline (slow — ~90s each)
  3. Stores each response in cache:static:{sha256} — NO TTL, NO session_id
  4. Future requests for these exact queries → instant (~5ms) from any session

After running:
  - All 13 suggested queries will be permanently cached in Redis
  - Any user clicking a suggested question → instant response
  - The cache persists until you explicitly flush it or Redis restarts

To refresh the cache (e.g. after re-ingesting data):
    python scripts/warmup_cache.py --flush-first

To check what's cached:
    python scripts/warmup_cache.py --check-only
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings
from app.core.cache.redis_cache import get_cache, _static_cache_key
from app.core.pipeline import get_pipeline
from app.models.query import QueryRequest

settings = get_settings()


# ══════════════════════════════════════════════════════════════════════════════
# SUGGESTED QUESTIONS — keep in sync with frontend/config.py
# ══════════════════════════════════════════════════════════════════════════════

SUGGESTED_QUERIES = [
    # Sidebar suggested questions (first 6 shown in sidebar, rest on empty state)
    "What bio-inspired algorithms are used in the thesis and what does each optimise?",
    "What is the PSO velocity update equation used for DR threshold optimisation?",
    "Compare Architecture 1 HG-MEF and Architecture 2 HG-MEF-Bio performance.",
    "How was Severe DR accuracy improved from 39.3% to 92.8%?",
    "What datasets were used for AMD, Glaucoma, and DR and how were they split?",
    "What is the Ben Graham preprocessing pipeline and why is it used for DR?",
    "What is the GateDiversityLoss and why does it prevent routing collapse?",
    "What are the five DR severity grades and their clinical significance?",
    "How does the Firefly Algorithm guide EfficientNet-B4 channel attention?",
    "What future work is proposed in the conclusion chapter?",

    # Example query types shown on empty state
    "What accuracy did the AMD expert achieve in Architecture 2?",
    "What is the PSO velocity update equation for DR threshold optimisation?",
    "Compare Architecture 1 HG-MEF vs Architecture 2 HG-MEF-Bio across all diseases.",
]

# Deduplicate (some queries overlap between sidebar and empty state)
SUGGESTED_QUERIES = list(dict.fromkeys(SUGGESTED_QUERIES))


# ══════════════════════════════════════════════════════════════════════════════
# WARM-UP LOGIC
# ══════════════════════════════════════════════════════════════════════════════

WARMUP_SESSION_ID = "__warmup__"


async def check_cached_status():
    """Check which suggested queries are already cached."""
    cache = await get_cache()
    print(f"\n{'─' * 70}")
    print(f"  Static cache status — {len(SUGGESTED_QUERIES)} suggested queries")
    print(f"{'─' * 70}\n")

    cached_count = 0
    for i, query in enumerate(SUGGESTED_QUERIES, 1):
        hit = await cache.get_static_query(query)
        status = "✅ CACHED" if hit else "❌ NOT CACHED"
        if hit:
            cached_count += 1
        print(f"  {i:2d}. {status}  {query[:65]}")

    print(f"\n  Total: {cached_count}/{len(SUGGESTED_QUERIES)} cached\n")
    return cached_count


async def flush_static_cache():
    """Delete all static cache entries."""
    cache = await get_cache()
    deleted = await cache.flush_all_static()
    print(f"\n  🗑  Flushed {deleted} static cache entries.\n")
    return deleted


async def warmup_all():
    """Run each suggested query through the full pipeline and cache the result."""
    cache    = await get_cache()
    pipeline = get_pipeline()

    total    = len(SUGGESTED_QUERIES)
    success  = 0
    skipped  = 0
    failed   = 0

    print(f"\n{'═' * 70}")
    print(f"  WARMING UP STATIC CACHE — {total} queries")
    print(f"  Each query runs the full 14-step pipeline (~60–120s on CPU)")
    print(f"  Estimated total time: {total * 90 // 60}–{total * 120 // 60} minutes")
    print(f"{'═' * 70}\n")

    overall_start = time.perf_counter()

    for i, query in enumerate(SUGGESTED_QUERIES, 1):
        # Check if already cached
        existing = await cache.get_static_query(query)
        if existing:
            print(f"  [{i}/{total}] ⏭  Already cached: {query[:55]}…")
            skipped += 1
            continue

        print(f"  [{i}/{total}] 🔄 Processing: {query[:55]}…", end="", flush=True)
        start = time.perf_counter()

        try:
            request = QueryRequest(
                query=query,
                session_id=WARMUP_SESSION_ID,
            )
            response = await pipeline.run(request)
            elapsed = time.perf_counter() - start

            # Store in static cache (permanent, session-independent)
            response_dict = response.model_dump(mode="json")
            ok = await cache.set_static_query(query, response_dict)

            if ok:
                print(f"  ✅ {elapsed:.1f}s")
                success += 1
            else:
                print(f"  ⚠️  Pipeline OK but cache write failed ({elapsed:.1f}s)")
                failed += 1

        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"  ❌ FAILED after {elapsed:.1f}s: {str(exc)[:80]}")
            failed += 1

    total_time = time.perf_counter() - overall_start

    print(f"\n{'═' * 70}")
    print(f"  WARM-UP COMPLETE")
    print(f"  ✅ Cached:  {success}")
    print(f"  ⏭  Skipped: {skipped} (already cached)")
    print(f"  ❌ Failed:  {failed}")
    print(f"  ⏱  Total:   {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'═' * 70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    args = sys.argv[1:]

    if "--check-only" in args:
        await check_cached_status()
        return

    if "--flush-first" in args:
        await flush_static_cache()

    if "--flush-only" in args:
        await flush_static_cache()
        return

    await warmup_all()

    # Show final status
    await check_cached_status()


if __name__ == "__main__":
    asyncio.run(main())
