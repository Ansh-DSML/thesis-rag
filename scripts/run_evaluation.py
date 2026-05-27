"""
scripts/run_evaluation.py

RAGAS evaluation runner for the Thesis RAG system.

Pipeline:
  1. Load test_questions.json  →  EvalDataset (58 questions)
  2. Pre-load chunks_with_metadata.json  →  chunk_id → full text lookup
  3. For each question: POST /chat  →  generated answer + cited chunk_ids
  4. Look up full chunk text for each cited chunk_id (RAGAS needs full text)
  5. Build a HuggingFace Dataset from all EvalRows
  6. Run ragas.evaluate() with Gemini as the judge LLM
  7. Build RAGASResult + EvalReport  →  save JSON + CSV
  8. Print score summary table with per-query-type breakdown

Usage examples:
  # Full run (all 58 questions)
  python scripts/run_evaluation.py

  # Quick smoke-test with 5 random questions
  python scripts/run_evaluation.py --sample 5

  # Only specific query type
  python scripts/run_evaluation.py --query-type factual

  # Only questions from Chapter 3
  python scripts/run_evaluation.py --chapter 3

  # Skip RAGAS scoring (just test API connectivity)
  python scripts/run_evaluation.py --sample 3 --dry-run

  # Verbose: print each question + answer as it runs
  python scripts/run_evaluation.py --verbose

  # Use a different questions file
  python scripts/run_evaluation.py --questions data/evaluation/test_questions.json

  # Point to a different running server
  python scripts/run_evaluation.py --api-base http://localhost:8001

Output files (in data/evaluation/ragas_results/):
  results_{timestamp}.json   Full per-question breakdown
  results_{timestamp}.csv    Excel-friendly flat table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Optional
import json
from pathlib import Path
from datasets import Dataset
from ragas import EvaluationDataset, SingleTurnSample
from ragas import evaluate as ragas_evaluate
from ragas.run_config import RunConfig
from dotenv import load_dotenv
load_dotenv()

EVAL_CHECKPOINT_PATH = "data/evaluation/ragas_results/eval_checkpoint.json"

# ── Path bootstrap — allow running from project root ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Internal imports (after path fix) ────────────────────────────────────────
from app.config.settings import get_settings
from app.models.evaluation import (
    EvalDataset,
    EvalQuestion,
    EvalReport,
    EvalRow,
    RAGASResult,
    THRESHOLDS,
)

# ── Third-party ───────────────────────────────────────────────────────────────
import httpx
from tqdm import tqdm

settings = get_settings()

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CHUNKS_PATH      = "data/processed/chunks_with_metadata.json"
QUESTIONS_PATH   = "data/evaluation/test_questions.json"
RESULTS_DIR      = "data/evaluation/ragas_results"

# POST /chat endpoint structure — adjust if your route differs
CHAT_ENDPOINT    = "/chat"

# How long to wait between /chat calls to avoid rate-limiting the LLM
INTER_CALL_DELAY = 1.5   # seconds

# If /chat returns no citations, fall back to searching chunk text for
# the top-N chunks that contain keywords from the question
FALLBACK_KEYWORD_CHUNKS = 5

SEP  = "─" * 76
SEP2 = "═" * 76


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK LOOKUP  — pre-load once, look up by chunk_id throughout
# ══════════════════════════════════════════════════════════════════════════════

def load_chunk_index(path: str = CHUNKS_PATH) -> dict[str, str]:
    """
    Returns {chunk_id: full_text} from chunks_with_metadata.json.

    RAGAS needs the FULL chunk text for context_precision and context_recall.
    Using the 200-char snippet from the /chat citation will produce artificially
    low scores.  We do this look-up here so the retrieval API stays unchanged.
    """
    p = Path(path)
    if not p.exists():
        print(f"[WARN] {p} not found — context will be empty (RAGAS scores will be ~0).")
        return {}

    chunks = json.loads(p.read_text(encoding="utf-8"))
    index  = {c["chunk_id"]: c.get("text", "") for c in chunks if "chunk_id" in c}
    print(f"[INFO] Chunk index loaded: {len(index):,} chunks from {p}")
    return index


def resolve_contexts(
    cited_ids: list[str],
    chunk_index: dict[str, str],
    fallback_texts: list[str] | None = None,
) -> list[str]:
    """
    Resolve chunk_ids → full text strings for RAGAS contexts field.

    Strategy:
      1. Look up each cited chunk_id in the pre-loaded index.
      2. If a chunk_id is not found (truncated ID, old format), try prefix match.
      3. If no citations at all, fall back to texts returned directly by /chat.
    """
    if not cited_ids and fallback_texts:
        return [t for t in fallback_texts if t.strip()]

    contexts: list[str] = []
    for cid in cited_ids:
        # Exact match first
        if cid in chunk_index:
            contexts.append(chunk_index[cid])
            continue
        # Prefix match (some IDs are truncated to 16 chars)
        matches = [text for key, text in chunk_index.items() if key.startswith(cid)]
        if matches:
            contexts.append(matches[0])
        # If still not found, skip (RAGAS will note reduced context)

    return contexts if contexts else (fallback_texts or [])


# ══════════════════════════════════════════════════════════════════════════════
# CHAT API CALLER
# ══════════════════════════════════════════════════════════════════════════════

def build_api_base(host: str, port: int) -> str:
    host = host.strip().rstrip("/")
    if not host.startswith("http"):
        host = f"http://{host}"
    return f"{host}:{port}"


def call_chat(
    question: str,
    api_base: str,
    session_id: str = "eval_session",
    timeout: int = 300,
    verbose: bool = False,
) -> dict:
    """
    POST /chat and return the raw JSON response dict.

    Expected request body (adjust to match your ChatRequest schema):
      { "message": "...", "session_id": "..." }

    Expected response fields used here:
      answer          : str   — the generated answer
      citations       : list  — [{chunk_id, key, snippet, ...}]
      contexts        : list  — raw context strings (fallback if no citations)
      cache_hit       : bool
      kg_used         : bool
      hyde_used       : bool
      latency_ms      : float
      token_usage     : int   (or usage.total_tokens)

    Returns {} on failure (caller checks for missing keys).
    """
    url = f"{api_base}{CHAT_ENDPOINT}"
    payload = {
        "query":      question,
        "session_id": session_id,
    }
    if verbose:
        print(f"  → POST {url}")

    try:
        with httpx.Client(timeout=timeout) as client:
            t0   = time.perf_counter()
            resp = client.post(url, json=payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        resp.raise_for_status()
        data = resp.json()

        if verbose:
            ans_preview = str(data.get("answer", ""))[:120].replace("\n", " ")
            print(f"  ← {resp.status_code}  {elapsed_ms:.0f}ms  answer[:120]: {ans_preview}")

        # Attach latency if not already in response
        if "latency_ms" not in data:
            data["latency_ms"] = elapsed_ms

        return data

    except httpx.HTTPStatusError as exc:
        print(f"  [ERROR] HTTP {exc.response.status_code}: {exc.response.text[:200]}")
    except httpx.RequestError as exc:
        print(f"  [ERROR] Request failed: {exc}")
    except Exception as exc:
        print(f"  [ERROR] Unexpected: {exc}")

    return {}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD EvalRow FROM /chat RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# DROP-IN REPLACEMENTS for three functions in scripts/run_evaluation.py
#
# Problems fixed:
#   1. build_api_base()   — Windows rejects 0.0.0.0 as a connection target
#   2. load_chunk_index() — now returns by_id AND by_chapter dicts
#   3. build_eval_row()   — parses [Chapter X] inline citations from answer text
#                           and resolves them to full chunk text for RAGAS
#
# How to apply:
#   Replace the three functions in run_evaluation.py with the versions below.
#   Everything else in run_evaluation.py stays the same.
# ─────────────────────────────────────────────────────────────────────────────

import re
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 — build_api_base
# ══════════════════════════════════════════════════════════════════════════════

def build_api_base(host: str, port: int) -> str:
    """
    Build the base URL for /chat calls.

    FIX: On Windows, 0.0.0.0 is a server bind address — it cannot be used
    as a connection target and raises [WinError 10049].
    Replace it with 127.0.0.1 (localhost) when building the client URL.
    """
    host = host.strip().rstrip("/")

    # 0.0.0.0 means "listen on all interfaces" — translate to localhost for client
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"

    if not host.startswith("http"):
        host = f"http://{host}"

    return f"{host}:{port}"


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 — load_chunk_index
# ══════════════════════════════════════════════════════════════════════════════

def load_chunk_index(path: str = "data/processed/chunks_with_metadata.json") -> dict:
    """
    Pre-load chunks_with_metadata.json and build TWO lookup structures.
 
    Returns a dict with three keys:
      by_id[chunk_id]         → full chunk text string
      by_chapter[chapter_num] → list of chunk dicts (text + metadata)
      all                     → flat list of all chunks
 
    FIX: The original only built by_id. Since your /chat response uses inline
    [Chapter X] citations (not chunk_ids), we also need by_chapter so that
    build_eval_row() can find the right chunks for RAGAS contexts.
    """
    import json
    from pathlib import Path
 
    p = Path(path)
    if not p.exists():
        print(f"[WARN] {p} not found — context will be empty (RAGAS scores will be low).")
        return {"by_id": {}, "by_chapter": defaultdict(list), "all": []}
 
    chunks = json.loads(p.read_text(encoding="utf-8"))
 
    by_id: dict[str, str]          = {}
    by_chapter: dict[int, list]    = defaultdict(list)
 
    for c in chunks:
        # by_id index
        cid  = c.get("chunk_id", "")
        text = c.get("text", "").strip()
        if cid and text:
            by_id[cid] = text
 
        # by_chapter index — thesis chunks only
        if c.get("source_type") == "thesis":
            ch_num = c.get("chapter_num")
            if ch_num and text:
                by_chapter[int(ch_num)].append(c)
 
    total_thesis = sum(len(v) for v in by_chapter.values())
    print(
        f"[INFO] Chunk index loaded: {len(by_id):,} total chunks  |  "
        f"{total_thesis:,} thesis chunks across chapters "
        f"{sorted(by_chapter.keys())}"
    )
    return {"by_id": by_id, "by_chapter": by_chapter, "all": chunks}


# ══════════════════════════════════════════════════════════════════════════════
# FIX 3 — build_eval_row
# ══════════════════════════════════════════════════════════════════════════════

# How many chunks per chapter to include as RAGAS context.
# More = better recall score; fewer = better precision score.
# 3 is a good starting value.
TOP_CHUNKS_PER_CHAPTER = 10

# Maximum total context strings passed to RAGAS per question.
# Keeps token usage and latency under control.
MAX_TOTAL_CONTEXTS = 10


def _parse_chapter_refs(text: str) -> list[int]:
    """
    Extract chapter numbers from inline [Chapter X] citations.

    Handles formats:
      [Chapter 1]
      [Chapter 4]
      [Chapter 1, Chapter 3]
      [Chapters 1 and 4]
      [Ch. 3]

    Returns deduplicated list of ints in order of appearance.
    """
    # Match any number after "Chapter" or "Ch." inside square brackets
    pattern = r"\[(?:Chapter|Ch\.?)\s*(\d+)[^\]]*\]"
    matches = re.findall(pattern, text, re.IGNORECASE)

    seen   = []
    unique = []
    for m in matches:
        n = int(m)
        if n not in seen:
            seen.append(n)
            unique.append(n)
    return unique


def _score_chunks_for_question(
    chunks: list[dict],
    question: str,
) -> list[dict]:
    """
    Simple keyword overlap ranking — find which chunks in a chapter
    are most relevant to the question.

    Scoring: count how many unique question words appear in the chunk text.
    Returns chunks sorted by score descending.
    """
    # Remove common stopwords so short function words don't dominate
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in",
        "to", "for", "with", "how", "what", "which", "does", "do",
        "by", "at", "on", "from", "that", "this", "it", "as", "be",
        "and", "or", "not", "have", "has", "had", "been", "will",
        "would", "could", "should", "its", "their", "they", "we",
    }

    q_words = {
        w.lower().strip("?.,!:;\"'")
        for w in question.split()
        if w.lower() not in stopwords and len(w) > 2
    }

    scored = []
    for c in chunks:
        text  = c.get("text", "").lower()
        words = set(text.split())
        score = len(q_words & words)
        scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored]


def build_eval_row(
    question:    "EvalQuestion",
    response:    dict,
    chunk_index: dict,
) -> "EvalRow":
    """
    Build a complete EvalRow from the /chat response.

    FIX: Your /chat response uses inline [Chapter X] citations, not chunk_ids.
    This function now:
      1. Parses [Chapter X] references from the answer text
      2. Also checks source_chapters from the EvalQuestion as a supplement
      3. Retrieves the most relevant chunks from those chapters (keyword-ranked)
      4. Falls back to chunk_id lookup if your API ever does return them

    Context selection strategy:
      - Rank chunks within each cited chapter by keyword overlap with question
      - Take top TOP_CHUNKS_PER_CHAPTER from each chapter
      - Cap total at MAX_TOTAL_CONTEXTS to avoid RAGAS token overload
    """
    from app.models.evaluation import EvalRow   # local import avoids circular

    answer = (
        response.get("answer")
        or response.get("response")
        or response.get("text")
        or ""
    )

    by_id      = chunk_index.get("by_id", {})
    by_chapter = chunk_index.get("by_chapter", defaultdict(list))

    # ── Strategy A: chunk_id lookup (for future-proofing) ────────────────────
    cited_ids: list[str]     = []
    citation_keys: list[str] = []

    citations = response.get("citations") or []
    for c in citations:
        if isinstance(c, dict):
            cid = c.get("chunk_id") or c.get("id") or ""
            key = c.get("key") or c.get("citation_key") or ""
        elif isinstance(c, str):
            cid, key = c, c
        else:
            continue
        if cid:
            cited_ids.append(cid)
        if key:
            citation_keys.append(str(key))

    # flat list fallback
    if not cited_ids:
        flat = response.get("cited_chunk_ids") or response.get("chunk_ids") or []
        cited_ids = [str(x) for x in flat if x]

    contexts_from_ids: list[str] = []
    for cid in cited_ids:
        text = by_id.get(cid)
        if text:
            contexts_from_ids.append(text)
            continue
        # Prefix match (truncated IDs)
        matches = [t for k, t in by_id.items() if k.startswith(cid)]
        if matches:
            contexts_from_ids.append(matches[0])

    # ── Strategy B: inline [Chapter X] citation parsing ──────────────────────
    # This is the path your system currently uses
    chapter_refs = _parse_chapter_refs(answer)

    # Also include chapters from the EvalQuestion's source_chapters field
    # e.g. source_chapters = ["Chapter 3"] → add chapter 3
    for sc in (question.source_chapters or []):
        m = re.search(r"(\d+)", sc)
        if m:
            ch = int(m.group(1))
            if ch not in chapter_refs:
                chapter_refs.append(ch)

    contexts_from_chapters: list[str] = []
    for ch_num in chapter_refs:
        ch_chunks = by_chapter.get(ch_num, [])
        if not ch_chunks:
            print(
                f"  [WARN] No chunks found for Chapter {ch_num}. "
                "Check: python scripts/inspect_chunks.py --chapter "
                f"{ch_num} --stats"
            )
            continue

        # Rank chunks by relevance to the question
        ranked = _score_chunks_for_question(ch_chunks, question.question)

        # Take top N from this chapter
        for c in ranked[:TOP_CHUNKS_PER_CHAPTER]:
            text = c.get("text", "").strip()
            if text and text not in contexts_from_chapters:
                # Truncate very long chunks to avoid RAGAS context-window issues
                words = text.split()
                if len(words) > 700:
                    text = " ".join(words[:700]) + "…"
                contexts_from_chapters.append(text)

    # ── Merge strategies — prefer chunk_id hits, supplement with chapter hits ─
    # chunk_id contexts first (more precise), then chapter contexts (broader)
    seen     = set()
    contexts = []
    for text in contexts_from_ids + contexts_from_chapters:
        key = text[:80]   # use first 80 chars as dedup key
        if key not in seen:
            seen.add(key)
            contexts.append(text)
        if len(contexts) >= MAX_TOTAL_CONTEXTS:
            break

    # ── Last-resort fallback: raw context strings from response ───────────────
    if not contexts:
        raw = response.get("contexts") or []
        if isinstance(raw, str):
            raw = [raw]
        contexts = [t for t in raw if t.strip()][:MAX_TOTAL_CONTEXTS]

    # ── Token usage ───────────────────────────────────────────────────────────
    token_usage = (
        response.get("token_usage")
        or response.get("tokens_used")
        or (response.get("usage") or {}).get("total_tokens")
        or 0
    )

    return EvalRow(
        question_id    = question.question_id,
        numeric_id     = question.id,
        question       = question.question,
        answer         = answer,
        contexts       = contexts,
        ground_truth   = question.ground_truth,
        chapter        = question.chapter,
        topic          = question.topic,
        query_type     = question.query_type,
        source_chapters= question.source_chapters,
        cited_chunk_ids= cited_ids,
        citation_keys  = citation_keys,
        cache_hit      = bool(response.get("cache_hit", False)),
        kg_used        = bool(response.get("kg_used", False)),
        hyde_used      = bool(response.get("hyde_used", False)),
        latency_ms     = response.get("latency_ms"),
        token_usage    = int(token_usage),
    )

# ══════════════════════════════════════════════════════════════════════════════
# RAGAS SETUP  — Gemini as judge LLM
# ══════════════════════════════════════════════════════════════════════════════

def build_ragas_evaluator():
    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        print("[WARN] GROQ_API_KEY missing")
        return []

    try:
        # IMPORTANT
        os.environ["OPENAI_API_KEY"] = groq_key
        os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"

        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )

        from ragas.llms import LangchainLLMWrapper
        from langchain_groq import ChatGroq

        judge_llm = LangchainLLMWrapper(
            ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                api_key=groq_key,
            )
        )

        metrics = [
            Faithfulness(llm=judge_llm),
            ContextPrecision(llm=judge_llm),
            ContextRecall(llm=judge_llm),
        ]

        print("[INFO] RAGAS evaluator configured with explicit Groq judge")

        return metrics

    except Exception as exc:
        print(f"[WARN] RAGAS evaluator setup failed: {exc}")
        import traceback
        traceback.print_exc()
        return []

# ══════════════════════════════════════════════════════════════════════════════
# RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def save_eval_checkpoint(
    scores: dict[str, list],
    completed_up_to: int,
    run_id: str,
    path: str = EVAL_CHECKPOINT_PATH,
) -> None:
    """Save evaluation progress after each batch."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id":           run_id,
        "completed_up_to":  completed_up_to,   # index of last completed row
        "scores":           scores,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
 
 
def load_eval_checkpoint(
    run_id: str,
    path: str = EVAL_CHECKPOINT_PATH,
) -> tuple[dict | None, int]:
    """
    Load existing checkpoint if it matches the current run_id.
 
    Returns:
      (scores_dict, completed_up_to)  if checkpoint exists and matches run_id
      (None, 0)                       if no checkpoint or different run_id
    """
    p = Path(path)
    if not p.exists():
        return None, 0
 
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None, 0
 
    if data.get("run_id") != run_id:
        print(f"  [INFO] Checkpoint found but belongs to run '{data.get('run_id')}' "
              f"— current run is '{run_id}'. Starting fresh.")
        return None, 0
 
    completed = data.get("completed_up_to", 0)
    scores    = data.get("scores", {})
    print(f"  [INFO] Resuming from checkpoint: {completed} rows already scored.")
    return scores, completed

def run_ragas(
    eval_rows: list,
    metrics: list,
    batch_size: int = 5,
    run_id: str = "",
) -> dict[str, list]:
    """
    Run ragas.evaluate() with checkpoint/resume support.
 
    If interrupted mid-run, re-run the same command and it will
    skip already-scored batches and continue from where it left off.
 
    Checkpoint is deleted automatically on successful completion.
    """
    from ragas import evaluate as ragas_evaluate
 
    n = len(eval_rows)
 
    # ── Try to resume from checkpoint ────────────────────────────────────────
    existing_scores, resume_from = load_eval_checkpoint(run_id)
 
    if existing_scores:
        scores = existing_scores
        # Ensure all score lists are the right length
        for key in ["faithfulness", "context_precision", "context_recall"]:
            if key not in scores or len(scores[key]) != n:
                scores[key] = (scores.get(key, []) + [None] * n)[:n]
    else:
        scores = {
            "faithfulness":      [None] * n,
            "context_precision": [None] * n,
            "context_recall":    [None] * n,
        }
 
    print(f"\n[INFO] Running RAGAS on {n} rows in batches of {batch_size} …")
    if resume_from > 0:
        print(f"[INFO] Skipping first {resume_from} rows (already scored).")
 
    for start in range(0, n, batch_size):
        # Skip batches already completed
        if start + batch_size <= resume_from:
            print(f"  Batch [{start+1}–{start+batch_size}] — skipped (already scored)")
            continue
        # Partial resume: if checkpoint is mid-batch, redo from batch start
        # (safe because RAGAS is deterministic at temperature=0)
 
        batch         = eval_rows[start : start + batch_size]
        batch_indices = list(range(start, start + len(batch)))
 
        valid_rows   = [(i, row) for i, row in zip(batch_indices, batch) if row.answer.strip()]
        invalid_rows = [(i, row) for i, row in zip(batch_indices, batch) if not row.answer.strip()]
 
        for i, row in invalid_rows:
            print(f"  [SKIP] {row.question_id} — empty answer")
 
        if not valid_rows:
            continue
 
        ragas_dicts = [row.to_ragas_dict() for _, row in valid_rows]

        for rd in ragas_dicts:
            rd["question"] = str(rd.get("question", ""))[:4000]

            rd["answer"] = str(rd.get("answer", ""))[:7000]

            rd["ground_truth"] = str(rd.get("ground_truth", ""))[:7000]

            rd["contexts"] = [
                str(c)[:2500]
                for c in rd.get("contexts", [])[:8]
                if c is not None and str(c).strip()
            ]
            if not rd["contexts"]:
                rd["contexts"] = ["No relevant context retrieved."]

        try:
            ds     = Dataset.from_list(ragas_dicts)
            result = ragas_evaluate(
                ds,
                metrics=metrics,
                run_config=RunConfig(
                    timeout=300,
                    max_workers=1,
                    max_wait=30,
                    max_retries=3,
                ),
            )
            df     = result.to_pandas()
 
            for local_idx, (global_idx, _) in enumerate(valid_rows):
                for metric in scores:
                    if metric in df.columns:
                        val = df.iloc[local_idx][metric]
                        scores[metric][global_idx] = (
                            float(val) if val is not None and str(val) != "nan" else None
                        )
 
            batch_ids = [row.question_id for _, row in valid_rows]
            print(f"  Batch [{start+1}–{start+len(batch)}] done  ({', '.join(batch_ids)})")
 
            # ── Save checkpoint after every successful batch ──────────────────
            save_eval_checkpoint(scores, start + len(batch), run_id)
 
        except Exception as exc:
            print(f"  [ERROR] RAGAS batch [{start}–{start+len(batch)-1}] failed: {exc}")
            print(f"  [INFO]  Progress saved to checkpoint. "
                  f"Re-run the same command to resume from batch {start+1}.")
 
            # Individual fallback
            for global_idx, row in valid_rows:
                try:
                    rd = row.to_ragas_dict()
                    if not rd["contexts"]:
                        rd["contexts"] = ["[No context retrieved]"]
                    ds_single     = Dataset.from_list([rd])
                    result_single = ragas_evaluate(
                        ds_single,
                        metrics=metrics,
                        run_config=RunConfig(
                            timeout=300,
                            max_workers=1,
                            max_wait=30,
                            max_retries=3,
                        ),
                    )
                    df_single     = result_single.to_pandas()
                    for metric in scores:
                        if metric in df_single.columns:
                            val = df_single.iloc[0][metric]
                            scores[metric][global_idx] = (
                                float(val) if val is not None and str(val) != "nan" else None
                            )
                    print(f"    ✓ Recovered {row.question_id} individually")
                except Exception as inner_exc:
                    print(f"    ✗ {row.question_id} failed individually: {inner_exc}")
 
            # Save whatever we managed to score before failing
            save_eval_checkpoint(scores, start, run_id)
 
        if start + batch_size < n:
            time.sleep(2.0)
 
    # ── Delete checkpoint on successful full completion ───────────────────────
    cp = Path(EVAL_CHECKPOINT_PATH)
    if cp.exists():
        cp.unlink()
        print(f"\n  [INFO] Checkpoint deleted (run complete).")
 
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# REPORT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_report(
    eval_rows: list[EvalRow],
    score_dict: dict[str, list],
    run_id: str,
) -> EvalReport:
    """Assemble RAGASResult objects and EvalReport from raw score lists."""

    results: list[RAGASResult] = []

    for i, row in enumerate(eval_rows):
        r = RAGASResult(
            question_id       = row.question_id,
            numeric_id        = row.numeric_id,
            question          = row.question,
            chapter           = row.chapter,
            topic             = row.topic,
            query_type        = row.query_type,
            faithfulness      = score_dict["faithfulness"][i],
            answer_relevancy = (
                score_dict["answer_relevancy"][i]
                if "answer_relevancy" in score_dict
                else None
            ),
            context_precision = score_dict["context_precision"][i],
            context_recall    = score_dict["context_recall"][i],
            latency_ms        = row.latency_ms,
            token_usage       = row.token_usage,
            cache_hit         = row.cache_hit,
            kg_used           = row.kg_used,
            context_count     = row.context_count,
            # Mark as errored if no answer was returned
            error = "Empty answer from /chat" if not row.answer.strip() else None,
        )
        results.append(r)

    successful   = sum(1 for r in results if r.error is None and r.mean_score is not None)
    failed       = len(results) - successful
    passed_count = sum(1 for r in results if r.passed)

    def _safe_avg(metric: str) -> Optional[float]:
        vals = [getattr(r, metric) for r in results if getattr(r, metric) is not None]
        return round(mean(vals), 4) if vals else None

    latencies = [r.latency_ms for r in results if r.latency_ms is not None]

    report = EvalReport(
        run_id           = run_id,
        total_questions  = len(results),
        successful       = successful,
        failed           = failed,
        passed_count     = passed_count,
        results          = results,

        # Pipeline config snapshot
        llm_model_main   = settings.llm_model_main,
        llm_model_router = settings.llm_model_router,
        embedding_model  = settings.embedding_model,
        reranker_model   = settings.reranker_model,
        qdrant_top_k     = settings.qdrant_top_k,
        reranker_top_k   = settings.reranker_top_k_output,
        rrf_vector_weight= settings.rrf_vector_weight,
        hyde_alpha       = settings.hyde_alpha,

        # Aggregates
        avg_faithfulness      = _safe_avg("faithfulness"),
        avg_answer_relevancy  = _safe_avg("answer_relevancy"),
        avg_context_precision = _safe_avg("context_precision"),
        avg_context_recall    = _safe_avg("context_recall"),
        avg_mean_score        = _safe_avg("mean_score"),
        avg_latency_ms        = round(mean(latencies), 1) if latencies else None,
    )

    return report


# ══════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(val: Optional[float], threshold: float) -> str:
    """Colour-code a score: green if above threshold, red if below."""
    if val is None:
        return "  N/A "
    flag = "✓" if val >= threshold else "✗"
    return f"{flag} {val:.3f}"


def print_per_question_table(report: EvalReport) -> None:
    print(f"\n{SEP2}")
    print("  PER-QUESTION RESULTS")
    print(SEP2)
    header = (
        f"{'QID':<6}  {'Type':<12}  {'Faith':>7}  {'Relev':>7}  "
        f"{'Prec':>7}  {'Rec':>7}  {'Mean':>7}  {'ms':>6}"
    )
    print(header)
    print(SEP)

    for r in sorted(report.results, key=lambda x: x.numeric_id):
        ms   = f"{r.mean_score:.3f}" if r.mean_score is not None else "  N/A"
        lat  = f"{r.latency_ms:.0f}" if r.latency_ms else "  —"
        flag = "✓" if r.passed else "✗"
        print(
            f"{r.question_id:<6}  {r.query_type or '?':<12}  "
            f"{_fmt(r.faithfulness,      THRESHOLDS['faithfulness']):>7}  "
            f"{_fmt(r.answer_relevancy,  THRESHOLDS['answer_relevancy']):>7}  "
            f"{_fmt(r.context_precision, THRESHOLDS['context_precision']):>7}  "
            f"{_fmt(r.context_recall,    THRESHOLDS['context_recall']):>7}  "
            f"{flag} {ms}  {lat:>6}"
        )


def print_summary(report: EvalReport) -> None:
    print(f"\n{SEP2}")
    print("  OVERALL SUMMARY")
    print(SEP2)
    d = report.summary_dict()
    print(f"  Run ID          : {d['run_id']}")
    print(f"  Questions       : {d['total_questions']}  (successful: {d['successful']})")
    print(f"  Pass rate       : {report.pass_rate:.1%}  ({report.passed_count}/{report.total_questions} questions passed all 4 thresholds)")
    print(f"  Avg latency     : {d['avg_latency_ms']:.0f} ms" if d["avg_latency_ms"] else "  Avg latency     : N/A")
    print()
    print(f"  {'Metric':<22}  {'Score':>7}  {'Target':>7}  {'Status':>6}")
    print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*6}")

    def _status_line(name: str, val: Optional[float], thresh: float) -> None:
        score  = f"{val:.4f}" if val else "  N/A "
        status = ("✓ PASS" if val and val >= thresh else "✗ FAIL") if val else "  N/A "
        print(f"  {name:<22}  {score:>7}  {thresh:>7.2f}  {status:>6}")

    _status_line("Faithfulness",      report.avg_faithfulness,      THRESHOLDS["faithfulness"])
    _status_line("Answer Relevancy",  report.avg_answer_relevancy,  THRESHOLDS["answer_relevancy"])
    _status_line("Context Precision", report.avg_context_precision, THRESHOLDS["context_precision"])
    _status_line("Context Recall",    report.avg_context_recall,    THRESHOLDS["context_recall"])

    if report.avg_mean_score:
        print(f"\n  {'Mean (all 4)':<22}  {report.avg_mean_score:>7.4f}")

    # Model config snapshot
    print(f"\n  Pipeline: main={d['llm_model_main']}  top_k={d['qdrant_top_k']}→{d['reranker_top_k']}")


def print_query_type_breakdown(report: EvalReport) -> None:
    breakdown = report.by_query_type()
    if not breakdown:
        return

    print(f"\n{SEP2}")
    print("  BREAKDOWN BY QUERY TYPE")
    print(SEP2)
    print(f"  {'Type':<14}  {'N':>3}  {'Pass':>4}  {'Faith':>6}  {'Relev':>6}  {'Prec':>6}  {'Rec':>6}  {'Mean':>6}")
    print(f"  {'─'*14}  {'─'*3}  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")

    for qt, row in breakdown.items():
        def _s(key: str) -> str:
            v = row.get(key)
            return f"{v:.3f}" if v else "  N/A"
        print(
            f"  {qt:<14}  {row['count']:>3}  {row['pass_count']:>4}  "
            f"{_s('faithfulness'):>6}  {_s('answer_relevancy'):>6}  "
            f"{_s('context_precision'):>6}  {_s('context_recall'):>6}  "
            f"{_s('mean_score'):>6}"
        )


def print_chapter_breakdown(report: EvalReport) -> None:
    breakdown = report.by_chapter()
    if not breakdown:
        return

    print(f"\n{SEP2}")
    print("  BREAKDOWN BY CHAPTER")
    print(SEP2)
    for ch, row in breakdown.items():
        def _s(key: str) -> str:
            v = row.get(key)
            return f"{v:.3f}" if v else "N/A"
        print(
            f"  {ch[:45]:<45}  N={row['count']:>2}  "
            f"F={_s('faithfulness')}  R={_s('answer_relevancy')}  "
            f"P={_s('context_precision')}  C={_s('context_recall')}"
        )


def print_weakest(report: EvalReport, n: int = 5) -> None:
    weakest = report.weakest_questions(n)
    if not weakest:
        return

    print(f"\n{SEP2}")
    print(f"  WEAKEST {n} QUESTIONS (targets for improvement)")
    print(SEP2)
    for r in weakest:
        failing = ", ".join(r.failing_metrics) or "none"
        ms = f"{r.mean_score:.3f}" if r.mean_score else "N/A"
        print(f"  {r.question_id}  mean={ms}  failing=[{failing}]")
        print(f"        {r.question[:80]}")
        print()


def print_actionable_hints(report: EvalReport) -> None:
    """Print targeted improvement hints based on which metric is lowest."""
    metrics_avg = {
        "faithfulness":      report.avg_faithfulness,
        "answer_relevancy":  report.avg_answer_relevancy,
        "context_precision": report.avg_context_precision,
        "context_recall":    report.avg_context_recall,
    }

    failing = {
        m: v for m, v in metrics_avg.items()
        if v is not None and v < THRESHOLDS[m]
    }

    if not failing:
        print(f"\n  ✓ All metrics above target thresholds. System is performing well.")
        return

    print(f"\n{SEP2}")
    print("  ACTIONABLE HINTS FOR FAILING METRICS")
    print(SEP2)

    hints = {
        "faithfulness": [
            "Lower llm_temperature (currently {:.1f}) → more conservative generation".format(settings.llm_temperature),
            "Strengthen system prompt rule: 'Answer ONLY from the provided context'",
            "Reduce reranker_top_k_output — fewer chunks = less hallucination surface",
        ],
        "answer_relevancy": [
            "Tune query router — check if query_type classification is correct",
            "Improve HyDE prompt — hypothetical doc should match answer format",
            f"Current hyde_alpha={settings.hyde_alpha} — try 0.6 (more query weight)",
        ],
        "context_precision": [
            f"Increase rrf_vector_weight (now {settings.rrf_vector_weight}) → reward semantic matches",
            "Add stricter metadata filters (chapter_num, source_type) in retrieval",
            f"Lower reranker_top_k_input (now {settings.reranker_top_k_input}) — fewer noisy candidates",
        ],
        "context_recall": [
            f"Increase qdrant_top_k (now {settings.qdrant_top_k}) — retrieve more candidates",
            f"Increase bm25_top_k (now {settings.bm25_top_k}) — wider BM25 net",
            "Check chunk coverage with: python scripts/inspect_chunks.py --stats",
            "Short chunks (avg 45 tokens) may split concepts — consider larger chunk sizes",
        ],
    }

    for metric, score in sorted(failing.items(), key=lambda x: x[1]):
        thresh = THRESHOLDS[metric]
        gap    = thresh - score
        print(f"\n  ✗ {metric}  score={score:.3f}  target={thresh:.2f}  gap={gap:.3f}")
        for hint in hints.get(metric, []):
            print(f"      • {hint}")


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the Thesis RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--questions",
        default=QUESTIONS_PATH,
        help=f"Path to test_questions.json (default: {QUESTIONS_PATH})",
    )
    p.add_argument(
        "--chunks",
        default=CHUNKS_PATH,
        help=f"Path to chunks_with_metadata.json (default: {CHUNKS_PATH})",
    )
    p.add_argument(
        "--api-base",
        default=None,
        help="Base URL of the running FastAPI server (default: from settings)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N questions (reproducible seed=21)",
    )
    p.add_argument(
        "--query-type",
        type=str,
        default=None,
        help="Only evaluate one query_type (factual|comparative|synthesis|methodology|equation|algorithm)",
    )
    p.add_argument(
        "--chapter",
        type=int,
        default=None,
        help="Only evaluate questions from this chapter number (1-6)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="RAGAS batch size (default: 5) — smaller = less memory / rate-limit risk",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=INTER_CALL_DELAY,
        help=f"Seconds between /chat calls (default: {INTER_CALL_DELAY})",
    )
    p.add_argument(
        "--output-dir",
        default=RESULTS_DIR,
        help=f"Directory for JSON/CSV output (default: {RESULTS_DIR})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Call /chat but skip RAGAS evaluation (tests connectivity only)",
    )
    p.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Same as --dry-run (alias)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print each question + answer preview as it runs",
    )
    p.add_argument(
        "--no-per-question",
        action="store_true",
        help="Skip the per-question table in terminal output",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Run ID ────────────────────────────────────────────────────────────────
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    print(f"\n{SEP2}")
    print(f"  THESIS RAG — RAGAS EVALUATION  [{run_id}]")
    print(SEP2)

    # ── API base URL ──────────────────────────────────────────────────────────
    api_base = args.api_base or build_api_base(settings.app_host, settings.app_port)
    print(f"  API endpoint : {api_base}{CHAT_ENDPOINT}")

    # ── Load questions ────────────────────────────────────────────────────────
    print(f"  Questions    : {args.questions}")
    dataset = EvalDataset.from_json(args.questions)
    questions: list[EvalQuestion] = dataset.questions

    # Filters
    if args.chapter is not None:
        questions = [q for q in questions if q.chapter_num == args.chapter]
        print(f"  Filter       : Chapter {args.chapter}  →  {len(questions)} questions")

    if args.query_type:
        questions = [q for q in questions if q.query_type == args.query_type]
        print(f"  Filter       : query_type={args.query_type}  →  {len(questions)} questions")

    if args.sample and args.sample < len(questions):
        questions = dataset.sample(args.sample) if not (args.chapter or args.query_type) \
                    else __import__("random").Random(21).sample(questions, args.sample)
        print(f"  Sample       : {len(questions)} questions (seed=21)")

    if not questions:
        print("[ERROR] No questions to evaluate after filters. Check your --chapter / --query-type flags.")
        sys.exit(1)

    print(f"  Evaluating   : {len(questions)} questions\n")

    # ── Load chunk index ──────────────────────────────────────────────────────
    chunk_index = load_chunk_index(args.chunks)

    # ── Call POST /chat for each question ─────────────────────────────────────
    print(f"\n{SEP}")
    print("  PHASE 1 — Collecting answers from POST /chat")
    print(SEP)

    eval_rows: list[EvalRow] = []
    session_id = f"eval_{run_id}"

    for q in tqdm(questions, desc="Calling /chat", unit="q"):
        if args.verbose:
            print(f"\n[{q.question_id}] {q.question[:90]}")

        response = call_chat(
            question   = q.question,
            api_base   = api_base,
            session_id = session_id,
            verbose    = args.verbose,
        )

        row = build_eval_row(q, response, chunk_index)
        eval_rows.append(row)

        if args.verbose:
            print(f"  contexts={row.context_count}  cache={row.cache_hit}  kg={row.kg_used}  hyde={row.hyde_used}")

        time.sleep(args.delay)

    # Summary of phase 1
    answered  = sum(1 for r in eval_rows if r.answer.strip())
    no_answer = len(eval_rows) - answered
    no_ctx    = sum(1 for r in eval_rows if not r.contexts)

    print(f"\n  Answered       : {answered}/{len(eval_rows)}")
    if no_answer:
        print(f"  ⚠ Empty answers: {no_answer}  (check server is running at {api_base})")
    if no_ctx:
        print(f"  ⚠ No contexts  : {no_ctx}  (check chunk_id format in /chat response)")

    if args.dry_run or args.skip_ragas:
        print("\n  --dry-run set: skipping RAGAS evaluation.")
        print("  /chat connectivity test complete.")
        return

    if answered == 0:
        print(f"\n[ERROR] All /chat calls failed. Check server at {api_base}")
        sys.exit(1)

    # ── RAGAS evaluation ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PHASE 2 — RAGAS evaluation")
    print(SEP)

    metrics    = build_ragas_evaluator()
    score_dict = run_ragas(eval_rows, metrics, batch_size=args.batch_size, run_id=run_id) 

    # ── Build report ──────────────────────────────────────────────────────────
    report = build_report(eval_rows, score_dict, run_id)

    # ── Save results ──────────────────────────────────────────────────────────
    json_path, csv_path = report.save(args.output_dir)
    print(f"\n  Results saved:")
    print(f"    JSON → {json_path}")
    print(f"    CSV  → {csv_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    if not args.no_per_question:
        print_per_question_table(report)

    print_summary(report)
    print_query_type_breakdown(report)
    print_chapter_breakdown(report)
    print_weakest(report, n=5)
    print_actionable_hints(report)

    print(f"\n{SEP2}\n")


if __name__ == "__main__":
    main()