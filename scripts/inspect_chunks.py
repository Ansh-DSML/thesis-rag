"""
scripts/inspect_chunks.py

Debug utility — inspect what is ACTUALLY in your indexed chunks.
Run this BEFORE running run_evaluation.py to verify ingestion quality.
If chunks look wrong here, RAGAS scores will be low for the right reason.

Data source: data/processed/chunks_with_metadata.json
(same data that is indexed in Qdrant — no network call needed)

Usage examples:
  # Overall collection statistics
  python scripts/inspect_chunks.py --stats

  # Show first 10 chunks from Chapter 3
  python scripts/inspect_chunks.py --chapter 3 --limit 10

  # Show all section headings in a chapter (structure overview)
  python scripts/inspect_chunks.py --chapter 3 --headings-only

  # Find chunks containing specific text
  python scripts/inspect_chunks.py --query "PSO" --limit 10

  # Find chunks where a specific entity was tagged
  python scripts/inspect_chunks.py --entity "PSO"

  # Show chunks from a specific paper
  python scripts/inspect_chunks.py --paper "sensors-25-04337.pdf"

  # Inspect one specific chunk by ID
  python scripts/inspect_chunks.py --chunk-id 9e9f108ca4fd2373061c

  # Show chunks with suspiciously short text (potential chunking issues)
  python scripts/inspect_chunks.py --short --limit 20

  # Show all paper titles indexed
  python scripts/inspect_chunks.py --papers
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
CHUNKS_PATH = "data/processed/chunks_with_metadata.json"
CHAPTER_NAMES = {
    1: "Introduction",
    2: "Literature Review",
    3: "Methodology",
    4: "Results",
    5: "Discussion",
    6: "Conclusion",
}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_chunks(path: str = CHUNKS_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] {p} not found. Run ingestion first.")
        sys.exit(1)
    chunks = json.loads(p.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {p}\n")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

SEP  = "─" * 72
SEP2 = "═" * 72

def _clean(text: str) -> str:
    """Fix common PDF encoding artefacts for display."""
    return (
        text.replace("\u00e2\u0080\u00a2", "•")
            .replace("â€˜", "'")
            .replace("â€™", "'")
            .replace("â€œ", '"')
            .replace("â€", '"')
            .replace("\ufffd", "?")
    )

def print_chunk(c: dict, idx: int, show_full: bool = False) -> None:
    print(SEP)
    src = c.get("source_type", "?")
    if src == "thesis":
        ch   = c.get("chapter_num", "?")
        name = c.get("chapter_name", "")
        head = c.get("section_heading", "")
        page = c.get("page_start", "?")
        print(f"[{idx}] Chapter {ch}: {name}  |  Section: {head}  |  Page: {page}")
    else:
        title  = c.get("paper_title") or c.get("paper_filename", "Unknown paper")
        authors = c.get("paper_authors") or []
        year   = c.get("paper_year", "")
        head   = c.get("section_heading", "")
        print(f"[{idx}] {title[:60]}  ({', '.join(authors[:2])}, {year})")
        if head:
            print(f"      Section: {head}")

    entities = c.get("entities", [])
    print(f"      chunk_id={c.get('chunk_id','?')[:16]}  "
          f"words={c.get('word_count',0)}  "
          f"tokens={c.get('token_count',0)}  "
          f"entities=[{', '.join(entities[:5])}{'...' if len(entities)>5 else ''}]")

    text = _clean(c.get("text", ""))
    if show_full:
        print(f"\n{text}\n")
    else:
        preview = text[:300].replace("\n", " ")
        if len(text) > 300:
            preview += "…"
        print(f"\n  {preview}\n")


# ══════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_stats(chunks: list[dict]) -> None:
    """Overall collection statistics."""
    thesis = [c for c in chunks if c.get("source_type") == "thesis"]
    papers = [c for c in chunks if c.get("source_type") == "paper"]

    print(SEP2)
    print("  COLLECTION STATISTICS")
    print(SEP2)
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Thesis chunks: {len(thesis)}")
    print(f"  Paper chunks : {len(papers)}")

    if thesis:
        by_chapter = Counter(c.get("chapter_num") for c in thesis)
        print("\n  Thesis chunks by chapter:")
        for num in sorted(by_chapter):
            name = CHAPTER_NAMES.get(num, "Unknown")
            bar  = "█" * (by_chapter[num] // 5)
            print(f"    Ch{num} {name:<20} {by_chapter[num]:>4}  {bar}")

    if papers:
        paper_titles = Counter(
            (c.get("paper_title") or c.get("paper_filename", "unknown"))
            for c in papers
        )
        print(f"\n  Papers indexed: {len(paper_titles)}")
        for title, count in sorted(paper_titles.items(), key=lambda x: -x[1])[:10]:
            short = title[:55]
            print(f"    {count:>4} chunks  {short}")

    # Token distribution
    tokens = [c.get("token_count", 0) for c in chunks if c.get("token_count", 0) > 0]
    if tokens:
        avg_tok = sum(tokens) / len(tokens)
        min_tok = min(tokens)
        max_tok = max(tokens)
        print(f"\n  Token counts: min={min_tok}  avg={avg_tok:.0f}  max={max_tok}")
        short_count = sum(1 for t in tokens if t < 50)
        if short_count:
            print(f"  ⚠ {short_count} chunks with < 50 tokens (possible chunking issues)")

    # Entity coverage
    with_entities = sum(1 for c in chunks if c.get("entities"))
    print(f"\n  Chunks with entities: {with_entities}/{len(chunks)} "
          f"({with_entities/len(chunks)*100:.1f}%)")

    # Top entities
    all_entities: list[str] = []
    for c in chunks:
        all_entities.extend(c.get("entities", []))
    top = Counter(all_entities).most_common(15)
    if top:
        print("\n  Top 15 entities by frequency:")
        for ent, freq in top:
            print(f"    {freq:>5}x  {ent}")

    print(SEP2)


def cmd_chapter(chunks: list[dict], chapter: int, limit: int, headings_only: bool, full: bool) -> None:
    """Show chunks from a specific thesis chapter."""
    filtered = [c for c in chunks
                if c.get("source_type") == "thesis" and c.get("chapter_num") == chapter]

    name = CHAPTER_NAMES.get(chapter, "Unknown")
    print(f"Chapter {chapter}: {name}  →  {len(filtered)} chunks total")

    if headings_only:
        headings = {}
        for c in filtered:
            h = c.get("section_heading", "(no heading)")
            if h not in headings:
                headings[h] = 0
            headings[h] += 1
        print(f"\nSection headings ({len(headings)} unique):\n")
        for h, count in headings.items():
            print(f"  [{count:>3} chunks]  {h[:70]}")
        return

    shown = filtered[:limit]
    print(f"Showing first {len(shown)} of {len(filtered)}:\n")
    for i, c in enumerate(shown, 1):
        print_chunk(c, i, show_full=full)


def cmd_query(chunks: list[dict], query: str, limit: int, full: bool) -> None:
    """Find chunks containing all query words (case-insensitive)."""
    q_words = query.lower().split()

    def _matches(c: dict) -> bool:
        text = c.get("text", "").lower()
        return all(w in text for w in q_words)

    filtered = [c for c in chunks if _matches(c)]
    print(f"Query: '{query}'  →  {len(filtered)} matching chunks")

    shown = filtered[:limit]
    print(f"Showing first {len(shown)}:\n")
    for i, c in enumerate(shown, 1):
        print_chunk(c, i, show_full=full)


def cmd_entity(chunks: list[dict], entity: str, limit: int, full: bool) -> None:
    """Find chunks where spaCy tagged a specific entity."""
    ent_lower = entity.lower()
    filtered = [
        c for c in chunks
        if any(e.lower() == ent_lower or ent_lower in e.lower()
               for e in c.get("entities", []))
    ]
    print(f"Entity: '{entity}'  →  {len(filtered)} chunks tagged with this entity")

    shown = filtered[:limit]
    print(f"Showing first {len(shown)}:\n")
    for i, c in enumerate(shown, 1):
        print_chunk(c, i, show_full=full)


def cmd_paper(chunks: list[dict], paper: str, limit: int, full: bool) -> None:
    """Show chunks from a specific paper (partial filename match)."""
    paper_lower = paper.lower()
    filtered = [
        c for c in chunks
        if c.get("source_type") == "paper"
        and paper_lower in (c.get("paper_filename") or c.get("paper_title") or "").lower()
    ]
    print(f"Paper: '{paper}'  →  {len(filtered)} chunks")

    if filtered:
        sample = filtered[0]
        print(f"  Title:   {sample.get('paper_title', 'N/A')}")
        print(f"  Authors: {', '.join(sample.get('paper_authors', [])[:3])}")
        print(f"  Year:    {sample.get('paper_year', 'N/A')}")
        print()

    shown = filtered[:limit]
    print(f"Showing first {len(shown)}:\n")
    for i, c in enumerate(shown, 1):
        print_chunk(c, i, show_full=full)


def cmd_chunk_id(chunks: list[dict], chunk_id: str) -> None:
    """Inspect one specific chunk by ID (full content)."""
    matches = [c for c in chunks if c.get("chunk_id", "").startswith(chunk_id)]
    if not matches:
        print(f"No chunk found with ID starting with '{chunk_id}'")
        return
    for c in matches:
        print_chunk(c, 1, show_full=True)
        print("Full metadata:")
        for k, v in c.items():
            if k != "text":
                print(f"  {k:<25}: {v}")


def cmd_short(chunks: list[dict], threshold: int, limit: int) -> None:
    """Show chunks with suspiciously short text (possible chunking issues)."""
    short = [c for c in chunks if c.get("word_count", 0) < threshold]
    print(f"Chunks with < {threshold} words: {len(short)}/{len(chunks)}")

    shown = short[:limit]
    print(f"Showing first {len(shown)}:\n")
    for i, c in enumerate(shown, 1):
        print_chunk(c, i, show_full=True)


def cmd_papers(chunks: list[dict]) -> None:
    """List all papers indexed with chunk counts."""
    paper_chunks = [c for c in chunks if c.get("source_type") == "paper"]
    papers: dict[str, dict] = {}

    for c in paper_chunks:
        key = c.get("paper_filename") or c.get("paper_title") or "unknown"
        if key not in papers:
            papers[key] = {
                "title":   c.get("paper_title", "N/A"),
                "authors": c.get("paper_authors", []),
                "year":    c.get("paper_year"),
                "count":   0,
            }
        papers[key]["count"] += 1

    print(f"{len(papers)} papers indexed:\n")
    for filename, info in sorted(papers.items()):
        authors = ", ".join(info["authors"][:2])
        if len(info["authors"]) > 2:
            authors += " et al."
        print(f"  [{info['count']:>3} chunks]  {filename}")
        if info["title"] and info["title"] != filename:
            print(f"             {info['title'][:65]}")
        if authors:
            print(f"             {authors} ({info['year'] or 'n.d.'})")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect indexed chunks for debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stats",       action="store_true", help="Show collection statistics")
    p.add_argument("--chapter",     type=int,            help="Chapter number (1-6)")
    p.add_argument("--query",       type=str,            help="Text search (all words must match)")
    p.add_argument("--entity",      type=str,            help="Find chunks tagged with this entity")
    p.add_argument("--paper",       type=str,            help="Paper filename or title (partial match)")
    p.add_argument("--papers",      action="store_true", help="List all indexed papers")
    p.add_argument("--chunk-id",    type=str,            help="Inspect specific chunk by ID prefix")
    p.add_argument("--short",       action="store_true", help="Show very short chunks (< 20 words)")
    p.add_argument("--headings-only", action="store_true", help="With --chapter: show section headings only")
    p.add_argument("--full",        action="store_true", help="Show full chunk text (not just preview)")
    p.add_argument("--limit",       type=int, default=10, help="Max chunks to display (default: 10)")
    p.add_argument("--chunks-path", type=str, default=CHUNKS_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_chunks(args.chunks_path)

    if args.stats:
        cmd_stats(chunks)
    elif args.chapter is not None:
        cmd_chapter(chunks, args.chapter, args.limit, args.headings_only, args.full)
    elif args.query:
        cmd_query(chunks, args.query, args.limit, args.full)
    elif args.entity:
        cmd_entity(chunks, args.entity, args.limit, args.full)
    elif args.paper:
        cmd_paper(chunks, args.paper, args.limit, args.full)
    elif args.papers:
        cmd_papers(chunks)
    elif args.chunk_id:
        cmd_chunk_id(chunks, args.chunk_id)
    elif args.short:
        cmd_short(chunks, threshold=20, limit=args.limit)
    else:
        # Default: show stats
        cmd_stats(chunks)


if __name__ == "__main__":
    main()