#!/usr/bin/env python3
"""
scripts/build_knowledge_graph.py

Builds a NetworkX knowledge graph from the processed chunks produced by
run_ingestion.py (Stage 8 output: data/processed/chunks_with_metadata.json).

Graph structure:
  Nodes:
    - ENTITY nodes  : named entities extracted by spaCy (entities, entity_labels)
    - CHUNK nodes   : individual text chunks (chunk_id)
    - DOCUMENT nodes: source documents (doc_id → chapter or paper)

  Edges:
    - CHUNK  → ENTITY       : chunk contains this entity
    - ENTITY → ENTITY       : two entities co-occur in the same chunk
    - CHUNK  → DOCUMENT     : chunk belongs to document
    - ENTITY → CHUNK (occ.) : reverse lookup for retrieval

  Node attributes (ENTITY):
    - label      : spaCy entity type (ORG, PERSON, GPE, …)
    - frequency  : how many chunks this entity appears in
    - source_types: set of "thesis" | "paper"

  Node attributes (CHUNK):
    - text, source_type, chapter_num/name, paper_title, section_heading, …

  Node attributes (DOCUMENT):
    - source_type, chapter_num, chapter_name, paper_title, paper_authors, paper_year

Graph is saved as JSON (node-link format) to:
  app/core/knowledge_graph/graph.json

Usage:
  python scripts/build_knowledge_graph.py
  python scripts/build_knowledge_graph.py --min-frequency 2
"""

import os
import sys
import json
import logging
import argparse
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("kg_build.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
PROCESSED_DIR   = BASE_DIR / os.getenv("PROCESSED_DIR", "data/processed")
CHUNKS_JSON     = PROCESSED_DIR / "chunks_with_metadata.json"
KG_OUTPUT_PATH  = BASE_DIR / os.getenv("KG_OUTPUT_PATH", "app/core/knowledge_graph/graph.json")
KG_MIN_FREQ     = int(os.getenv("KG_MIN_ENTITY_FREQUENCY", "1"))


# ══════════════════════════════════════════════════════════════════════════════
# LOAD CHUNKS
# ══════════════════════════════════════════════════════════════════════════════

def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        log.error(
            f"Chunks file not found: {path}\n"
            "Run 'python scripts/run_ingestion.py' first."
        )
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    log.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# BUILD KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_knowledge_graph(chunks: list[dict], min_entity_frequency: int):
    """Build a NetworkX DiGraph from processed chunks.

    Strategy
    ────────
    1. Pass 1  — Count entity frequencies across all chunks.
    2. Filter  — Keep only entities that appear >= min_entity_frequency times.
    3. Pass 2  — Build nodes and edges.
    """
    try:
        import networkx as nx
    except ImportError:
        log.error("networkx not installed. Run: pip install networkx")
        sys.exit(1)

    G = nx.DiGraph()

    # ── Pass 1: count entity frequencies ────────────────────────────────────
    log.info("Pass 1 — Counting entity frequencies ...")
    entity_freq: dict[str, int] = defaultdict(int)
    entity_label_map: dict[str, str] = {}
    entity_sources: dict[str, set] = defaultdict(set)

    for chunk in chunks:
        entities: list[str] = chunk.get("entities", [])
        labels: dict[str, str] = chunk.get("entity_labels", {})
        src = chunk.get("source_type", "unknown")
        for ent in entities:
            entity_freq[ent] += 1
            entity_sources[ent].add(src)
            if ent not in entity_label_map and ent in labels:
                entity_label_map[ent] = labels[ent]

    total_entities = len(entity_freq)
    # Filter by minimum frequency
    kept_entities: set[str] = {
        ent for ent, freq in entity_freq.items()
        if freq >= min_entity_frequency
    }
    log.info(
        f"  {total_entities} unique entities found. "
        f"{len(kept_entities)} kept with frequency >= {min_entity_frequency}."
    )

    # ── Pass 2: build nodes ──────────────────────────────────────────────────
    log.info("Pass 2 — Building graph nodes ...")

    # ENTITY nodes
    for ent in kept_entities:
        G.add_node(
            f"entity::{ent}",
            node_type="entity",
            entity_text=ent,
            entity_label=entity_label_map.get(ent, "UNKNOWN"),
            frequency=entity_freq[ent],
            source_types=list(entity_sources[ent]),
        )

    # DOCUMENT nodes — deduplicated by doc_id
    doc_nodes_seen: set[str] = set()
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "unknown")
        node_key = f"doc::{doc_id}"
        if node_key in doc_nodes_seen:
            continue
        doc_nodes_seen.add(node_key)

        if chunk.get("source_type") == "thesis":
            G.add_node(
                node_key,
                node_type="document",
                source_type="thesis",
                doc_id=doc_id,
                chapter_num=chunk.get("chapter_num"),
                chapter_name=chunk.get("chapter_name"),
                chapter_filename=chunk.get("chapter_filename"),
            )
        else:
            G.add_node(
                node_key,
                node_type="document",
                source_type="paper",
                doc_id=doc_id,
                paper_title=chunk.get("paper_title"),
                paper_authors=chunk.get("paper_authors", []),
                paper_year=chunk.get("paper_year"),
                paper_journal=chunk.get("paper_journal"),
                paper_doi=chunk.get("paper_doi"),
                paper_filename=chunk.get("paper_filename"),
            )

    # CHUNK nodes + edges
    log.info("Pass 2 — Building chunk nodes and edges ...")
    cooccurrence_edges: dict[tuple[str, str], int] = defaultdict(int)

    for chunk in chunks:
        chunk_id  = chunk.get("chunk_id", "")
        doc_id    = chunk.get("doc_id", "")
        chunk_key = f"chunk::{chunk_id}"

        # ── Chunk node ──────────────────────────────────────────────────────
        G.add_node(
            chunk_key,
            node_type="chunk",
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=chunk.get("text", "")[:500],   # Store first 500 chars (preview)
            full_text=chunk.get("text", ""),
            source_type=chunk.get("source_type"),
            section_heading=chunk.get("section_heading"),
            page_start=chunk.get("page_start"),
            token_count=chunk.get("token_count", 0),
            word_count=chunk.get("word_count", 0),
            # Thesis-specific
            chapter_num=chunk.get("chapter_num"),
            chapter_name=chunk.get("chapter_name"),
            # Paper-specific
            paper_title=chunk.get("paper_title"),
            paper_year=chunk.get("paper_year"),
            paper_authors=chunk.get("paper_authors", []),
        )

        # ── Chunk → Document edge ────────────────────────────────────────────
        doc_key = f"doc::{doc_id}"
        if G.has_node(doc_key):
            G.add_edge(
                chunk_key,
                doc_key,
                relation="BELONGS_TO",
            )

        # ── Chunk ↔ Entity edges ─────────────────────────────────────────────
        chunk_ents = [
            e for e in chunk.get("entities", [])
            if e in kept_entities
        ]

        for ent in chunk_ents:
            ent_key = f"entity::{ent}"
            # Chunk → Entity (chunk CONTAINS entity)
            G.add_edge(chunk_key, ent_key, relation="CONTAINS_ENTITY")
            # Entity → Chunk (reverse for retrieval)
            G.add_edge(ent_key, chunk_key, relation="APPEARS_IN")

        # ── Entity–Entity co-occurrence ──────────────────────────────────────
        if len(chunk_ents) >= 2:
            for e1, e2 in combinations(sorted(chunk_ents), 2):
                key = (f"entity::{e1}", f"entity::{e2}")
                cooccurrence_edges[key] += 1

    # Add co-occurrence edges with weight
    log.info(f"  Adding {len(cooccurrence_edges)} entity co-occurrence edges ...")
    for (ent_key1, ent_key2), weight in cooccurrence_edges.items():
        if G.has_node(ent_key1) and G.has_node(ent_key2):
            G.add_edge(
                ent_key1,
                ent_key2,
                relation="CO_OCCURS_WITH",
                weight=weight,
            )
            G.add_edge(
                ent_key2,
                ent_key1,
                relation="CO_OCCURS_WITH",
                weight=weight,
            )

    log.info(
        f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
    )
    return G


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def print_graph_stats(G) -> None:
    import networkx as nx

    node_types = {}
    for _, data in G.nodes(data=True):
        t = data.get("node_type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    edge_types = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "unknown")
        edge_types[r] = edge_types.get(r, 0) + 1

    log.info("── Graph statistics ──────────────────────────────────")
    log.info(f"  Total nodes : {G.number_of_nodes()}")
    for t, n in sorted(node_types.items()):
        log.info(f"    {t:<15}: {n}")
    log.info(f"  Total edges : {G.number_of_edges()}")
    for r, n in sorted(edge_types.items()):
        log.info(f"    {r:<25}: {n}")

    # Top 10 entities by frequency
    entity_nodes = [
        (data["entity_text"], data["frequency"])
        for _, data in G.nodes(data=True)
        if data.get("node_type") == "entity"
    ]
    entity_nodes.sort(key=lambda x: x[1], reverse=True)
    log.info("  Top 10 entities by frequency:")
    for name, freq in entity_nodes[:10]:
        log.info(f"    [{freq:>4}] {name}")
    log.info("─" * 54)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def save_graph(G, output_path: Path) -> None:
    """Serialize NetworkX graph to JSON (node-link format)."""
    import networkx as nx

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # node_link_data produces a JSON-serialisable dict
    data = nx.node_link_data(G)

    # Convert any sets to lists for JSON serialisation
    def _make_serialisable(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serialisable(v) for v in obj]
        return obj

    data = _make_serialisable(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info(f"  Knowledge graph saved → {output_path}  ({size_mb:.2f} MB)")


def load_graph(path: Path):
    """Load a previously saved NetworkX graph from JSON."""
    import networkx as nx

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.node_link_graph(data, directed=True)
    log.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build knowledge graph for Thesis RAG")
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=KG_MIN_FREQ,
        help=f"Minimum entity frequency to include in KG (default: {KG_MIN_FREQ} from .env)",
    )
    parser.add_argument(
        "--chunks-path",
        type=str,
        default=str(CHUNKS_JSON),
        help="Path to chunks_with_metadata.json produced by run_ingestion.py",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(KG_OUTPUT_PATH),
        help="Output path for graph.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("═" * 70)
    log.info("  Thesis RAG — Knowledge Graph Builder")
    log.info("═" * 70)

    chunks_path = Path(args.chunks_path)
    output_path = Path(args.output_path)
    min_freq    = args.min_frequency

    log.info(f"  Chunks source   : {chunks_path}")
    log.info(f"  Output path     : {output_path}")
    log.info(f"  Min entity freq : {min_freq}")

    t_start = time.time()

    # Load chunks
    chunks = load_chunks(chunks_path)

    # Build graph
    G = build_knowledge_graph(chunks, min_entity_frequency=min_freq)

    # Print stats
    print_graph_stats(G)

    # Save graph
    save_graph(G, output_path)

    elapsed = time.time() - t_start
    log.info("═" * 70)
    log.info(f"  ✓ Knowledge graph built in {elapsed:.1f}s")
    log.info(f"  ✓ Graph JSON saved to {output_path}")
    log.info(
        f"  ✓ Load it in your app with:\n"
        f"      import networkx as nx, json\n"
        f"      G = nx.node_link_graph(json.load(open('{output_path}')))"
    )
    log.info("═" * 70)


if __name__ == "__main__":
    main()