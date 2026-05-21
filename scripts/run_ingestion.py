#!/usr/bin/env python3
"""
scripts/run_ingestion.py

Full ingestion pipeline for the Thesis RAG system.

Pipeline stages:
  1. Parse     — PyMuPDF (thesis chapters) + Grobid (research papers)
  2. Dedup     — MinHash LSH removes near-duplicate paragraphs
  3. Chunk     — Semantic chunking into coherent passages
  4. Tag       — spaCy NER attaches entity metadata to every chunk
  5. Embed     — BGE-large-en-v1.5 generates 1024-dim vectors
  6. Index     — Vectors + metadata stored in Qdrant Cloud (HNSW)
  7. BM25      — rank_bm25 index built and saved to disk
  8. Save      — Processed chunks saved to disk for KG builder

Usage:
  python scripts/run_ingestion.py                  # Full run
  python scripts/run_ingestion.py --resume         # Resume from last checkpoint
  python scripts/run_ingestion.py --skip-papers    # Skip Grobid paper parsing
  python scripts/run_ingestion.py --skip-thesis    # Skip thesis chapter parsing

IMPORTANT — Before running:
  1. Start Grobid:  docker compose --profile ingestion up grobid -d
  2. Wait ~30s, then verify: curl http://localhost:8070/api/isalive
  3. Then run this script.
  After ingestion is done: docker compose stop grobid
"""

import os
import sys
import json
import pickle
import hashlib
import logging
import argparse
import time
import re
import math
import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any

import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ── Load .env ────────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingestion.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
class Config:
    # Paths
    BASE_DIR       = Path(__file__).resolve().parent.parent
    THESIS_DIR     = BASE_DIR / os.getenv("THESIS_DIR", "data/raw/thesis")
    PAPERS_DIR     = BASE_DIR / os.getenv("PAPERS_DIR", "data/raw/papers")
    PROCESSED_DIR  = BASE_DIR / os.getenv("PROCESSED_DIR", "data/processed")
    CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints"
    BM25_PATH      = BASE_DIR / os.getenv("BM25_INDEX_PATH", "data/processed/bm25_index.pkl")
    KG_OUTPUT_PATH = BASE_DIR / os.getenv("KG_OUTPUT_PATH", "app/core/knowledge_graph/graph.json")
    CHUNKS_JSON    = PROCESSED_DIR / "chunks_with_metadata.json"
    EMBEDDINGS_NPY = PROCESSED_DIR / "embeddings.npy"

    # Dedup
    DEDUP_THRESHOLD    = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.85"))
    DEDUP_PERMUTATIONS = int(os.getenv("DEDUP_NUM_PERMUTATIONS", "128"))
    DEDUP_SHINGLE_SIZE = int(os.getenv("DEDUP_SHINGLE_SIZE", "5"))

    # Chunking
    CHUNK_BREAKPOINT   = float(os.getenv("CHUNK_BREAKPOINT_THRESHOLD", "0.85"))
    CHUNK_MIN_TOKENS   = int(os.getenv("CHUNK_MIN_SIZE", "150"))
    CHUNK_MAX_TOKENS   = int(os.getenv("CHUNK_MAX_SIZE", "600"))
    CHUNK_PARENT_OVERLAP = int(os.getenv("CHUNK_PARENT_OVERLAP", "100"))

    # spaCy
    SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

    # Embedding
    EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    EMBEDDING_DIM       = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    EMBEDDING_DEVICE    = os.getenv("EMBEDDING_DEVICE", "cpu")

    # Qdrant
    QDRANT_USE_CLOUD     = os.getenv("QDRANT_USE_CLOUD", "true").lower() == "true"
    QDRANT_CLOUD_URL     = os.getenv("QDRANT_CLOUD_URL", "")
    QDRANT_API_KEY       = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION    = os.getenv("QDRANT_COLLECTION_NAME", "thesis_rag")
    QDRANT_HNSW_M        = int(os.getenv("QDRANT_HNSW_M", "16"))
    QDRANT_HNSW_EF_CONSTRUCT = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "200"))

    # Grobid
    GROBID_HOST     = os.getenv("GROBID_HOST", "localhost")
    GROBID_PORT     = int(os.getenv("GROBID_PORT", "8070"))
    GROBID_TIMEOUT  = int(os.getenv("GROBID_TIMEOUT_SECONDS", "120"))
    GROBID_MAX_CONCURRENT = int(os.getenv("GROBID_MAX_CONCURRENT_REQUESTS", "2"))

    # Thesis chapter name → (chapter_num, human_name)
    # These map the EXACT filenames you placed in data/raw/thesis (without .pdf)
    THESIS_CHAPTER_MAP: dict[str, tuple[int, str]] = {
        "Chapter1_Introduction":    (1, "Introduction"),
        "Chapter2_LiteratureReview": (2, "Literature Review"),
        "chapter3_methodology":     (3, "Methodology"),
        "Chapter4_Results":         (4, "Results"),
        "chapter5_discussion":      (5, "Discussion"),
        "Chapter6_Conclusion":      (6, "Conclusion"),
    }

    @classmethod
    def validate(cls) -> None:
        errors = []
        if not cls.THESIS_DIR.exists():
            errors.append(f"THESIS_DIR not found: {cls.THESIS_DIR}")
        if not cls.PAPERS_DIR.exists():
            errors.append(f"PAPERS_DIR not found: {cls.PAPERS_DIR}")
        if cls.QDRANT_USE_CLOUD and not cls.QDRANT_CLOUD_URL:
            errors.append("QDRANT_CLOUD_URL is empty but QDRANT_USE_CLOUD=true")
        if cls.QDRANT_USE_CLOUD and not cls.QDRANT_API_KEY:
            errors.append("QDRANT_API_KEY is empty but QDRANT_USE_CLOUD=true")
        if errors:
            for e in errors:
                log.error(e)
            sys.exit(1)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.KG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        log.info("Config validated OK.")


# ── Grobid TEI XML namespace ─────────────────────────────────────────────────
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class RawDocument:
    """A parsed document (thesis chapter or research paper) before chunking."""
    doc_id: str                          # SHA1 of filename
    source_type: str                     # "thesis" or "paper"
    filename: str
    paragraphs: list[dict[str, Any]] = field(default_factory=list)
    # Each paragraph: {text, section_heading, page_num, metadata: {...}}
    # metadata for thesis:  {chapter_num, chapter_name}
    # metadata for papers:  {title, authors, year, journal, doi, abstract}


@dataclass
class ProcessedChunk:
    """A single chunk ready for embedding and indexing."""
    chunk_id: str           # SHA1 of text content
    doc_id: str
    text: str
    source_type: str        # "thesis" | "paper"
    # Thesis-specific
    chapter_num: Optional[int]  = None
    chapter_name: Optional[str] = None
    chapter_filename: Optional[str] = None
    # Paper-specific
    paper_title: Optional[str]  = None
    paper_authors: Optional[list[str]] = None
    paper_year: Optional[int]   = None
    paper_journal: Optional[str] = None
    paper_doi: Optional[str]    = None
    paper_abstract: Optional[str] = None
    paper_filename: Optional[str] = None
    # Common
    section_heading: Optional[str] = None
    page_start: Optional[int]   = None
    token_count: int = 0
    word_count: int  = 0
    entities: list[str] = field(default_factory=list)
    entity_labels: dict[str, str] = field(default_factory=dict)
    embedding: Optional[list[float]] = None  # filled in Stage 5

    def to_qdrant_payload(self) -> dict:
        """Returns metadata dict suitable for Qdrant payload (no embedding)."""
        d = asdict(self)
        d.pop("embedding", None)
        return d


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def checkpoint_path(stage: str) -> Path:
    return Config.CHECKPOINT_DIR / f"stage_{stage}.pkl"

def save_checkpoint(stage: str, data: Any) -> None:
    p = checkpoint_path(stage)
    with open(p, "wb") as f:
        pickle.dump(data, f)
    log.info(f"  ✓ Checkpoint saved: {p.name}")

def load_checkpoint(stage: str) -> Optional[Any]:
    p = checkpoint_path(stage)
    if p.exists():
        with open(p, "rb") as f:
            data = pickle.load(f)
        log.info(f"  ↺ Loaded checkpoint: {p.name}")
        return data
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — PARSE
# ══════════════════════════════════════════════════════════════════════════════

def make_doc_id(filename: str) -> str:
    return hashlib.sha1(filename.encode()).hexdigest()[:16]

# ── Thesis parser (PyMuPDF) ──────────────────────────────────────────────────

def _detect_section_heading(block: dict, page_font_stats: dict) -> bool:
    """Heuristic: a text block is a section heading if its font size is
    larger than the median body font size on that page, or it is bold/all-caps."""
    text = block.get("text", "").strip()
    if not text or len(text) > 200:
        return False
    # All-caps short line
    if text.isupper() and len(text.split()) <= 10:
        return True
    # Font-size heuristic
    median_size = page_font_stats.get("median_size", 11.0)
    block_size  = block.get("size", 11.0)
    if block_size >= median_size * 1.15 and len(text.split()) <= 15:
        return True
    # Numbered section: "1.1 Something", "2. Background", etc.
    if re.match(r"^\d+(\.\d+)*\.?\s+\w", text):
        return True
    return False

def _page_font_stats(page) -> dict:
    """Compute median font size for a page to distinguish headings from body."""
    sizes = []
    for block in page.get_text("dict", flags=0)["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sz = span.get("size", 11.0)
                sizes.append(sz)
    if not sizes:
        return {"median_size": 11.0}
    sizes.sort()
    median = sizes[len(sizes) // 2]
    return {"median_size": median}

def parse_thesis_chapters(skip: bool = False) -> list[RawDocument]:
    """Stage 1a — Parse thesis chapter PDFs with PyMuPDF."""
    if skip:
        log.info("Stage 1a skipped (--skip-thesis).")
        return []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.error("PyMuPDF not installed. Run: pip install pymupdf")
        sys.exit(1)

    thesis_pdfs = list(Config.THESIS_DIR.glob("*.pdf"))
    if not thesis_pdfs:
        log.warning(f"No PDF files found in {Config.THESIS_DIR}")
        return []

    # Build a lookup: stem (lowercase) → (chapter_num, chapter_name)
    chapter_lookup: dict[str, tuple[int, str]] = {
        k.lower(): v for k, v in Config.THESIS_CHAPTER_MAP.items()
    }

    documents: list[RawDocument] = []
    log.info(f"Stage 1a — Parsing {len(thesis_pdfs)} thesis chapter(s) with PyMuPDF ...")

    for pdf_path in sorted(thesis_pdfs):
        stem = pdf_path.stem  # filename without extension
        match = chapter_lookup.get(stem.lower())
        if match is None:
            log.warning(f"  ⚠ '{stem}.pdf' does not match any chapter in THESIS_CHAPTER_MAP — skipping.")
            continue

        chapter_num, chapter_name = match
        doc_id = make_doc_id(pdf_path.name)
        log.info(f"  Parsing Ch{chapter_num}: {pdf_path.name} ...")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            log.error(f"  ✗ Failed to open {pdf_path.name}: {exc}")
            continue

        paragraphs: list[dict] = []
        current_heading = f"Chapter {chapter_num}: {chapter_name}"

        for page_num, page in enumerate(doc, start=1):
            font_stats = _page_font_stats(page)
            raw_blocks = page.get_text("dict", flags=0)["blocks"]

            for block in raw_blocks:
                if block.get("type") != 0:  # skip images/tables
                    continue
                spans_text = []
                block_max_size = 0.0
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if t:
                            spans_text.append(t)
                        sz = span.get("size", 11.0)
                        if sz > block_max_size:
                            block_max_size = sz

                full_text = " ".join(spans_text).strip()
                if not full_text or len(full_text) < 15:
                    continue

                synthetic_block = {"text": full_text, "size": block_max_size}
                if _detect_section_heading(synthetic_block, font_stats):
                    current_heading = full_text
                    continue  # headings are metadata, not content

                paragraphs.append({
                    "text": full_text,
                    "section_heading": current_heading,
                    "page_num": page_num,
                    "metadata": {
                        "chapter_num": chapter_num,
                        "chapter_name": chapter_name,
                        "chapter_filename": pdf_path.name,
                    },
                })

        doc.close()
        raw_doc = RawDocument(
            doc_id=doc_id,
            source_type="thesis",
            filename=pdf_path.name,
            paragraphs=paragraphs,
        )
        documents.append(raw_doc)
        log.info(f"    → {len(paragraphs)} paragraphs extracted.")

    log.info(f"Stage 1a complete. {len(documents)} thesis chapter(s) parsed.")
    return documents


# ── Paper parser (Grobid) ────────────────────────────────────────────────────

def check_grobid_health() -> bool:
    url = f"http://{Config.GROBID_HOST}:{Config.GROBID_PORT}/api/isalive"
    try:
        r = requests.get(url, timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False

def _grobid_parse_pdf(pdf_path: Path) -> Optional[str]:
    """Call Grobid processFulltextDocument and return TEI XML string."""
    url = f"http://{Config.GROBID_HOST}:{Config.GROBID_PORT}/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        try:
            resp = requests.post(
                url,
                files={"input": (pdf_path.name, f, "application/pdf")},
                data={"consolidateHeader": "1", "consolidateCitations": "0"},
                timeout=Config.GROBID_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            log.error(f"  ✗ Grobid failed for {pdf_path.name}: {exc}")
            return None

def _tei_text(element, ns=TEI_NS) -> str:
    """Recursively collect all text from a TEI element."""
    parts = []
    if element.text:
        parts.append(element.text.strip())
    for child in element:
        parts.append(_tei_text(child, ns))
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)

def _parse_tei_xml(tei_xml: str, filename: str) -> Optional[RawDocument]:
    """Parse Grobid TEI XML into a RawDocument."""
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as exc:
        log.error(f"  ✗ XML parse error for {filename}: {exc}")
        return None

    # ── Extract header metadata ──────────────────────────────────────────────
    header = root.find(".//tei:teiHeader", TEI_NS)

    title = ""
    title_el = root.find(".//tei:titleStmt/tei:title[@level='a']", TEI_NS) or \
               root.find(".//tei:titleStmt/tei:title", TEI_NS)
    if title_el is not None:
        title = (_tei_text(title_el) or "").strip()

    authors: list[str] = []
    for pers in root.findall(".//tei:analytic/tei:author/tei:persName", TEI_NS):
        forename = pers.findtext("tei:forename", "", TEI_NS)
        surname  = pers.findtext("tei:surname", "", TEI_NS)
        name = f"{forename} {surname}".strip()
        if name:
            authors.append(name)

    year: Optional[int] = None
    date_el = root.find(".//tei:publicationStmt/tei:date[@type='published']", TEI_NS) or \
              root.find(".//tei:imprint/tei:date[@type='published']", TEI_NS)
    if date_el is not None:
        raw_when = date_el.get("when", "")
        m = re.search(r"\b(19|20)\d{2}\b", raw_when)
        if m:
            year = int(m.group())

    journal = ""
    journal_el = root.find(".//tei:monogr/tei:title[@level='j']", TEI_NS)
    if journal_el is not None:
        journal = (_tei_text(journal_el) or "").strip()

    doi = ""
    doi_el = root.find(".//tei:idno[@type='DOI']", TEI_NS)
    if doi_el is not None:
        doi = (doi_el.text or "").strip()

    abstract_text = ""
    abstract_el = root.find(".//tei:abstract", TEI_NS)
    if abstract_el is not None:
        abstract_text = _tei_text(abstract_el).strip()

    paper_meta = {
        "title": title or filename,
        "authors": authors,
        "year": year,
        "journal": journal,
        "doi": doi,
        "abstract": abstract_text,
        "paper_filename": filename,
    }

    # ── Extract body sections and paragraphs ────────────────────────────────
    paragraphs: list[dict] = []
    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is None:
        log.warning(f"  ⚠ No <body> found in TEI XML for {filename}")
        return None

    def _process_div(div_el, parent_heading=""):
        head_el = div_el.find("tei:head", TEI_NS)
        heading = (_tei_text(head_el) or "").strip() if head_el is not None else parent_heading

        for para_el in div_el.findall("tei:p", TEI_NS):
            text = _tei_text(para_el).strip()
            if text and len(text) >= 30:
                paragraphs.append({
                    "text": text,
                    "section_heading": heading,
                    "page_num": None,
                    "metadata": paper_meta,
                })

        # Recurse into nested divs
        for sub_div in div_el.findall("tei:div", TEI_NS):
            _process_div(sub_div, heading)

    for div in body.findall("tei:div", TEI_NS):
        _process_div(div)

    # Also add abstract as a paragraph
    if abstract_text:
        paragraphs.insert(0, {
            "text": abstract_text,
            "section_heading": "Abstract",
            "page_num": None,
            "metadata": paper_meta,
        })

    doc_id = make_doc_id(filename)
    return RawDocument(
        doc_id=doc_id,
        source_type="paper",
        filename=filename,
        paragraphs=paragraphs,
    )

def parse_papers_grobid(skip: bool = False) -> list[RawDocument]:
    """Stage 1b — Parse research paper PDFs with Grobid."""
    if skip:
        log.info("Stage 1b skipped (--skip-papers).")
        return []

    paper_pdfs = list(Config.PAPERS_DIR.glob("*.pdf"))
    if not paper_pdfs:
        log.warning(f"No PDF files found in {Config.PAPERS_DIR}")
        return []

    log.info("Stage 1b — Checking Grobid health ...")
    if not check_grobid_health():
        log.error(
            "\n"
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║  GROBID IS NOT RUNNING                                          ║\n"
            "║  Start it with:                                                  ║\n"
            "║    docker compose --profile ingestion up grobid -d              ║\n"
            "║  Then wait ~30 seconds and verify:                              ║\n"
            "║    curl http://localhost:8070/api/isalive                       ║\n"
            "║  Then re-run this script.                                        ║\n"
            "╚══════════════════════════════════════════════════════════════════╝"
        )
        sys.exit(1)

    log.info(f"Stage 1b — Parsing {len(paper_pdfs)} paper(s) with Grobid ...")
    documents: list[RawDocument] = []
    semaphore_slots = Config.GROBID_MAX_CONCURRENT

    for pdf_path in tqdm(paper_pdfs, desc="Grobid parsing"):
        tei_xml = _grobid_parse_pdf(pdf_path)
        if tei_xml is None:
            continue
        raw_doc = _parse_tei_xml(tei_xml, pdf_path.name)
        if raw_doc is None:
            continue
        documents.append(raw_doc)
        # Grobid rate limiting
        time.sleep(0.5 / semaphore_slots)

    log.info(f"Stage 1b complete. {len(documents)} paper(s) parsed.")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — DEDUPLICATION (MinHash LSH)
# ══════════════════════════════════════════════════════════════════════════════

def _get_shingles(text: str, n: int) -> set[str]:
    words = text.lower().split()
    if len(words) < n:
        return {text.lower()}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

def deduplicate_documents(documents: list[RawDocument]) -> list[RawDocument]:
    """Stage 2 — Remove near-duplicate paragraphs using MinHash LSH."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        log.error("datasketch not installed. Run: pip install datasketch")
        sys.exit(1)

    threshold  = Config.DEDUP_THRESHOLD
    num_perm   = Config.DEDUP_PERMUTATIONS
    shingle_n  = Config.DEDUP_SHINGLE_SIZE

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    seen_ids: set[str] = set()
    deduped_docs: list[RawDocument] = []
    total_before = sum(len(d.paragraphs) for d in documents)

    log.info(f"Stage 2 — Deduplicating {total_before} paragraphs (threshold={threshold}) ...")

    for doc in documents:
        clean_paragraphs = []
        for i, para in enumerate(doc.paragraphs):
            text = para["text"]
            para_id = f"{doc.doc_id}_{i}"
            shingles = _get_shingles(text, shingle_n)

            mh = MinHash(num_perm=num_perm)
            for shingle in shingles:
                mh.update(shingle.encode("utf-8"))

            try:
                result = lsh.query(mh)
            except Exception:
                result = []

            if result:
                # Near-duplicate found — skip
                continue

            try:
                lsh.insert(para_id, mh)
                seen_ids.add(para_id)
                clean_paragraphs.append(para)
            except ValueError:
                # Already inserted (shouldn't happen, but guard anyway)
                clean_paragraphs.append(para)

        deduped_doc = copy.deepcopy(doc)
        deduped_doc.paragraphs = clean_paragraphs
        deduped_docs.append(deduped_doc)

    total_after = sum(len(d.paragraphs) for d in deduped_docs)
    removed = total_before - total_after
    log.info(f"Stage 2 complete. Removed {removed} duplicates. {total_after} paragraphs remain.")
    return deduped_docs


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SEMANTIC CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def _approx_token_count(text: str) -> int:
    """Approximate token count: avg English word is ~1.3 tokens."""
    return int(len(text.split()) * 1.3)

def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (no NLTK dependency)."""
    # Split on '. ', '! ', '? ' but be careful with abbreviations
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'(\[])", text)
    result = []
    for s in sentences:
        s = s.strip()
        if s:
            result.append(s)
    return result if result else [text]

def _force_split_large_chunk(text: str, max_tokens: int) -> list[str]:
    """Split an oversized chunk by sentences until each piece is under max_tokens."""
    sentences = _split_into_sentences(text)
    chunks, current, current_tokens = [], [], 0
    for sent in sentences:
        t = _approx_token_count(sent)
        if current_tokens + t > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_tokens = [sent], t
        else:
            current.append(sent)
            current_tokens += t
    if current:
        chunks.append(" ".join(current))
    return chunks

class SemanticChunker:
    """
    Splits a text into semantically coherent chunks using sentence embeddings.
    Finds breakpoints where cosine similarity between consecutive sentences drops
    below `breakpoint_threshold`.
    """

    def __init__(self, model, breakpoint_threshold: float, min_tokens: int, max_tokens: int):
        self.model = model
        self.threshold = breakpoint_threshold
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def chunk(self, text: str) -> list[str]:
        sentences = _split_into_sentences(text)
        if len(sentences) <= 2:
            return self._enforce_limits([text])

        # Encode all sentences at once
        embeddings = self.model.encode(
            sentences,
            batch_size=min(32, len(sentences)),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Find semantic breakpoints
        breakpoints: list[int] = []
        for i in range(len(sentences) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            if sim < self.threshold:
                breakpoints.append(i + 1)

        # Build raw chunks from breakpoints
        raw_chunks, start = [], 0
        for bp in breakpoints + [len(sentences)]:
            chunk_text = " ".join(sentences[start:bp]).strip()
            if chunk_text:
                raw_chunks.append(chunk_text)
            start = bp

        return self._enforce_limits(raw_chunks)

    def _enforce_limits(self, chunks: list[str]) -> list[str]:
        """Merge chunks that are too small; split chunks that are too large."""
        # Pass 1: merge small chunks into their neighbour
        merged: list[str] = []
        i = 0
        while i < len(chunks):
            t = _approx_token_count(chunks[i])
            if t < self.min_tokens and merged:
                merged[-1] = merged[-1] + " " + chunks[i]
            else:
                merged.append(chunks[i])
            i += 1

        # Pass 2: force-split chunks that exceed max_tokens
        final: list[str] = []
        for chunk in merged:
            if _approx_token_count(chunk) > self.max_tokens:
                final.extend(_force_split_large_chunk(chunk, self.max_tokens))
            else:
                final.append(chunk)

        return [c.strip() for c in final if c.strip()]


def chunk_documents(
    documents: list[RawDocument],
    embed_model,
) -> list[ProcessedChunk]:
    """Stage 3 — Semantically chunk all paragraphs from all documents."""
    chunker = SemanticChunker(
        model=embed_model,
        breakpoint_threshold=Config.CHUNK_BREAKPOINT,
        min_tokens=Config.CHUNK_MIN_TOKENS,
        max_tokens=Config.CHUNK_MAX_TOKENS,
    )

    all_chunks: list[ProcessedChunk] = []
    log.info("Stage 3 — Semantic chunking ...")

    for doc in tqdm(documents, desc="Chunking documents"):
        for para in doc.paragraphs:
            text = para["text"]
            if not text.strip():
                continue

            sub_chunks = chunker.chunk(text)
            meta = para.get("metadata", {})
            section = para.get("section_heading", "")
            page_num = para.get("page_num")

            for idx, chunk_text in enumerate(sub_chunks):
                chunk_id = hashlib.sha1(chunk_text.encode()).hexdigest()[:20]

                if doc.source_type == "thesis":
                    chunk = ProcessedChunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        source_type="thesis",
                        chapter_num=meta.get("chapter_num"),
                        chapter_name=meta.get("chapter_name"),
                        chapter_filename=meta.get("chapter_filename"),
                        section_heading=section,
                        page_start=page_num,
                        token_count=_approx_token_count(chunk_text),
                        word_count=len(chunk_text.split()),
                    )
                else:
                    chunk = ProcessedChunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        source_type="paper",
                        paper_title=meta.get("title"),
                        paper_authors=meta.get("authors", []),
                        paper_year=meta.get("year"),
                        paper_journal=meta.get("journal"),
                        paper_doi=meta.get("doi"),
                        paper_abstract=meta.get("abstract"),
                        paper_filename=meta.get("paper_filename"),
                        section_heading=section,
                        page_start=page_num,
                        token_count=_approx_token_count(chunk_text),
                        word_count=len(chunk_text.split()),
                    )

                all_chunks.append(chunk)

    thesis_count = sum(1 for c in all_chunks if c.source_type == "thesis")
    paper_count  = sum(1 for c in all_chunks if c.source_type == "paper")
    log.info(
        f"Stage 3 complete. {len(all_chunks)} chunks total "
        f"({thesis_count} thesis, {paper_count} paper)."
    )
    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — ENTITY TAGGING (spaCy)
# ══════════════════════════════════════════════════════════════════════════════

def tag_chunks(chunks: list[ProcessedChunk]) -> list[ProcessedChunk]:
    """Stage 4 — Tag each chunk with named entities using spaCy NER."""
    try:
        import spacy
    except ImportError:
        log.error("spaCy not installed. Run: pip install spacy")
        sys.exit(1)

    model_name = Config.SPACY_MODEL
    try:
        nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
    except OSError:
        log.error(
            f"spaCy model '{model_name}' not found. "
            f"Run: python -m spacy download {model_name}"
        )
        sys.exit(1)

    log.info(f"Stage 4 — Entity tagging with spaCy '{model_name}' ...")
    KEEP_LABELS = {"ORG", "PERSON", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART",
                   "LAW", "LANGUAGE", "NORP", "FAC"}

    texts = [c.text for c in chunks]
    batch_size = 64

    tagged = 0
    for i, doc in enumerate(
        tqdm(
            nlp.pipe(texts, batch_size=batch_size),
            total=len(texts),
            desc="spaCy tagging",
        )
    ):
        ents: list[str] = []
        labels: dict[str, str] = {}
        for ent in doc.ents:
            if ent.label_ in KEEP_LABELS and len(ent.text) > 2:
                clean = ent.text.strip()
                if clean not in labels:
                    ents.append(clean)
                    labels[clean] = ent.label_
        chunks[i].entities = ents
        chunks[i].entity_labels = labels
        if ents:
            tagged += 1

    log.info(f"Stage 4 complete. {tagged}/{len(chunks)} chunks have entities.")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — GENERATE EMBEDDINGS (BGE-large)
# ══════════════════════════════════════════════════════════════════════════════

def load_embedding_model():
    """Load BGE-large from sentence-transformers (downloads from HF on first run)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    log.info(f"Loading embedding model: {Config.EMBEDDING_MODEL} ...")
    model = SentenceTransformer(
        Config.EMBEDDING_MODEL,
        device=Config.EMBEDDING_DEVICE,
        cache_folder=os.getenv("HF_HOME", "~/.cache/huggingface"),
    )
    log.info(f"  Embedding model loaded on device: {Config.EMBEDDING_DEVICE}")
    return model

def embed_chunks(chunks: list[ProcessedChunk], model) -> tuple[list[ProcessedChunk], np.ndarray]:
    """Stage 5 — Generate 1024-dim BGE embeddings for all chunks.

    BGE-large notes:
    - Documents are encoded WITHOUT instruction prefix.
    - Queries (at retrieval time) use prefix 'Represent this sentence for searching relevant passages: '
    - normalize_embeddings=True for cosine similarity.
    """
    log.info(f"Stage 5 — Embedding {len(chunks)} chunks (batch={Config.EMBEDDING_BATCH_SIZE}) ...")
    texts = [c.text for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=Config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Validate dimension
    if embeddings.shape[1] != Config.EMBEDDING_DIM:
        log.error(
            f"Embedding dimension mismatch: got {embeddings.shape[1]}, "
            f"expected {Config.EMBEDDING_DIM}. Check EMBEDDING_MODEL in .env."
        )
        sys.exit(1)

    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i].tolist()

    log.info(f"Stage 5 complete. Embeddings shape: {embeddings.shape}")
    return chunks, embeddings


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — INDEX IN QDRANT
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_id_to_qdrant_int(chunk_id: str) -> int:
    """Convert 20-char hex chunk_id to a stable integer for Qdrant point ID."""
    return int(chunk_id[:16], 16)  # use first 16 hex chars → 64-bit int

def setup_qdrant_collection(client) -> None:
    """Create or verify Qdrant collection with HNSW config."""
    from qdrant_client.models import (
        Distance,
        VectorParams,
        HnswConfigDiff,
        OptimizersConfigDiff,
    )

    existing = [c.name for c in client.get_collections().collections]
    if Config.QDRANT_COLLECTION in existing:
        log.info(f"  Qdrant collection '{Config.QDRANT_COLLECTION}' already exists — using it.")
        return

    log.info(f"  Creating Qdrant collection '{Config.QDRANT_COLLECTION}' ...")
    client.create_collection(
        collection_name=Config.QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=Config.EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=Config.QDRANT_HNSW_M,
            ef_construct=Config.QDRANT_HNSW_EF_CONSTRUCT,
            full_scan_threshold=10_000,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20_000,
        ),
    )
    log.info("  Collection created.")

def index_chunks_qdrant(chunks: list[ProcessedChunk], embeddings: np.ndarray) -> None:
    """Stage 6 — Upsert all chunks + embeddings into Qdrant Cloud."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
    except ImportError:
        log.error("qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)

    log.info("Stage 6 — Connecting to Qdrant Cloud ...")
    client = QdrantClient(
        url=Config.QDRANT_CLOUD_URL,
        api_key=Config.QDRANT_API_KEY,
        timeout=60,
    )
    setup_qdrant_collection(client)

    BATCH_SIZE = 64
    total_upserted = 0
    log.info(f"Stage 6 — Upserting {len(chunks)} vectors to Qdrant ...")

    for batch_start in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        desc="Qdrant upsert",
        total=math.ceil(len(chunks) / BATCH_SIZE),
    ):
        batch_chunks = chunks[batch_start : batch_start + BATCH_SIZE]
        batch_embs   = embeddings[batch_start : batch_start + BATCH_SIZE]

        points = []
        for chunk, emb in zip(batch_chunks, batch_embs):
            point_id = _chunk_id_to_qdrant_int(chunk.chunk_id)
            payload  = chunk.to_qdrant_payload()
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=Config.QDRANT_COLLECTION,
            points=points,
            wait=True,
        )
        total_upserted += len(points)

    log.info(f"Stage 6 complete. {total_upserted} vectors upserted to Qdrant.")

    # Verify
    info = client.get_collection(Config.QDRANT_COLLECTION)
    log.info(f"  Qdrant collection vectors count: {info.vectors_count}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — BUILD BM25 INDEX
# ══════════════════════════════════════════════════════════════════════════════

def build_bm25_index(chunks: list[ProcessedChunk]) -> None:
    """Stage 7 — Build and serialize a rank_bm25 index for keyword retrieval."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.error("rank_bm25 not installed. Run: pip install rank-bm25")
        sys.exit(1)

    log.info(f"Stage 7 — Building BM25 index over {len(chunks)} chunks ...")
    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Store the BM25 object alongside chunk_ids for retrieval mapping
    index_data = {
        "bm25": bm25,
        "chunk_ids": [c.chunk_id for c in chunks],
        "texts": [c.text for c in chunks],
    }

    Config.BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.BM25_PATH, "wb") as f:
        pickle.dump(index_data, f)

    log.info(f"Stage 7 complete. BM25 index saved to {Config.BM25_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — SAVE PROCESSED CHUNKS TO DISK (for KG builder)
# ══════════════════════════════════════════════════════════════════════════════

def save_chunks_to_disk(chunks: list[ProcessedChunk], embeddings: np.ndarray) -> None:
    """Stage 8 — Persist chunks (without embeddings) as JSON and embeddings as .npy.
    These files are consumed by build_knowledge_graph.py."""

    log.info("Stage 8 — Saving processed chunks to disk ...")

    # Save chunks as JSON (no embedding — too large for JSON)
    chunk_dicts = []
    for c in chunks:
        d = asdict(c)
        d.pop("embedding", None)
        chunk_dicts.append(d)

    with open(Config.CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)
    log.info(f"  Chunks JSON → {Config.CHUNKS_JSON} ({len(chunk_dicts)} chunks)")

    # Save embeddings as numpy array
    np.save(str(Config.EMBEDDINGS_NPY), embeddings)
    log.info(f"  Embeddings  → {Config.EMBEDDINGS_NPY} shape={embeddings.shape}")

    log.info("Stage 8 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thesis RAG ingestion pipeline")
    parser.add_argument("--resume",       action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--skip-papers",  action="store_true", help="Skip Grobid paper parsing")
    parser.add_argument("--skip-thesis",  action="store_true", help="Skip thesis chapter parsing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("═" * 70)
    log.info("  Thesis RAG — Ingestion Pipeline")
    log.info("═" * 70)

    Config.validate()

    t_start = time.time()

    # ── Stage 1: Parse ───────────────────────────────────────────────────────
    ck_parse = load_checkpoint("01_parse") if args.resume else None
    if ck_parse is not None:
        log.info("Stage 1 — Loaded from checkpoint.")
        raw_documents: list[RawDocument] = ck_parse
    else:
        thesis_docs = parse_thesis_chapters(skip=args.skip_thesis)
        paper_docs  = parse_papers_grobid(skip=args.skip_papers)
        raw_documents = thesis_docs + paper_docs

        if not raw_documents:
            log.error("No documents parsed. Check your data directories and .env config.")
            sys.exit(1)

        total_paras = sum(len(d.paragraphs) for d in raw_documents)
        log.info(
            f"Stage 1 complete. {len(raw_documents)} documents, "
            f"{total_paras} total paragraphs."
        )
        save_checkpoint("01_parse", raw_documents)

    # ── Stage 2: Dedup ───────────────────────────────────────────────────────
    ck_dedup = load_checkpoint("02_dedup") if args.resume else None
    if ck_dedup is not None:
        log.info("Stage 2 — Loaded from checkpoint.")
        deduped_docs: list[RawDocument] = ck_dedup
    else:
        deduped_docs = deduplicate_documents(raw_documents)
        save_checkpoint("02_dedup", deduped_docs)

    # ── Load embedding model (needed for Stage 3 + Stage 5) ──────────────────
    embed_model = load_embedding_model()

    # ── Stage 3: Chunk ───────────────────────────────────────────────────────
    ck_chunk = load_checkpoint("03_chunk") if args.resume else None
    if ck_chunk is not None:
        log.info("Stage 3 — Loaded from checkpoint.")
        chunks: list[ProcessedChunk] = ck_chunk
    else:
        chunks = chunk_documents(deduped_docs, embed_model)
        save_checkpoint("03_chunk", chunks)

    # ── Stage 4: Tag ─────────────────────────────────────────────────────────
    ck_tag = load_checkpoint("04_tag") if args.resume else None
    if ck_tag is not None:
        log.info("Stage 4 — Loaded from checkpoint.")
        chunks = ck_tag
    else:
        chunks = tag_chunks(chunks)
        save_checkpoint("04_tag", chunks)

    # ── Stage 5: Embed ───────────────────────────────────────────────────────
    ck_embed = load_checkpoint("05_embed") if args.resume else None
    if ck_embed is not None:
        log.info("Stage 5 — Loaded from checkpoint.")
        chunks, embeddings = ck_embed
    else:
        chunks, embeddings = embed_chunks(chunks, embed_model)
        save_checkpoint("05_embed", (chunks, embeddings))

    # ── Stage 6: Index Qdrant ────────────────────────────────────────────────
    ck_qdrant = load_checkpoint("06_qdrant") if args.resume else None
    if ck_qdrant is not None:
        log.info("Stage 6 — Qdrant indexing already done (checkpoint). Skipping.")
    else:
        index_chunks_qdrant(chunks, embeddings)
        save_checkpoint("06_qdrant", True)

    # ── Stage 7: BM25 ────────────────────────────────────────────────────────
    ck_bm25 = load_checkpoint("07_bm25") if args.resume else None
    if ck_bm25 is not None:
        log.info("Stage 7 — BM25 already done (checkpoint). Skipping.")
    else:
        build_bm25_index(chunks)
        save_checkpoint("07_bm25", True)

    # ── Stage 8: Save chunks to disk ─────────────────────────────────────────
    save_chunks_to_disk(chunks, embeddings)

    elapsed = time.time() - t_start
    log.info("═" * 70)
    log.info(f"  ✓ Ingestion pipeline complete in {elapsed:.1f}s")
    log.info(f"  ✓ {len(chunks)} chunks indexed in Qdrant collection '{Config.QDRANT_COLLECTION}'")
    log.info(f"  ✓ BM25 index saved to {Config.BM25_PATH}")
    log.info(f"  ✓ Chunks JSON saved for KG builder")
    log.info("  ✓ Next step: python scripts/build_knowledge_graph.py")
    log.info("═" * 70)


if __name__ == "__main__":
    main()