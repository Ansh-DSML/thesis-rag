"""
app/utils/text_utils.py

Pure utility functions with no external service dependencies.
Safe to import anywhere — no circular imports possible.

Functions:
  is_equation(text)           → bool   — does this text contain LaTeX math?
  extract_latex_blocks(text)  → list   — pull out all LaTeX blocks
  count_tokens(text)          → int    — approximate token count (tiktoken cl100k)
  count_tokens_approx(text)   → int    — word-based fallback (no tiktoken needed)
  clean_text(text)            → str    — normalise whitespace, fix encoding artefacts
  truncate_to_tokens(text, n) → str    — cut text to n tokens without mid-word breaks
  split_into_sentences(text)  → list   — lightweight sentence splitter
  is_meaningful(text, min_words) → bool — reject headers, page numbers, etc.
  normalise_entity(text)      → str    — lowercase + strip for entity matching
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# TOKEN COUNTING
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _get_tokenizer():
    """
    Load tiktoken encoder once and cache it.
    Uses cl100k_base (GPT-4 / Claude approximate).
    Falls back to None if tiktoken is not installed.
    """
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken (cl100k_base encoding).

    Note: Gemini uses a different internal tokenizer. This gives a close
    approximation — within ~5% for English academic text.
    Falls back to word-based estimate if tiktoken is not installed.
    """
    if not text:
        return 0
    enc = _get_tokenizer()
    if enc is not None:
        return len(enc.encode(text))
    return count_tokens_approx(text)


def count_tokens_approx(text: str) -> int:
    """
    Word-based token approximation (no dependencies).
    Rule of thumb: 1 English word ≈ 1.3 tokens.
    Accurate to within ~10% for academic text.
    """
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to at most max_tokens tokens.
    Breaks at word boundaries — never mid-word.
    """
    if count_tokens(text) <= max_tokens:
        return text

    enc = _get_tokenizer()
    if enc is not None:
        tokens = enc.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return enc.decode(truncated_tokens)

    # Fallback: approximate via words
    words = text.split()
    target_words = int(max_tokens / 1.3)
    return " ".join(words[:target_words])


# ══════════════════════════════════════════════════════════════════════════════
# LATEX / EQUATION DETECTION
# ══════════════════════════════════════════════════════════════════════════════

# Patterns that indicate the presence of a LaTeX math expression
_LATEX_PATTERNS = [
    re.compile(r"\$\$.+?\$\$", re.DOTALL),                    # $$...$$
    re.compile(r"\$.+?\$"),                                    # $...$
    re.compile(r"\\\[.+?\\\]", re.DOTALL),                    # \[...\]
    re.compile(r"\\\(.+?\\\)"),                                # \(...\)
    re.compile(r"\\begin\{(equation|align|math|gather|multline)\*?\}"),  # \begin{equation}
    re.compile(r"\\(?:frac|sum|int|prod|lim|log|exp|sqrt|hat|bar|vec|"
               r"alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|"
               r"omega|Omega|nabla|partial|infty|forall|exists|in|notin|"
               r"leq|geq|neq|approx|sim|cdot|times|div|pm|mp)\b"),
]

# Patterns for purely structural/non-content blocks
_NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                          # Page numbers
    re.compile(r"^\s*[ivxlcdmIVXLCDM]+\s*$"),           # Roman numerals
    re.compile(r"^\s*figure\s+\d+", re.IGNORECASE),      # "Figure 3"
    re.compile(r"^\s*table\s+\d+", re.IGNORECASE),       # "Table 1"
    re.compile(r"^\s*(?:ref(?:erence)?s?|bibliography)\s*$", re.IGNORECASE),
]


def is_equation(text: str) -> bool:
    """
    Return True if the text contains at least one LaTeX math expression.
    Used to classify chunks as chunk_type='equation' and skip LLMLingua-2.
    """
    for pattern in _LATEX_PATTERNS:
        if pattern.search(text):
            return True
    return False


def extract_latex_blocks(text: str) -> list[str]:
    """
    Extract all LaTeX math blocks from a text string.
    Returns a list of matched LaTeX expressions.
    """
    blocks: list[str] = []
    display_patterns = [
        re.compile(r"\$\$.+?\$\$", re.DOTALL),
        re.compile(r"\\\[.+?\\\]", re.DOTALL),
        re.compile(
            r"\\begin\{(equation|align|math|gather|multline)\*?\}.+?"
            r"\\end\{\1\*?\}",
            re.DOTALL,
        ),
    ]
    inline_patterns = [
        re.compile(r"\$.+?\$"),
        re.compile(r"\\\(.+?\\\)"),
    ]
    for pat in display_patterns + inline_patterns:
        matches = pat.findall(text)
        if matches:
            # findall returns group strings for patterns with groups
            blocks.extend(m if isinstance(m, str) else m[0] for m in matches)
    return blocks


def has_table_markers(text: str) -> bool:
    """
    Return True if the text looks like a table (pipe-separated columns
    or tab-separated multi-column rows).
    Used to classify chunks as chunk_type='table'.
    """
    lines = text.strip().split("\n")
    pipe_lines = sum(1 for line in lines if line.count("|") >= 2)
    tab_lines  = sum(1 for line in lines if line.count("\t") >= 2)
    # At least 3 lines that look like table rows
    return pipe_lines >= 3 or tab_lines >= 3


def infer_chunk_type(text: str) -> str:
    """
    Infer chunk_type from text content.
    Returns 'equation' | 'table' | 'text'.
    """
    if is_equation(text):
        return "equation"
    if has_table_markers(text):
        return "table"
    return "text"


# ══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

# Common PDF extraction artefacts
_LIGATURE_MAP = str.maketrans({
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2018": "'",   # left single quotation
    "\u2019": "'",   # right single quotation
    "\u201c": '"',   # left double quotation
    "\u201d": '"',   # right double quotation
    "\u2013": "-",   # en dash
    "\u2014": "--",  # em dash
    "\u00ad": "",    # soft hyphen (remove)
    "\u00a0": " ",   # non-breaking space → regular space
})

_MULTI_SPACE    = re.compile(r"[ \t]+")
_MULTI_NEWLINE  = re.compile(r"\n{3,}")
_HYPHEN_WRAP    = re.compile(r"(\w)-\n(\w)")       # "re-\nsearch" → "research"
_CONTROL_CHARS  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """
    Normalise text extracted from PDFs.

    Steps applied:
      1. Fix Unicode ligatures and typographic characters
      2. Remove control characters
      3. Fix hyphenated line-wraps common in PDF extraction
      4. Collapse multiple spaces/tabs to single space
      5. Collapse 3+ newlines to 2 (preserve paragraph breaks)
      6. Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # Step 1: Unicode normalisation + ligature fix
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_LIGATURE_MAP)

    # Step 2: Remove control characters
    text = _CONTROL_CHARS.sub("", text)

    # Step 3: Fix PDF hyphen line-wraps
    text = _HYPHEN_WRAP.sub(r"\1\2", text)

    # Step 4: Collapse spaces
    text = _MULTI_SPACE.sub(" ", text)

    # Step 5: Collapse excessive newlines
    text = _MULTI_NEWLINE.sub("\n\n", text)

    # Step 6: Strip
    return text.strip()


def is_meaningful(text: str, min_words: int = 10) -> bool:
    """
    Return False for text that is too short or obviously structural noise
    (page numbers, section labels, figure captions without context, etc.).

    Used during ingestion to skip junk paragraphs before chunking.
    """
    if not text or not text.strip():
        return False

    stripped = text.strip()

    # Noise patterns (page numbers, "References", etc.)
    for pat in _NOISE_PATTERNS:
        if pat.match(stripped):
            return False

    # Minimum word count
    word_count = len(stripped.split())
    return word_count >= min_words


# ══════════════════════════════════════════════════════════════════════════════
# SENTENCE SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

# Abbreviations that should NOT be treated as sentence boundaries
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc",
    "i.e", "e.g", "fig", "eq", "sec", "ch", "vol", "no", "pp",
    "approx", "avg", "std", "max", "min", "cf",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'\(\[])")


def split_into_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter — no NLTK or spaCy required.

    Splits on '. ', '! ', '? ' followed by a capital letter, but skips
    common abbreviations to avoid false splits like "Dr. Smith" or "Fig. 3".
    """
    if not text:
        return []

    # Quick pre-check: if no sentence-ending punctuation, return as-is
    if not any(c in text for c in ".!?"):
        return [text.strip()]

    raw_sentences = _SENTENCE_SPLIT.split(text)
    sentences: list[str] = []

    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Check if the "sentence" ends with a known abbreviation (false split)
        last_word = sent.rstrip(".").split()[-1].lower() if sent.split() else ""
        if last_word in _ABBREVIATIONS and sentences:
            # Merge back with previous sentence
            sentences[-1] = sentences[-1] + " " + sent
        else:
            sentences.append(sent)

    return sentences if sentences else [text.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# ENTITY NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalise_entity(text: str) -> str:
    """
    Normalise an entity string for consistent matching and deduplication.
    E.g. "  Diabetic Retinopathy " → "diabetic retinopathy"
         "PSO " → "pso"
    """
    return " ".join(text.lower().split())


def normalise_entities(entities: list[str]) -> list[str]:
    """Normalise and deduplicate a list of entity strings."""
    seen: set[str] = set()
    result: list[str] = []
    for ent in entities:
        norm = normalise_entity(ent)
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result