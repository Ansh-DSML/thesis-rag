"""
app/core/generation/prompts/compression_prompt.py

Prompt 3 — LLMLingua-2 Context Compression Instruction.

This is NOT a conversational LLM prompt — it is an instruction string
passed to the LLMLingua-2 compressor telling it what to preserve verbatim
during compression.

LLMLingua-2 compresses retrieved context from ~3000+ tokens down to ~50%
by removing lower-information tokens. Without guidance it may discard:
  - Exact numbers / accuracy values (critical for factual answers)
  - Mathematical equations (must be reproduced exactly)
  - Table cell values (removing a number breaks the table)
  - Citation keys like [Chapter 3] (needed for answer attribution)
  - Metric names and units (e.g. "94.2% AUC" must stay as-is)

This instruction string is passed to LLMLingua-2's `condition_in_question`
parameter so the compressor biases toward retaining these elements.

Chunks that bypass compression entirely (re-injected verbatim at the front):
  - chunk_type == "equation"
  - chunk_type == "table"
These are handled in compressor.py, not here.

Usage (in compressor.py):
    from app.core.generation.prompts.compression_prompt import (
        get_compression_instruction,
        get_compression_config,
    )
    instruction = get_compression_instruction(query_type)
    config = get_compression_config(query_type)
"""

from __future__ import annotations

from app.models.query import QueryType

# ══════════════════════════════════════════════════════════════════════════════
# BASE INSTRUCTION  (always included)
# ══════════════════════════════════════════════════════════════════════════════

_BASE_INSTRUCTION = """\
Compress the following academic context while strictly preserving:
1. All numerical values, percentages, and measurement units exactly as written
2. All mathematical equations, formulas, and symbolic expressions verbatim
3. All citation keys in brackets: [Chapter N] and [Author Year]
4. All table cell values and column headers
5. All named entities: algorithm names, dataset names, medical condition names
6. All metric names: accuracy, AUC, F1, precision, recall, sensitivity, specificity
7. The logical structure of arguments and causal relationships
Remove: transitional phrases, redundant restatements, generic background sentences.\
"""

# ══════════════════════════════════════════════════════════════════════════════
# QUERY-TYPE ADDITIONS  (appended to base for specific types)
# ══════════════════════════════════════════════════════════════════════════════

_TYPE_ADDITIONS: dict[str, str] = {
    "factual": (
        " Prioritise sentences containing specific facts, numbers, "
        "and direct answers over background context."
    ),
    "comparative": (
        " Prioritise sentences that explicitly compare two or more methods, "
        "including performance deltas, ranked results, and stated trade-offs. "
        "Preserve parallel structures used for comparison."
    ),
    "synthesis": (
        " Preserve topic sentences and concluding sentences of each paragraph "
        "as these carry the highest information density for synthesis."
    ),
    "methodology": (
        " Prioritise sentences describing experimental steps, parameter values, "
        "dataset splits, hardware configuration, and training settings. "
        "Preserve ordered sequences — do not reorder procedural steps."
    ),
    "equation": (
        " CRITICAL: Every equation, formula, and mathematical expression in "
        "this context must be preserved 100% verbatim — including all LaTeX "
        "notation, subscripts, superscripts, and operator symbols. "
        "Preserve variable definitions immediately following each equation."
    ),
    "algorithm": (
        " Preserve algorithmic steps in their original order. "
        "Do not merge or reorder numbered or sequential steps. "
        "Preserve pseudocode blocks and conditional logic exactly."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_compression_instruction(query_type: QueryType) -> str:
    """
    Return the complete compression instruction string for a given query type.

    This string is passed to LLMLingua-2 as the `condition_in_question`
    parameter — it biases token retention toward elements listed here.

    Args:
        query_type: From RouterOutput — controls which additions are appended.

    Returns:
        Instruction string for LLMLingua-2.
    """
    addition = _TYPE_ADDITIONS.get(str(query_type), "")
    return _BASE_INSTRUCTION + addition


def get_compression_config(query_type: QueryType) -> dict:
    """
    Return LLMLingua-2 configuration parameters for a given query type.

    Equation and algorithm queries use a higher target rate (keep more)
    because their content is dense and precision-critical.
    Synthesis queries use a lower rate (compress more aggressively)
    because they draw from many sources and context is longer.

    Returns a dict ready to unpack into the LLMLingua-2 compressor call:
        compressor.compress_prompt(**get_compression_config(query_type))

    Fields:
        rate           : Fraction of tokens to KEEP (0.5 = keep 50%)
        force_tokens   : List of token patterns always kept verbatim
        drop_consecutive: Whether to drop consecutive low-info sentences
    """
    # Base config
    config: dict = {
        "rate": 0.5,
        "force_tokens": [
            # Citation key patterns — always keep
            "[Chapter", "[chapter",
            # Common equation delimiters
            "$$", "\\[", "\\]", "\\begin", "\\end",
            # Critical metric tokens
            "accuracy", "AUC", "F1", "precision", "recall",
            "sensitivity", "specificity", "loss",
        ],
        "drop_consecutive": True,
        "condition_in_question": get_compression_instruction(query_type),
    }

    # Type-specific overrides
    if query_type == "equation":
        # Keep more context for equations — formula + variable definitions
        config["rate"] = 0.7
        config["drop_consecutive"] = False

    elif query_type == "algorithm":
        # Keep more — step sequences must stay intact
        config["rate"] = 0.65
        config["drop_consecutive"] = False

    elif query_type == "synthesis":
        # Compress more aggressively — synthesis context is large
        config["rate"] = 0.45

    elif query_type == "comparative":
        # Moderate — keep comparison sentences
        config["rate"] = 0.55

    return config


def should_compress(context_token_count: int, token_threshold: int = 3000) -> bool:
    """
    Return True if the context is long enough to warrant compression.

    Compression adds latency (~200ms). For short contexts it's not worth it.

    Args:
        context_token_count: Approximate token count of the full context string.
        token_threshold:     From settings.compression_token_threshold (default 3000).

    Returns:
        True if compression should run.
    """
    return context_token_count > token_threshold