"""
app/core/generation/prompts/hyde_prompt.py

Prompt 2 — HyDE (Hypothetical Document Expansion).

HyDE improves recall for non-trivial queries by generating a short expert
paragraph that LOOKS LIKE text from the thesis or papers. This paragraph is
then embedded and averaged with the original query embedding:

    final_vector = alpha × query_emb + (1 - alpha) × hyde_emb
                 (alpha=0.5 by default — equal mix)

The intuition: the generated paragraph will be semantically closer to actual
retrieved chunks than the bare user question, which is often phrased differently
from how the thesis was written.

Skipped for:
  - equation queries (exact symbolic match beats semantic expansion)
  - when use_hyde=False in RouterOutput

Token budget: keep the generated paragraph 100–150 words.
Longer = richer but slower to embed and harder to average meaningfully.
"""

from __future__ import annotations

from app.models.query import QueryType
from app.utils.logger import get_logger

log = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES  (one per query type for better domain priming)
# ══════════════════════════════════════════════════════════════════════════════

# Shared domain context injected into every HyDE prompt.
# Primes the model to generate text that matches the thesis/paper style.
_DOMAIN_PRIMER = """\
You are an expert researcher in medical image analysis, bio-inspired \
optimisation, and machine learning applied to clinical diagnostics. \
You write in the precise academic style of a doctoral thesis and \
peer-reviewed journal papers.\
"""

_HYDE_TEMPLATES: dict[str, str] = {

    "factual": """\
{domain_primer}

A researcher asks: "{query}"

Write a single paragraph of 100–150 words that directly answers this question \
as it would appear in a thesis chapter or research paper. \
Include specific values, names, or references where relevant. \
Write in third-person academic prose. Do not use headers or bullet points. \
Do not start with "In this paper" or "In this thesis". \
Go straight to the answer.
""",

    "comparative": """\
{domain_primer}

A researcher asks: "{query}"

Write a single paragraph of 100–150 words that compares the relevant \
methods, models, or results as they would be discussed in a thesis Results \
or Discussion chapter. \
Include performance metrics, trade-offs, and context where relevant. \
Write in third-person academic prose. \
Do not use tables, bullet points, or headers. \
Use precise technical terminology.
""",

    "synthesis": """\
{domain_primer}

A researcher asks: "{query}"

Write a single paragraph of 100–150 words that synthesises the key findings, \
themes, or contributions related to this topic as they would appear in a \
thesis Literature Review or Discussion chapter. \
Cover the main ideas concisely. \
Write in third-person academic prose. \
Do not use bullet points or headers.
""",

    "methodology": """\
{domain_primer}

A researcher asks: "{query}"

Write a single paragraph of 100–150 words that describes the methodology, \
experimental setup, or pipeline as it would appear in a thesis Methodology \
chapter. \
Include parameter values, dataset details, and procedural steps where relevant. \
Write in past tense, third-person academic prose. \
Do not use bullet points or headers.
""",

    "algorithm": """\
{domain_primer}

A researcher asks: "{query}"

Write a single paragraph of 100–150 words that explains the algorithm's \
mechanism, steps, and purpose as they would be described in a thesis \
Methodology chapter or a methods-focused research paper. \
Be precise about the sequence of operations and the role of each component. \
Write in academic prose. \
Do not use numbered lists or pseudocode.
""",

    # equation is excluded — HyDE is skipped for equation queries
    # but we include a fallback template just in case
    "equation": """\
{domain_primer}

A researcher asks: "{query}"

Write a single paragraph of 100–150 words that contextualises the \
mathematical formulation being asked about. \
Describe what the formula computes, why it is used, and how its components \
relate to the broader methodology. \
Write in academic prose. Do not attempt to reproduce the equation itself.
""",
}

# Default template used if query_type is unknown
_DEFAULT_TEMPLATE = _HYDE_TEMPLATES["factual"]


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_hyde_prompt(query: str, query_type: QueryType) -> str:
    """
    Build the HyDE prompt string for a given query and query type.

    Args:
        query:      The user's original query string.
        query_type: Query type from RouterOutput — selects the right template.

    Returns:
        Prompt string to send to Gemini (llm_model_main).
        The response is the hypothetical paragraph — embed it directly.
    """
    template = _HYDE_TEMPLATES.get(str(query_type), _DEFAULT_TEMPLATE)
    prompt = template.format(
        domain_primer=_DOMAIN_PRIMER,
        query=query.strip(),
    )
    log.debug("hyde_prompt_built", query_type=query_type)
    return prompt


def should_run_hyde(query_type: QueryType, use_hyde_flag: bool) -> bool:
    """
    Determine whether HyDE should run for this query.

    HyDE is always skipped for equation queries — exact keyword matching
    (BM25) and formula-specific embedding are more effective than semantic
    expansion for symbolic mathematical content.

    Args:
        query_type:    Classified type from RouterOutput.
        use_hyde_flag: The use_hyde flag from RouterOutput.

    Returns:
        True if HyDE should generate and embed a paragraph.
    """
    if query_type == "equation":
        log.debug("hyde_skipped", reason="equation_query")
        return False
    if not use_hyde_flag:
        log.debug("hyde_skipped", reason="router_disabled")
        return False
    return True