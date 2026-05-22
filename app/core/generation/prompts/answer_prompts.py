"""
app/core/generation/prompts/answer_prompts.py

Prompt 4 — Answer Generation Templates.

Six sub-templates, one per query type. Each is optimised for the kind of
answer that query type demands:

  factual     → direct, cited, concise — one clear answer
  comparative → structured side-by-side with metrics and trade-offs
  synthesis   → comprehensive overview with sections
  methodology → past-tense pipeline walkthrough with parameters
  equation    → equation reproduced verbatim + variable definitions
  algorithm   → numbered steps with purpose of each

get_answer_prompt(query_type) returns the right template string.
build_answer_prompt(...)       injects all runtime variables and returns
                               the complete prompt ready for the LLM.
"""

from __future__ import annotations

from app.models.query import QueryType
from app.utils.logger import get_logger

log = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED INJECTION BLOCK
# Prepended to every template — provides context, history, and citation index.
# ══════════════════════════════════════════════════════════════════════════════

_CONTEXT_BLOCK = """\
━━━ RETRIEVED CONTEXT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The following passages were retrieved from the thesis and research papers.
Each passage is labelled with its citation key. Use ONLY these passages.

{context}

━━━ AVAILABLE CITATION KEYS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{citation_keys}

━━━ CONVERSATION HISTORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chat_history}

━━━ CURRENT QUESTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{query}

"""

# ══════════════════════════════════════════════════════════════════════════════
# SUB-TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

_TEMPLATES: dict[str, str] = {

# ── 1. Factual ────────────────────────────────────────────────────────────────
"factual": """\
{context_block}
━━━ INSTRUCTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Answer the question directly and specifically using ONLY the retrieved context.

Rules:
• Open with the direct answer in the first sentence — do not build up to it.
• Cite every factual claim immediately after it: [Chapter N] or [Author Year].
• If the answer is a number, percentage, or metric — quote it exactly as it
  appears in the context. Do not round or paraphrase.
• Length: 2–4 sentences for simple facts; up to 2 short paragraphs for
  multi-part answers.
• If the context does not contain the answer, say:
  "The retrieved content does not contain sufficient information to answer
  this question."
• Do not add background knowledge not present in the retrieved passages.
""",

# ── 2. Comparative ────────────────────────────────────────────────────────────
"comparative": """\
{context_block}
━━━ INSTRUCTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare the methods, models, or results asked about using ONLY the retrieved context.

Structure your answer as follows:
1. **Opening** (1–2 sentences): State what is being compared and why it matters.
2. **[Method/Model A]**: Key characteristics, performance metrics, strengths,
   and limitations. Cite all claims: [Chapter N] or [Author Year].
3. **[Method/Model B]** (and C, D if present): Same structure as above.
4. **Summary**: 2–3 sentences stating the overall comparative conclusion,
   which method performs better and under what conditions.

Rules:
• Use exact numbers from the context — do not approximate or estimate.
• If a metric is only available for one method, note that explicitly rather
  than inventing a value for the other.
• Cite every claim. A comparison without citations will be rejected.
• Do not introduce methods or results not present in the retrieved context.
""",

# ── 3. Synthesis ──────────────────────────────────────────────────────────────
"synthesis": """\
{context_block}
━━━ INSTRUCTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provide a comprehensive, well-structured overview of the topic using ONLY
the retrieved context.

Structure your answer as follows:
## Overview
2–3 sentences introducing the topic and its significance in the context of
the thesis.

## Key Themes / Findings
Cover the main themes or findings drawn from the retrieved passages.
Group related ideas under sub-points where helpful.
Cite every claim: [Chapter N] or [Author Year].

## Synthesis
2–3 sentences drawing the retrieved ideas together — what do they collectively
tell us? Identify patterns, agreements, or tensions between sources.

Rules:
• Prioritise breadth — cover all relevant aspects present in the context.
• Do not repeat the same point multiple times.
• Use the exact terminology from the retrieved passages for technical concepts.
• Every paragraph must contain at least one citation.
• If sources contradict each other, note the contradiction explicitly.
""",

# ── 4. Methodology ────────────────────────────────────────────────────────────
"methodology": """\
{context_block}
━━━ INSTRUCTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Describe the methodology, pipeline, or experimental setup using ONLY the
retrieved context. Write in past tense (the research was conducted).

Structure your answer as follows:
## Overview
1–2 sentences: what the methodology aims to achieve and its position in the
overall research pipeline.

## Pipeline / Steps
Describe each stage or component in the order it appears in the retrieved
context. For each stage include:
  - What was done
  - Why (the rationale, if stated)
  - Key parameter values, dataset details, or settings
  - Citation: [Chapter N] or [Author Year]

## Implementation Details
Any relevant hardware, software, library versions, or computational settings
mentioned in the context.

Rules:
• Preserve every parameter value exactly (e.g. "learning rate = 0.001",
  "population size = 30"). Do not round or omit.
• Maintain the original order of steps — do not reorder the pipeline.
• Write in past tense throughout ("was applied", "were split", "achieved").
• Every step must have a citation.
""",

# ── 5. Equation ───────────────────────────────────────────────────────────────
"equation": """\
{context_block}
━━━ INSTRUCTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Answer the question about the mathematical formulation using ONLY the
retrieved context.

Structure your answer as follows:
**Equation:**
Reproduce the equation EXACTLY as it appears in the retrieved context.
Preserve all LaTeX notation, subscripts, superscripts, and operator symbols.
Do not simplify, rearrange, or paraphrase. [Cite source: Chapter N or Author Year]

**Variable Definitions:**
Define every variable, constant, and subscript that appears in the equation:
  • α — [definition exactly as stated in context]
  • N — [definition exactly as stated in context]
  (continue for every symbol)

**Purpose:**
1–2 sentences: what does this equation compute and why is it used in the
methodology?

**Context:**
1–2 sentences: where in the pipeline is this equation applied and what
are its inputs/outputs?

CRITICAL RULES:
• The equation must be reproduced character-for-character. Any change — even
  a sign or exponent — is a factual error.
• If the context contains multiple related equations, present all of them
  in the order they appear.
• Never invent variable definitions not stated in the retrieved context.
• If the equation is not present in the context, state this explicitly.
""",

# ── 6. Algorithm ──────────────────────────────────────────────────────────────
"algorithm": """\
{context_block}
━━━ INSTRUCTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Explain the algorithm using ONLY the retrieved context.

Structure your answer as follows:
## Purpose
1–2 sentences: what problem does this algorithm solve and why was it chosen?
[Chapter N] or [Author Year]

## Algorithm Steps
Present each step in numbered order exactly as described in the retrieved
context. For each step:
  1. **[Step name]**: What happens, what inputs are used, what is produced.
     [Cite: Chapter N or Author Year]
  2. Continue for every step...

## Key Parameters
List all parameters mentioned in the context with their values:
  • Parameter name: value (unit if applicable) [citation]

## Termination / Output
What is the stopping criterion and what does the algorithm output?

Rules:
• Preserve the original step order — do not reorder or merge steps.
• If the context contains pseudocode, reproduce the logic faithfully in prose
  (do not invent pseudocode not in the source).
• All parameter values must be exact — never estimate or round.
• Every numbered step must have a citation.
• If a step is described across multiple retrieved chunks, integrate them
  while citing each source.
""",
}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_answer_prompt(query_type: QueryType) -> str:
    """
    Return the raw template string for a given query type.
    Useful for inspection or testing individual templates.
    """
    return _TEMPLATES.get(str(query_type), _TEMPLATES["factual"])


def build_answer_prompt(
    query: str,
    query_type: QueryType,
    formatted_context: str,
    citation_keys: list[str],
    chat_history: str = "",
) -> str:
    """
    Build the complete Prompt 4 string ready to send to Gemini.

    Args:
        query:             The user's question.
        query_type:        From RouterOutput — selects the right sub-template.
        formatted_context: Pre-formatted string of retrieved chunk text.
                           Built by answer_generator.py from ScoredChunks.
        citation_keys:     List of citation key strings available in context.
                           e.g. ["[Chapter 3]", "[Zhang 2022]", "[Chapter 4]"]
        chat_history:      Formatted conversation history string from
                           ChatMemory.get_formatted(). Empty string for new sessions.

    Returns:
        Complete prompt string for the LLM call.
    """
    template = _TEMPLATES.get(str(query_type), _TEMPLATES["factual"])

    # Format citation keys as a readable list
    if citation_keys:
        keys_str = "\n".join(f"  • {key}" for key in sorted(set(citation_keys)))
    else:
        keys_str = "  (no citation keys available — state this in your answer)"

    # Format chat history block
    if chat_history and chat_history.strip():
        history_str = chat_history.strip()
    else:
        history_str = "(no previous conversation — this is the first message)"

    # Build the shared context block
    context_block = _CONTEXT_BLOCK.format(
        context=formatted_context.strip(),
        citation_keys=keys_str,
        chat_history=history_str,
        query=query.strip(),
    )

    # Inject context block into selected sub-template
    full_prompt = template.format(context_block=context_block)

    log.debug(
        "answer_prompt_built",
        query_type=query_type,
        context_chars=len(formatted_context),
        citation_count=len(citation_keys),
        has_history=bool(chat_history.strip()),
    )

    return full_prompt


def format_chunks_for_context(chunks: list) -> tuple[str, list[str]]:
    """
    Format a list of ScoredChunks into a readable context string
    and collect all citation keys.

    Args:
        chunks: list[ScoredChunk] — the reranked top-k chunks.

    Returns:
        Tuple of:
          - formatted_context: multi-line string with labelled passages
          - citation_keys:     list of unique citation key strings
    """
    parts: list[str] = []
    citation_keys: list[str] = []

    for i, sc in enumerate(chunks, start=1):
        chunk = sc.chunk
        key = chunk.citation_key
        citation_keys.append(key)

        # Build source label
        if chunk.source_type == "thesis":
            label = f"Chapter {chunk.chapter_num}: {chunk.chapter_name}"
            if chunk.section_heading:
                label += f" | Section: {chunk.section_heading}"
        else:
            authors = ", ".join(chunk.paper_authors[:2]) if chunk.paper_authors else "Unknown"
            if len(chunk.paper_authors) > 2:
                authors += " et al."
            year = chunk.paper_year or "n.d."
            label = f"{authors} ({year})"
            if chunk.section_heading:
                label += f" | {chunk.section_heading}"

        parts.append(
            f"[PASSAGE {i} | Citation: {key} | Source: {label}]\n"
            f"{chunk.text}\n"
        )

    formatted = "\n".join(parts)
    return formatted, citation_keys