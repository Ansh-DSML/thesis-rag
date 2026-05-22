"""
app/core/generation/prompts/router_prompt.py

Prompt 1 — Query Router.

Cheapest and fastest LLM call in the pipeline (uses llm_model_router).
Runs on EVERY query before any retrieval happens.

Responsibilities:
  1. Classify query into one of 6 types
  2. Extract domain entities (algorithms, diseases, loss functions, metrics)
  3. Set routing flags: use_graph, use_hyde
  4. Set optional filters: source_filter, chapter_filter

Output is pure JSON — parsed into RouterOutput (app/models/query.py).
If parsing fails, RouterOutput.fallback() is used so the pipeline never crashes.

Token budget: keep this prompt under 600 tokens.
The query itself + last conversation turn are appended at call time.
"""

from __future__ import annotations

import json
import re

from app.models.query import RouterOutput
from app.utils.logger import get_logger

log = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

ROUTER_PROMPT_TEMPLATE = """\
You are a query classification engine for an academic research assistant.
Analyse the user query and return a single JSON object — no other text.

━━━ QUERY TYPES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
factual      — Asks for a specific fact, number, name, or definition.
               Example: "What accuracy did the model achieve?"
comparative  — Compares two or more methods, models, or results.
               Example: "How does PSO compare to GA for feature selection?"
synthesis    — Requests a broad overview or summary of a topic.
               Example: "Summarise the findings on diabetic retinopathy detection."
methodology  — Asks how something was done: pipeline, setup, parameters, steps.
               Example: "What preprocessing steps were applied to the images?"
equation     — Asks for a formula, loss function, or mathematical expression.
               Example: "What is the objective function used in PSO?"
algorithm    — Asks for a step-by-step algorithm description or pseudocode.
               Example: "Explain the HyDE retrieval algorithm step by step."

━━━ ENTITY TYPES TO EXTRACT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract named entities from the query that can narrow retrieval:
  • Algorithms / optimisers    : "PSO", "GA", "Adam", "ResNet"
  • Medical conditions         : "diabetic retinopathy", "glaucoma", "DR"
  • Loss functions / metrics   : "focal loss", "F1 score", "AUC", "MSE"
  • Dataset names              : "DRIVE", "STARE", "IDRiD"
  • Specific methods / concepts: "feature selection", "SMOTE", "HNSW"
  • Chapter references         : if query mentions "methodology chapter" → chapter_filter=3

━━━ ROUTING FLAGS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
use_graph  → true  if entities were found AND query is about relationships
                    between concepts (e.g. "how does X relate to Y")
           → false for broad synthesis, factual lookups, or equation queries

use_hyde   → true  for factual, comparative, synthesis, methodology, algorithm
           → ALWAYS false for equation (exact symbolic match needed)

source_filter → "thesis"  if query is clearly about the author's own work
                           ("in your thesis", "your methodology", "your results")
             → "paper"   if query is about external literature
                           ("what do papers say", "according to the literature")
             → null      if both sources are relevant (most cases)

chapter_filter → integer 1-6 if query references a specific chapter, else null
  1=Introduction, 2=Literature Review, 3=Methodology, 4=Results,
  5=Discussion, 6=Conclusion

━━━ OUTPUT FORMAT (strict JSON, no markdown, no explanation) ━━━━━━━━━━━━━━━
{{
  "query_type": "<one of: factual|comparative|synthesis|methodology|equation|algorithm>",
  "entities": ["<entity1>", "<entity2>"],
  "use_graph": <true|false>,
  "use_hyde": <true|false>,
  "source_filter": <"thesis"|"paper"|null>,
  "chapter_filter": <1|2|3|4|5|6|null>
}}

━━━ EXAMPLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Query: "What loss function was used during training?"
Output: {{"query_type":"equation","entities":["loss function"],"use_graph":false,"use_hyde":false,"source_filter":"thesis","chapter_filter":3}}

Query: "Compare PSO and GA for feature selection performance"
Output: {{"query_type":"comparative","entities":["PSO","GA","feature selection"],"use_graph":true,"use_hyde":true,"source_filter":null,"chapter_filter":null}}

Query: "Summarise what the literature says about diabetic retinopathy detection"
Output: {{"query_type":"synthesis","entities":["diabetic retinopathy"],"use_graph":false,"use_hyde":true,"source_filter":"paper","chapter_filter":null}}

Query: "What was the test accuracy on the DRIVE dataset?"
Output: {{"query_type":"factual","entities":["DRIVE","accuracy"],"use_graph":false,"use_hyde":true,"source_filter":"thesis","chapter_filter":4}}

━━━ CONVERSATION CONTEXT (last turn, may be empty) ━━━━━━━━━━━━━━━━━━━━━━━━
{last_turn}

━━━ USER QUERY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{query}

Return ONLY the JSON object. No markdown. No explanation. No code fences.
"""


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_router_prompt(query: str, last_turn: dict | None = None) -> str:
    """
    Build the complete router prompt string for the LLM call.

    Args:
        query:     The user's raw query string.
        last_turn: The most recent conversation turn dict
                   {"human": ..., "ai": ...} or None for new sessions.
                   Helps the router classify follow-up queries correctly.
    Returns:
        Complete prompt string ready to send to Gemini.
    """
    if last_turn:
        last_turn_str = (
            f"Previous user message : {last_turn['human']}\n"
            f"Previous assistant reply: {last_turn['ai'][:200]}..."
            if len(last_turn.get("ai", "")) > 200
            else f"Previous user message : {last_turn['human']}\n"
                 f"Previous assistant reply: {last_turn.get('ai', '')}"
        )
    else:
        last_turn_str = "(none — this is the first message in the session)"

    return ROUTER_PROMPT_TEMPLATE.format(
        query=query.strip(),
        last_turn=last_turn_str,
    )


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT PARSER
# ══════════════════════════════════════════════════════════════════════════════

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_router_output(raw_response: str, query: str) -> RouterOutput:
    """
    Parse the LLM's raw text response into a RouterOutput Pydantic model.

    Handles:
      - Clean JSON (ideal case)
      - JSON wrapped in ```json ... ``` fences (model sometimes adds these)
      - Partial/malformed JSON (falls back to RouterOutput.fallback())

    Args:
        raw_response: Raw text string returned by the LLM.
        query:        Original query string (used for fallback logging).
    Returns:
        RouterOutput instance — never raises.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to find a JSON object if there's surrounding text
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        text = text[brace_start : brace_end + 1]

    try:
        data = json.loads(text)
        router_output = RouterOutput.model_validate(data)
        log.debug(
            "router_parsed",
            query_type=router_output.query_type,
            entities=router_output.entities,
            use_graph=router_output.use_graph,
            use_hyde=router_output.use_hyde,
        )
        return router_output

    except (json.JSONDecodeError, ValueError, Exception) as exc:
        log.warning(
            "router_parse_failed",
            error=str(exc),
            raw_preview=raw_response[:100],
            query_preview=query[:60],
        )
        return RouterOutput.fallback(query)