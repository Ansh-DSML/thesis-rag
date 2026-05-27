# frontend/config.py
# ─────────────────────────────────────────────────────────────────────────────
# All configuration for the Streamlit frontend.
# Import anywhere with: from config import cfg
#
# On Streamlit Community Cloud → set API_URL in the Secrets panel.
# Locally → set it in frontend/.env  or  export API_URL=http://...
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get_api_url() -> str:
    """
    Resolve API_URL in priority order:
      1. Streamlit secrets  (st.secrets["API_URL"])   ← production on Streamlit Cloud
      2. OS environment variable API_URL               ← local dev / Docker
      3. Localhost fallback                            ← bare local dev
    """
    try:
        import streamlit as st
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"].rstrip("/")
    except Exception:
        pass

    url = os.getenv("API_URL", "http://localhost:8000")
    return url.rstrip("/")


@dataclass(frozen=True)
class AppConfig:

    # ── Backend ───────────────────────────────────────────────────────────────
    api_url: str        = field(default_factory=_get_api_url)
    api_timeout: int    = 300   # seconds — Llama-3.3-70b on CPU can take 90–120 s
    api_retries: int    = 2     # retry count on transient network errors

    # ── App identity ──────────────────────────────────────────────────────────
    app_title:    str = "Thesis RAG — Ophthalmic AI Research Assistant"
    app_subtitle: str = (
        "Ask anything about the HG-MEF-Bio thesis: "
        "AMD · Diabetic Retinopathy · Glaucoma"
    )
    page_icon: str = "🔬"

    # ── Session ───────────────────────────────────────────────────────────────
    session_prefix:        str = "user"
    max_history_display:   int = 50   # messages shown before truncation

    # ── Query-type display labels and badge colours ───────────────────────────
    # Routed by Gemini-2.5-Flash; values must match your backend QueryType enum.
    query_type_labels: dict = field(default_factory=lambda: {
        "factual":           ("Factual",      "#E6F1FB", "#0C447C"),
        "comparative":       ("Comparative",  "#EEEDFE", "#3C3489"),
        "synthesis":         ("Synthesis",    "#E1F5EE", "#085041"),
        "methodology":       ("Methodology",  "#FAEEDA", "#633806"),
        "equation":          ("Equation",     "#FAECE7", "#712B13"),
        "algorithm":         ("Algorithm",    "#EAF3DE", "#27500A"),
        "methodology_trace": ("Methodology",  "#FAEEDA", "#633806"),
    })

    # ── Chapter badge colours ─────────────────────────────────────────────────
    chapter_colors: dict = field(default_factory=lambda: {
        1: ("#E6F1FB", "#0C447C"),  # Introduction
        2: ("#EEEDFE", "#3C3489"),  # Literature Review
        3: ("#FAEEDA", "#633806"),  # Methodology
        4: ("#E1F5EE", "#085041"),  # Results
        5: ("#FAECE7", "#712B13"),  # Discussion
        6: ("#EAF3DE", "#27500A"),  # Conclusion
    })

    chapter_names: dict = field(default_factory=lambda: {
        1: "Introduction",
        2: "Literature Review",
        3: "Methodology",
        4: "Results",
        5: "Discussion",
        6: "Conclusion & Future Work",
    })

    # ── Suggested starter questions ───────────────────────────────────────────
    suggested_questions: list = field(default_factory=lambda: [
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
    ])

    # ── Sidebar system info ───────────────────────────────────────────────────
    # Keep in sync with your .env values — these are display-only strings.
    system_info: dict = field(default_factory=lambda: {
        "Domain":      "Medical AI — Ophthalmic Disease Screening",
        "Diseases":    "AMD · Diabetic Retinopathy · Glaucoma",
        "Chapters":    "6 thesis chapters",
        "Papers":      "Up to 30 research papers",
        "Chunks":      "2,765 indexed chunks",
        "Embeddings":  "BAAI/bge-large-en-v1.5 (1024-dim)",
        "Retrieval":   "Hybrid vector + BM25 + Knowledge Graph",
        "Reranker":    "BAAI/bge-reranker-large (cross-encoder)",
        "Router LLM":  "Gemini-2.5-Flash",
        "Answer LLM":  "Llama-3.3-70b-versatile (Groq)",
        "Vector DB":   "Qdrant Cloud",
        "Cache":       "Redis Cloud",
    })


# Module-level singleton — import this everywhere
cfg = AppConfig()