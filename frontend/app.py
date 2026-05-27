# frontend/app.py
# ─────────────────────────────────────────────────────────────────────────────
# Main Streamlit application for the Thesis RAG system.
# Connects to the FastAPI backend via api_client.py.
#
# Run locally:
#   cd frontend
#   streamlit run app.py
#
# Run with custom backend URL:
#   API_URL=http://your-ec2-ip:8000 streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid

import streamlit as st

from api_client import ChatResponse, check_health, send_message, clear_session
from config import cfg


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be the very first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=cfg.app_title,
    page_icon=cfg.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Thesis RAG — HG-MEF-Bio Research Assistant**\n\n"
            "Production-grade RAG system over a 6-chapter PhD thesis "
            "on deep learning and bio-inspired optimisation for "
            "AMD, Diabetic Retinopathy, and Glaucoma screening.\n\n"
            "**LLMs:** Gemini-2.5-Flash (router) · Llama-3.3-70b via Groq (answers)\n"
            "**Retrieval:** Hybrid vector + BM25 + Knowledge Graph · BGE reranker\n"
            f"**Backend:** {cfg.api_url}"
        )
    },
)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Chapter citation badges ── */
.citation-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    margin: 2px 3px 2px 0;
    border: 1px solid;
}

/* ── Query type badge (Factual / Comparative / Equation etc.) ── */
.query-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 6px;
}

/* ── Metadata row below each answer ── */
.meta-row {
    font-size: 11px;
    color: #888;
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

/* ── Backend status dot in sidebar ── */
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE  — initialise all keys on first run
# ══════════════════════════════════════════════════════════════════════════════

def _init_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"{cfg.session_prefix}-{uuid.uuid4().hex[:8]}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "health" not in st.session_state:
        st.session_state.health = None
    if "waiting" not in st.session_state:
        st.session_state.waiting = False
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "cache_hits" not in st.session_state:
        st.session_state.cache_hits = 0
    if "suggested_clicked" not in st.session_state:
        st.session_state.suggested_clicked = None


_init_session()


# ══════════════════════════════════════════════════════════════════════════════
# BADGE / METADATA RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _query_type_badge(query_type: str | None) -> str:
    """Return HTML for a query-type badge (Factual, Equation, Comparative …)."""
    if not query_type:
        return ""
    label, bg, color = cfg.query_type_labels.get(
        query_type, (query_type.title(), "#F0F0F0", "#555")
    )
    return (
        f'<span class="query-badge" '
        f'style="background:{bg};color:{color};border:1px solid {color}40;">'
        f'{label}</span>'
    )


def _citation_badges(chapter_refs: list[int]) -> str:
    """Return HTML for one badge per cited chapter."""
    if not chapter_refs:
        return ""
    badges = []
    for ch in chapter_refs:
        bg, color = cfg.chapter_colors.get(ch, ("#F0F0F0", "#555"))
        name = cfg.chapter_names.get(ch, f"Chapter {ch}")
        badges.append(
            f'<span class="citation-badge" '
            f'style="background:{bg};color:{color};border-color:{color}60;">'
            f'Ch.{ch} · {name}</span>'
        )
    return "".join(badges)


def _meta_row(response: ChatResponse) -> str:
    """Return HTML for the metadata strip shown below every answer."""
    parts = []

    if response.query_type:
        parts.append(_query_type_badge(response.query_type))

    if response.cache_hit:
        # Redis cache hit → instant response
        parts.append('<span style="color:#0F6E56;font-weight:500;">⚡ Cached</span>')

    if response.latency_str:
        parts.append(f'<span>⏱ {response.latency_str}</span>')

    if response.kg_used:
        parts.append('<span title="Knowledge Graph retrieval used">🕸 KG</span>')

    if response.hyde_used:
        parts.append('<span title="HyDE query expansion used">💡 HyDE</span>')

    return (
        f'<div class="meta-row">{"".join(parts)}</div>'
        if parts else ""
    )


def _status_indicator(health) -> str:
    """Return small HTML status indicator for the sidebar header."""
    if health is None:
        return '<span class="status-dot" style="background:#ccc;"></span>Checking…'
    if health.all_ok:
        return '<span class="status-dot" style="background:#0F6E56;"></span>Online'
    if health.reachable:
        return '<span class="status-dot" style="background:#EF9F27;"></span>Degraded'
    return '<span class="status-dot" style="background:#D0342C;"></span>Offline'


def _render_answer(response: ChatResponse) -> None:
    """Render answer text, chapter citation badges, and metadata row."""
    st.markdown(response.answer)

    citations_html = _citation_badges(response.chapter_refs)
    meta_html      = _meta_row(response)

    if citations_html or meta_html:
        st.markdown(
            f'<div style="margin-top:4px">{citations_html}</div>{meta_html}',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"## {cfg.page_icon} Thesis RAG")
    st.caption(
        "HG-MEF-Bio: Bio-Inspired Hierarchical Multi-Expert Fusion "
        "for Ophthalmic Disease Classification"
    )

    st.divider()

    # ── Backend health ────────────────────────────────────────────────────────
    # Only check once per session — not on every rerun — to avoid spamming /health
    if st.session_state.health is None:
        with st.spinner("Connecting to backend…"):
            st.session_state.health = check_health()

    health = st.session_state.health
    st.markdown(
        f"**Backend:** {_status_indicator(health)}",
        unsafe_allow_html=True,
    )

    # Show extra detail when partially degraded
    if health and health.reachable and not health.all_ok:
        details = []
        if not health.qdrant_ok:
            details.append("Qdrant ✗")
        if not health.redis_ok:
            details.append("Redis ✗")
        if not health.model_ok:
            details.append("Models loading…")
        if details:
            st.caption(" · ".join(details))

    if st.button("↻ Refresh connection", use_container_width=True):
        st.session_state.health = None
        st.rerun()

    st.divider()

    # ── System info ───────────────────────────────────────────────────────────
    with st.expander("System info", expanded=False):
        for key, val in cfg.system_info.items():
            st.markdown(f"**{key}:** {val}")
        st.caption(f"Endpoint: `{cfg.api_url}`")

    st.divider()

    # ── Session controls ──────────────────────────────────────────────────────
    st.markdown("**Session**")
    st.code(st.session_state.session_id, language=None)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New session", use_container_width=True):
            # Fresh session ID = fresh Redis memory on the backend
            st.session_state.session_id    = f"{cfg.session_prefix}-{uuid.uuid4().hex[:8]}"
            st.session_state.messages      = []
            st.session_state.total_queries = 0
            st.session_state.cache_hits    = 0
            st.rerun()
    with col2:
        if st.button("Clear chat", use_container_width=True):
            # Delete Redis history for this session, keep same session ID
            clear_session(st.session_state.session_id)
            st.session_state.messages      = []
            st.session_state.total_queries = 0
            st.session_state.cache_hits    = 0
            st.rerun()

    st.divider()

    # ── Session stats ─────────────────────────────────────────────────────────
    total = st.session_state.total_queries
    hits  = st.session_state.cache_hits
    if total > 0:
        st.markdown("**This session**")
        c1, c2 = st.columns(2)
        c1.metric("Queries",    total)
        c2.metric("Cache hits", f"{hits}/{total}")
        st.divider()

    # ── Suggested questions ───────────────────────────────────────────────────
    st.markdown("**Suggested questions**")
    for q in cfg.suggested_questions[:6]:
        label = q[:58] + ("…" if len(q) > 58 else "")
        if st.button(label, use_container_width=True, key=f"sug_{q[:24]}"):
            st.session_state.suggested_clicked = q
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"### {cfg.app_title}")
st.caption(cfg.app_subtitle)

# Warn the user if backend is unreachable
if st.session_state.health and not st.session_state.health.reachable:
    st.error(
        f"Cannot reach backend at **{cfg.api_url}**. "
        "Make sure the FastAPI server is running on EC2, then click "
        "**Refresh connection** in the sidebar."
    )

# ── Render conversation history ───────────────────────────────────────────────
for msg in st.session_state.messages[-cfg.max_history_display:]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "response_obj" in msg:
            # Re-render with badges and metadata
            _render_answer(msg["response_obj"])
        else:
            st.markdown(msg["content"])


# ══════════════════════════════════════════════════════════════════════════════
# QUERY PROCESSOR  — shared by chat input and suggested-question buttons
# ══════════════════════════════════════════════════════════════════════════════

def _process_query(query: str) -> None:
    """Send query to backend, stream response into chat, update state."""
    # Append user message immediately so the UI feels responsive
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner(
            "Thinking…  "
            "*(Gemini routes → HyDE → hybrid retrieval → BGE rerank → Groq LLM)*"
        ):
            response = send_message(query, st.session_state.session_id)

        if response.error:
            st.error(f"**Error:** {response.error}")
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"Error: {response.error}",
            })
        else:
            _render_answer(response)
            st.session_state.messages.append({
                "role":         "assistant",
                "content":      response.answer,
                "response_obj": response,   # kept for re-render on history scroll
            })
            st.session_state.total_queries += 1
            if response.cache_hit:
                st.session_state.cache_hits += 1


# ── Fire suggested question if a sidebar button was clicked ───────────────────
if st.session_state.suggested_clicked:
    query = st.session_state.suggested_clicked
    st.session_state.suggested_clicked = None
    _process_query(query)

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input(
    "Ask anything about the thesis — equations, algorithms, results, comparisons…",
    disabled=st.session_state.waiting,
):
    _process_query(user_input)


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE  — shown only when no messages exist yet
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.messages:
    st.markdown("---")
    st.markdown("#### Example query types")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**Factual lookup**
> *What accuracy did the AMD expert achieve in Architecture 2?*
        """)
    with col2:
        st.markdown("""
**Equation**
> *What is the PSO velocity update equation for DR threshold optimisation?*
        """)
    with col3:
        st.markdown("""
**Comparative**
> *Compare Architecture 1 HG-MEF vs Architecture 2 HG-MEF-Bio across all diseases.*
        """)

    st.markdown("---")
    st.markdown("#### Or pick a suggested question below")

    cols = st.columns(2)
    for i, q in enumerate(cfg.suggested_questions[6:]):
        with cols[i % 2]:
            if st.button(q, use_container_width=True, key=f"empty_sug_{i}"):
                st.session_state.suggested_clicked = q
                st.rerun()