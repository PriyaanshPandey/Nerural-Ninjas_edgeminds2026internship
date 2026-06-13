"""
Scholar Minds – AI Research Assistant
Streamlit UI — single-file multi-page application.
"""

import streamlit as st
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# ── Path Setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHROMA_DIR = str(ROOT / "chroma_db")
PAPERS_DIR = str(ROOT / "uploaded_papers")
CHATS_FILE = str(ROOT / "saved_chats.json")

os.makedirs(PAPERS_DIR, exist_ok=True)


# ── Page Config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Scholar Minds · AI Research Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "page": "home",
        "papers": [],           # [{name, upload_time, num_chunks, num_pages, path}]
        "chat_history": [],     # [{role, content, timestamp}]
        "saved_chats": [],      # [{name, messages, timestamp}]
        "pending_question": "", # question passed from home → ask page
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Load saved chats from disk
    if not st.session_state.saved_chats and os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r", encoding="utf-8") as fh:
                st.session_state.saved_chats = json.load(fh)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  RAG BACKEND HELPERS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_retriever():
    """Create (or reconnect to) the ChromaDB retriever — cached across reruns."""
    try:
        from rag.retriever import ChromaRetriever
        return ChromaRetriever(
            persist_dir=CHROMA_DIR,
            collection_name="papers",
            embed_model="nomic-embed-text",
        )
    except Exception as exc:
        st.error(f"⚠️ Could not initialise retriever: {exc}")
        return None


def process_paper(uploaded_file):
    """Parse → chunk → embed → index an uploaded PDF. Returns metadata dict."""
    from rag.pdf_parser import read_pdf
    from rag.chunking import make_chunks

    # Persist the uploaded file
    dest = os.path.join(PAPERS_DIR, uploaded_file.name)
    with open(dest, "wb") as fh:
        fh.write(uploaded_file.getbuffer())

    pages = read_pdf(dest)
    docs = [{"source": uploaded_file.name, "text": p["text"], "page": p["page"]} for p in pages]
    chunks = make_chunks(docs)

    retriever = get_retriever()
    if retriever:
        retriever.index_chunks(chunks)

    return {
        "name": uploaded_file.name,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_chunks": len(chunks),
        "num_pages": len(pages),
        "path": dest,
    }


def ask_rag(question: str, source_filter: str = None, model: str = "llama3.2:1b") -> str:
    """Run a single-question RAG query and return the answer string."""
    from rag.generator import summarize_question

    retriever = get_retriever()
    if retriever is None:
        return "⚠️ Retriever not available. Make sure Ollama is running and dependencies are installed."

    chunks = retriever.query(question, top_k=5, source_filter=source_filter)
    if not chunks:
        return "No relevant content found. Please upload research papers first."

    return summarize_question(question, chunks, model=model)


def _persist_chats():
    """Write saved_chats list to disk as JSON."""
    try:
        with open(CHATS_FILE, "w", encoding="utf-8") as fh:
            json.dump(st.session_state.saved_chats, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
/* ── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ──────────────────────────────────────────────── */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stTextInput label, .stButton button,
.stRadio label, .stSelectbox label,
.stFileUploader label {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp {
    background-color: #F5F7FA;
}

/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer,
.stDeployButton,
div[data-testid="stToolbar"],
div[data-testid="stStatusWidget"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}

/* Hide sidebar collapse/expand buttons to disable sidebar hiding feature */
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"],
button[aria-label="Collapse sidebar"],
[data-testid="baseButton-header"] {
    display: none !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E8EAF0 !important;
}

section[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 240px !important;
    max-width: 240px !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.6rem !important;
}

/* Radio-as-nav: hide the group label */
section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none !important;
}

/* Radio-as-nav: hide circles / checkmarks */
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] {
    /* keep text */
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
    gap: 2px !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
    background: transparent !important;
    border-radius: 10px !important;
    padding: 11px 18px !important;
    margin: 0 10px !important;
    cursor: pointer !important;
    transition: background 0.2s ease, color 0.2s ease !important;
    color: #4B5563 !important;
    font-weight: 500 !important;
    font-size: 0.93rem !important;
    border: none !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label p {
    color: #4B5563 !important;
    transition: color 0.2s ease !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    background: #EEF2FF !important;
    color: #3B6BF5 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover p {
    color: #3B6BF5 !important;
}

/* Checked / active item */
section[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
    background: #3B6BF5 !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] p {
    color: #FFFFFF !important;
}

/* Hide radio dot */
section[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* ── Cards ───────────────────────────────────────────────── */
.scholar-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
    border: 1px solid #F0F1F3;
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}
.scholar-card:hover {
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}

/* ── Buttons (global) ────────────────────────────────────── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    font-size: 0.9rem !important;
}

/* Primary */
div[data-testid="stButton"] > button[kind="primary"],
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3B6BF5, #5B85F7) !important;
    color: #fff !important;
    border: none !important;
    padding: 8px 22px !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2E5AD4, #4A74E6) !important;
    box-shadow: 0 4px 14px rgba(59,107,245,0.35) !important;
}

/* Secondary */
.stButton > button[kind="secondary"] {
    background: #fff !important;
    color: #3B6BF5 !important;
    border: 1.5px solid #3B6BF5 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #EEF2FF !important;
}

/* ── Text inputs ─────────────────────────────────────────── */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #E5E7EB !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    background: #F9FAFB !important;
    color: #1A1A2E !important;
    caret-color: #1A1A2E !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3B6BF5 !important;
    box-shadow: 0 0 0 3px rgba(59,107,245,0.12) !important;
}

/* ── File uploader ───────────────────────────────────────── */
[data-testid="stFileUploader"] section {
    border-radius: 14px !important;
    border: 2px dashed #CBD5E1 !important;
    background: #F8FAFC !important;
    padding: 2rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #3B6BF5 !important;
}

/* ── Expander ────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E8EAF0 !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span {
    color: #1A1A2E !important;
}

div[data-testid="stExpander"] summary svg {
    fill: #1A1A2E !important;
    color: #1A1A2E !important;
}

.streamlit-expanderHeader {
    background: transparent !important;
    color: #1A1A2E !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-radius: 10px !important;
}

.streamlit-expanderHeader p {
    color: #1A1A2E !important;
}

.streamlit-expanderHeader:hover {
    background: #F9FAFB !important;
}

.streamlit-expanderHeader svg {
    fill: #1A1A2E !important;
    color: #1A1A2E !important;
}

/* ── Metrics ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #fff;
    padding: 18px 20px;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid #F0F1F3;
}
[data-testid="stMetric"] label {
    color: #6B7280 !important;
    font-weight: 500 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1A1A2E !important;
    font-weight: 700 !important;
}

/* ── Pill / example-question buttons ─────────────────────── */
.pill-btn .stButton > button {
    background: #FFFFFF !important;
    color: #3B6BF5 !important;
    border: 1.5px solid #BFD0FC !important;
    border-radius: 24px !important;
    padding: 8px 18px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}
.pill-btn .stButton > button:hover {
    background: #EEF2FF !important;
    border-color: #3B6BF5 !important;
}

/* ── Spinner ─────────────────────────────────────────────── */
.stSpinner > div {
    border-color: #3B6BF5 !important;
}

/* ── Main content max-width ──────────────────────────────── */
.main .block-container {
    max-width: 1060px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* ── About Page Redesign ─────────────────────────────────── */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}
@keyframes pulse-ring {
    0% { box-shadow: 0 0 0 0 rgba(59,107,245,0.4); }
    70% { box-shadow: 0 0 0 20px rgba(59,107,245,0); }
    100% { box-shadow: 0 0 0 0 rgba(59,107,245,0); }
}
@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes slide-up {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.about-hero-logo {
    animation: float 4s ease-in-out infinite, pulse-ring 3s infinite;
}

.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
}
.pipeline-card {
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    border: 1px solid;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
    position: relative;
}
.pipeline-card:hover {
    transform: translateY(-3px);
}

.tech-badge {
    border-radius: 12px;
    padding: 10px 20px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
    border: 1.5px solid;
}
.tech-badge:hover {
    transform: translateY(-2px);
}

.team-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
}
.team-card {
    text-align: center;
    padding: 20px 12px;
    background: #FAFBFC;
    border-radius: 16px;
    border: 1px solid #F0F1F3;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    cursor: default;
}
.team-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}
"""


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
NAV_OPTIONS = [
    "🏠  Home",
    "💬  Ask a Question",
    "📄  Upload Papers",
    "📚  My Library",
    "⭐  Saved Chats",
    "ℹ️   About",
]

NAV_KEYS = ["home", "ask", "upload", "library", "saved", "about"]


def render_sidebar():
    with st.sidebar:
        # ── Brand header ─────────────────────────────────────
        st.markdown(
            """
            <div style="padding: 12px 18px 28px 18px; display:flex; align-items:center; gap:12px;">
                <div style="
                    background: linear-gradient(135deg, #3B6BF5 0%, #6B8FF7 100%);
                    border-radius: 12px; width:42px; height:42px;
                    display:flex; align-items:center; justify-content:center;
                    box-shadow: 0 4px 12px rgba(59,107,245,0.25);
                ">
                    <span style="font-size:22px;">🎓</span>
                </div>
                <div>
                    <div style="font-weight:700; font-size:1.1rem; color:#3B6BF5; line-height:1.2;">
                        Scholar Minds
                    </div>
                    <div style="font-size:0.73rem; color:#9CA3AF; font-weight:400;">
                        AI Research Assistant
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Navigation radio ────────────────────────────────
        current_idx = NAV_KEYS.index(st.session_state.page) if st.session_state.page in NAV_KEYS else 0
        choice = st.radio(
            "Navigation",
            options=NAV_OPTIONS,
            index=current_idx,
            label_visibility="collapsed",
        )
        selected_key = NAV_KEYS[NAV_OPTIONS.index(choice)]
        if selected_key != st.session_state.page:
            st.session_state.page = selected_key
            st.rerun()

        # ── Footer ──────────────────────────────────────────
        st.markdown(
            """
            <div style="
                position:fixed; bottom:14px; left:0; width:240px;
                text-align:center; font-size:0.78rem; color:#9CA3AF;
            ">
                Made with <span style="color:#EF4444;">❤️</span> by
                <span style="color:#3B6BF5; font-weight:600;">Scholar Minds</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE — HOME
# ═══════════════════════════════════════════════════════════════════════════
def page_home():
    # ── Welcome hero ─────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding:28px 0 8px 0;">
            <div style="
                display:inline-flex; align-items:center; justify-content:center;
                width:64px; height:64px; border-radius:16px;
                background:linear-gradient(135deg,#3B6BF5,#6B8FF7);
                box-shadow:0 6px 20px rgba(59,107,245,0.25);
                margin-bottom:14px;
            ">
                <span style="font-size:32px;">🎓</span>
            </div>
            <h1 style="font-size:2.1rem; font-weight:800; color:#1A1A2E; margin:0 0 6px 0;">
                Welcome to Scholar Minds
            </h1>
            <p style="font-size:1.05rem; color:#6B7280; margin:0;">
                Ask questions, get insights, and explore research papers with AI.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Search card ──────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        """<p style="font-weight:600; color:#1A1A2E; font-size:0.95rem; margin-bottom:2px;">
        Ask anything about your research…</p>""",
        unsafe_allow_html=True,
    )

    col_input, col_btn = st.columns([6, 1])
    with col_input:
        query = st.text_input(
            "search_home",
            placeholder="e.g., What is RAG in NLP?",
            label_visibility="collapsed",
            key="home_search_input",
        )
    with col_btn:
        ask_clicked = st.button("✈️  Ask", type="primary", use_container_width=True, key="home_ask_btn")

    if ask_clicked and query:
        st.session_state.pending_question = query
        st.session_state.page = "ask"
        st.rerun()

    # ── Example questions ────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.84rem; color:#6B7280; margin:10px 0 6px 0;'>Try example questions:</p>",
        unsafe_allow_html=True,
    )
    examples = [
        "What are the key findings of this research paper?",
        "Summarize this paper",
        "Which methodology is used in this",
    ]
    eq_cols = st.columns(3)
    for idx, (col, q) in enumerate(zip(eq_cols, examples)):
        with col:
            # Wrap in a div so we can target the CSS class .pill-btn
            st.markdown('<div class="pill-btn">', unsafe_allow_html=True)
            if st.button(q, key=f"example_q_{idx}", use_container_width=True):
                st.session_state.pending_question = q
                st.session_state.page = "ask"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Recent Papers ────────────────────────────────────────
    st.markdown(
        """
        <h2 style="font-size:1.3rem; font-weight:700; color:#1A1A2E; margin:36px 0 14px 0;">
            Your Recent Papers
        </h2>
        """,
        unsafe_allow_html=True,
    )

    papers = st.session_state.papers
    if papers:
        pcols = st.columns(min(len(papers), 3))
        for i, paper in enumerate(papers[:3]):
            with pcols[i]:
                st.markdown(
                    f"""
                    <div class="scholar-card" style="padding:22px; min-height:150px;">
                        <div style="display:flex; align-items:flex-start; gap:12px; margin-bottom:14px;">
                            <div style="
                                background:#FEE2E2; border-radius:10px;
                                width:38px; height:38px;
                                display:flex; align-items:center; justify-content:center;
                                flex-shrink:0;
                            ">
                                <span style="color:#EF4444; font-size:17px;">📄</span>
                            </div>
                            <div>
                                <div style="font-weight:600; color:#1A1A2E; font-size:0.95rem; line-height:1.35;">
                                    {paper["name"].replace(".pdf", "")}
                                </div>
                                <div style="font-size:0.78rem; color:#9CA3AF; margin-top:3px;">
                                    {paper.get("num_pages", "?")} pages &middot; {paper.get("num_chunks", "?")} chunks
                                </div>
                                <div style="font-size:0.73rem; color:#CBD5E1; margin-top:2px;">
                                    Uploaded {paper.get("upload_time", "")}
                                </div>
                            </div>
                        </div>
                        <div style="display:flex; align-items:center; justify-content:space-between;">
                            <span style="
                                display:inline-block; padding:5px 14px;
                                border:1.5px solid #10B981; border-radius:8px;
                                color:#10B981; font-weight:600; font-size:0.8rem;
                                cursor:pointer; transition:background 0.2s;
                            ">View Summary</span>
                            <span style="color:#CBD5E1; font-size:18px; cursor:pointer;">🔖</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            """
            <div class="scholar-card" style="text-align:center; padding:44px 24px; color:#9CA3AF;">
                <div style="font-size:40px; margin-bottom:10px;">📚</div>
                <p style="font-weight:600; color:#6B7280; margin:0 0 4px 0;">No papers uploaded yet</p>
                <p style="font-size:0.85rem; margin:0;">
                    Head to <b>Upload Papers</b> to add your first research paper.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Feature highlight cards ──────────────────────────────
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    features = [
        ("🔍", "#EEF2FF", "#3B6BF5", "Smart Search",
         "Search across papers using natural language."),
        ("📋", "#ECFDF5", "#10B981", "Summarize",
         "Get concise summaries of complex papers."),
        ("💬", "#F5F3FF", "#8B5CF6", "Chat with Papers",
         "Ask questions and get answers with citations."),
    ]
    fcols = st.columns(3)
    for col, (icon, bg, color, title, desc) in zip(fcols, features):
        with col:
            st.markdown(
                f"""
                <div class="scholar-card" style="padding:22px; display:flex; align-items:flex-start; gap:14px;">
                    <div style="
                        background:{bg}; border-radius:12px;
                        width:46px; height:46px;
                        display:flex; align-items:center; justify-content:center;
                        flex-shrink:0;
                    ">
                        <span style="font-size:22px;">{icon}</span>
                    </div>
                    <div>
                        <div style="font-weight:600; color:#1A1A2E; font-size:0.95rem;">{title}</div>
                        <div style="font-size:0.83rem; color:#6B7280; margin-top:4px; line-height:1.45;">{desc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE — ASK A QUESTION
# ═══════════════════════════════════════════════════════════════════════════
def page_ask():
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
            <div style="
                background:linear-gradient(135deg,#3B6BF5,#6B8FF7);
                border-radius:10px; width:38px; height:38px;
                display:flex; align-items:center; justify-content:center;
            ">
                <span style="font-size:20px;">💬</span>
            </div>
            <h1 style="font-size:1.5rem; font-weight:700; color:#1A1A2E; margin:0;">
                Ask a Question
            </h1>
        </div>
        <p style="color:#6B7280; font-size:0.88rem; margin:0 0 20px 0;">
            Ask anything about your uploaded research papers. Answers include citations.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Paper Selection ──────────────────────────────────────
    paper_options = ["All Papers"]
    if st.session_state.papers:
        paper_options.extend([p["name"] for p in st.session_state.papers])
    
    if "selected_paper" not in st.session_state:
        st.session_state.selected_paper = "All Papers"
    
    if st.session_state.selected_paper not in paper_options:
        st.session_state.selected_paper = "All Papers"
        
    selected_idx = paper_options.index(st.session_state.selected_paper)
    st.session_state.selected_paper = st.selectbox(
        "Select target paper for Q&A:",
        options=paper_options,
        index=selected_idx,
        key="target_paper_select",
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Chat history ─────────────────────────────────────────
    chat_container = st.container(height=450)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div class="scholar-card" style="text-align:center; padding:56px 28px; margin-bottom:20px;">
                    <div style="font-size:50px; margin-bottom:14px;">🤖</div>
                    <h3 style="color:#1A1A2E; font-weight:700; margin:0 0 6px 0;">Start a Conversation</h3>
                    <p style="color:#6B7280; font-size:0.9rem; margin:0;">
                        Type your question below to get AI-powered answers from your research papers.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                        <div style="display:flex; justify-content:flex-end; margin:10px 0;">
                            <div style="
                                background:linear-gradient(135deg,#3B6BF5,#5B85F7);
                                color:#fff; padding:12px 18px;
                                border-radius:16px 16px 4px 16px;
                                max-width:70%; font-size:0.93rem;
                                line-height:1.5; box-shadow:0 2px 8px rgba(59,107,245,0.18);
                            ">{msg["content"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="display:flex; justify-content:flex-start; margin:10px 0;">
                            <div style="
                                background:#FFFFFF; color:#1A1A2E;
                                padding:14px 18px;
                                border-radius:16px 16px 16px 4px;
                                max-width:80%; font-size:0.93rem;
                                line-height:1.65;
                                border:1px solid #E5E7EB;
                                box-shadow:0 1px 3px rgba(0,0,0,0.04);
                            ">{msg["content"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ── Input area ───────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    default_q = st.session_state.pop("pending_question", "")
    col_in, col_send = st.columns([6, 1])
    with col_in:
        question = st.text_input(
            "ask_input",
            value=default_q,
            placeholder="Type your question here…",
            label_visibility="collapsed",
            key="ask_text_input",
        )
    with col_send:
        send_clicked = st.button("✈️  Ask", type="primary", use_container_width=True, key="ask_send_btn")

    # ── Action bar ───────────────────────────────────────────
    ab1, ab2, ab_spacer = st.columns([1, 1, 4])
    with ab1:
        if st.button("🗑️  Clear Chat", use_container_width=True, key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.rerun()
    with ab2:
        if st.button("💾  Save Chat", use_container_width=True, key="save_chat_btn"):
            if st.session_state.chat_history:
                st.session_state.saved_chats.append({
                    "name": f"Chat {len(st.session_state.saved_chats) + 1}",
                    "messages": list(st.session_state.chat_history),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
                _persist_chats()
                st.success("✅ Chat saved successfully!")
            else:
                st.warning("Nothing to save — start a conversation first.")

    # ── Send logic ───────────────────────────────────────────
    if send_clicked and question.strip():
        st.session_state.chat_history.append({
            "role": "user",
            "content": question.strip(),
            "timestamp": datetime.now().isoformat(),
        })
        with st.spinner("🔍  Searching papers and generating answer…"):
            try:
                source_filter = None if st.session_state.selected_paper == "All Papers" else st.session_state.selected_paper
                answer = ask_rag(question.strip(), source_filter=source_filter)
            except Exception as exc:
                answer = f"⚠️ Error: {exc}. Make sure Ollama is running with required models."
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().isoformat(),
        })
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE — UPLOAD PAPERS
# ═══════════════════════════════════════════════════════════════════════════
def page_upload():
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
            <div style="
                background:linear-gradient(135deg,#10B981,#34D399);
                border-radius:10px; width:38px; height:38px;
                display:flex; align-items:center; justify-content:center;
            ">
                <span style="font-size:20px;">📄</span>
            </div>
            <h1 style="font-size:1.5rem; font-weight:700; color:#1A1A2E; margin:0;">
                Upload Papers
            </h1>
        </div>
        <p style="color:#6B7280; font-size:0.88rem; margin:0 0 22px 0;">
            Upload research paper PDFs to build your knowledge base.
            Papers are parsed, chunked, and indexed for Q&amp;A.
        </p>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="file_uploader_widget",
    )

    if uploaded_files:
        existing_names = {p["name"] for p in st.session_state.papers}
        for uf in uploaded_files:
            if uf.name in existing_names:
                st.info(f"📄 **{uf.name}** is already in your library.")
                continue
            with st.spinner(f"⏳ Processing **{uf.name}**…"):
                try:
                    meta = process_paper(uf)
                    st.session_state.papers.append(meta)
                    st.session_state.selected_paper = uf.name
                    st.success(
                        f"✅ **{uf.name}** — {meta['num_pages']} pages, "
                        f"{meta['num_chunks']} chunks indexed!"
                    )
                except Exception as exc:
                    st.error(f"❌ Failed to process **{uf.name}**: {exc}")

    # ── Uploaded papers list ─────────────────────────────────
    if st.session_state.papers:
        st.markdown(
            """
            <h2 style="font-size:1.15rem; font-weight:600; color:#1A1A2E; margin:32px 0 14px 0;">
                📚 Indexed Papers
            </h2>
            """,
            unsafe_allow_html=True,
        )

        for i, paper in enumerate(st.session_state.papers):
            c_info, c_del = st.columns([6, 1])
            with c_info:
                st.markdown(
                    f"""
                    <div class="scholar-card" style="padding:16px 20px; margin-bottom:8px;
                         display:flex; align-items:center; gap:14px;">
                        <div style="
                            background:#FEE2E2; border-radius:9px;
                            width:36px; height:36px;
                            display:flex; align-items:center; justify-content:center;
                            flex-shrink:0;
                        ">
                            <span style="color:#EF4444;">📄</span>
                        </div>
                        <div>
                            <div style="font-weight:600; color:#1A1A2E; font-size:0.93rem;">
                                {paper["name"]}
                            </div>
                            <div style="font-size:0.78rem; color:#9CA3AF;">
                                {paper["num_pages"]} pages &middot;
                                {paper["num_chunks"]} chunks &middot;
                                Uploaded {paper["upload_time"]}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_del:
                st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_paper_{i}"):
                    paper_name = st.session_state.papers[i]["name"]
                    retriever = get_retriever()
                    if retriever:
                        try:
                            retriever._collection.delete(where={"source": paper_name})
                        except Exception:
                            pass
                    st.session_state.papers.pop(i)
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE — MY LIBRARY
# ═══════════════════════════════════════════════════════════════════════════
def page_library():
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
            <div style="
                background:linear-gradient(135deg,#8B5CF6,#A78BFA);
                border-radius:10px; width:38px; height:38px;
                display:flex; align-items:center; justify-content:center;
            ">
                <span style="font-size:20px;">📚</span>
            </div>
            <h1 style="font-size:1.5rem; font-weight:700; color:#1A1A2E; margin:0;">
                My Library
            </h1>
        </div>
        <p style="color:#6B7280; font-size:0.88rem; margin:0 0 22px 0;">
            All your uploaded research papers in one place.
        </p>
        """,
        unsafe_allow_html=True,
    )

    papers = st.session_state.papers
    if not papers:
        st.markdown(
            """
            <div class="scholar-card" style="text-align:center; padding:56px; max-width:480px; margin:0 auto;">
                <div style="font-size:50px; margin-bottom:12px;">📚</div>
                <h3 style="color:#1A1A2E; font-weight:700; margin:0 0 6px 0;">Your library is empty</h3>
                <p style="color:#6B7280; font-size:0.9rem; margin:0;">
                    Upload research papers to start building your knowledge base.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("📄  Upload Papers", type="primary", key="lib_upload_btn"):
            st.session_state.page = "upload"
            st.rerun()
        return

    # ── Stats row ────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("📄 Total Papers", len(papers))
    with m2:
        st.metric("🧩 Total Chunks", sum(p.get("num_chunks", 0) for p in papers))
    with m3:
        st.metric("📃 Total Pages", sum(p.get("num_pages", 0) for p in papers))

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Paper grid ───────────────────────────────────────────
    cols = st.columns(3)
    for i, paper in enumerate(papers):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="scholar-card" style="padding:22px; margin-bottom:14px;">
                    <div style="display:flex; align-items:flex-start; gap:12px;">
                        <div style="
                            background:#FEE2E2; border-radius:10px;
                            width:40px; height:40px;
                            display:flex; align-items:center; justify-content:center;
                            flex-shrink:0;
                        ">
                            <span style="color:#EF4444; font-size:18px;">📄</span>
                        </div>
                        <div>
                            <div style="font-weight:600; color:#1A1A2E; font-size:0.95rem; line-height:1.3;">
                                {paper["name"].replace(".pdf", "")}
                            </div>
                            <div style="font-size:0.78rem; color:#9CA3AF; margin-top:4px;">
                                {paper["num_pages"]} pages &middot; {paper["num_chunks"]} chunks
                            </div>
                            <div style="font-size:0.73rem; color:#CBD5E1; margin-top:2px;">
                                Uploaded {paper["upload_time"]}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE — SAVED CHATS
# ═══════════════════════════════════════════════════════════════════════════
def page_saved():
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
            <div style="
                background:linear-gradient(135deg,#F59E0B,#FBBF24);
                border-radius:10px; width:38px; height:38px;
                display:flex; align-items:center; justify-content:center;
            ">
                <span style="font-size:20px;">⭐</span>
            </div>
            <h1 style="font-size:1.5rem; font-weight:700; color:#1A1A2E; margin:0;">
                Saved Chats
            </h1>
        </div>
        <p style="color:#6B7280; font-size:0.88rem; margin:0 0 22px 0;">
            Review your saved Q&amp;A sessions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    chats = st.session_state.saved_chats
    if not chats:
        st.markdown(
            """
            <div class="scholar-card" style="text-align:center; padding:56px; max-width:480px; margin:0 auto;">
                <div style="font-size:50px; margin-bottom:12px;">💬</div>
                <h3 style="color:#1A1A2E; font-weight:700; margin:0 0 6px 0;">No saved chats yet</h3>
                <p style="color:#6B7280; font-size:0.9rem; margin:0;">
                    Use the <b>Save Chat</b> button on the Ask a Question page to keep conversations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for i, chat in enumerate(reversed(chats)):
        real_idx = len(chats) - 1 - i
        with st.expander(f"💬  {chat['name']}  —  {chat.get('timestamp', '')}", expanded=False):
            for msg in chat.get("messages", []):
                is_user = msg["role"] == "user"
                icon = "🧑" if is_user else "🤖"
                label = "You" if is_user else "Scholar Minds"
                bg = "#EEF2FF" if is_user else "#F0FDF4"
                st.markdown(
                    f"""
                    <div style="background:{bg}; border-radius:10px; padding:12px 16px; margin:6px 0;">
                        <div style="font-weight:600; font-size:0.78rem; color:#6B7280; margin-bottom:4px;">
                            {icon} {label}
                        </div>
                        <div style="color:#1A1A2E; font-size:0.9rem; line-height:1.6;">
                            {msg["content"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if st.button("🗑️  Delete this chat", key=f"del_chat_{real_idx}"):
                st.session_state.saved_chats.pop(real_idx)
                _persist_chats()
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE — ABOUT
# ═══════════════════════════════════════════════════════════════════════════
def page_about():
    def clean_html(html_str):
        return "\n".join(line.lstrip() for line in html_str.splitlines())

    # ── Render Hero ──────────────────────────────────────────
    hero_html = clean_html("""
        <div style="
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 25%, #3730a3 50%, #3B6BF5 75%, #6366f1 100%);
            background-size: 200% 200%;
            animation: gradient-shift 8s ease infinite;
            border-radius: 20px;
            padding: 56px 40px 48px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
            margin-bottom: 28px;
        ">
            <!-- Decorative circles -->
            <div style="
                position:absolute; top:-40px; right:-40px;
                width:160px; height:160px; border-radius:50%;
                background:rgba(255,255,255,0.04);
            "></div>
            <div style="
                position:absolute; bottom:-30px; left:-30px;
                width:120px; height:120px; border-radius:50%;
                background:rgba(255,255,255,0.03);
            "></div>
            <div style="
                position:absolute; top:30%; left:10%;
                width:8px; height:8px; border-radius:50%;
                background:rgba(255,255,255,0.2);
                animation: float 3s ease-in-out infinite;
            "></div>
            <div style="
                position:absolute; top:20%; right:15%;
                width:6px; height:6px; border-radius:50%;
                background:rgba(255,255,255,0.15);
                animation: float 4s ease-in-out infinite 1s;
            "></div>
            <div style="
                position:absolute; bottom:25%; right:25%;
                width:5px; height:5px; border-radius:50%;
                background:rgba(255,255,255,0.12);
                animation: float 3.5s ease-in-out infinite 0.5s;
            "></div>

            <!-- Logo -->
            <div class="about-hero-logo" style="
                display:inline-flex; align-items:center; justify-content:center;
                width:88px; height:88px; border-radius:22px;
                background:rgba(255,255,255,0.15);
                backdrop-filter:blur(10px);
                -webkit-backdrop-filter:blur(10px);
                border:1px solid rgba(255,255,255,0.2);
                margin-bottom:20px;
            ">
                <span style="font-size:44px;">🎓</span>
            </div>
            <h1 style="
                font-size:2.4rem; font-weight:800; color:#FFFFFF;
                margin:0 0 8px 0; letter-spacing:-0.02em;
            ">
                Scholar Minds
            </h1>
            <p style="
                font-size:1.1rem; color:rgba(255,255,255,0.8);
                margin:0 0 20px 0; font-weight:400;
            ">
                Offline Research Paper Q&amp;A powered by RAG on Edge Devices
            </p>
            <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
                <span style="
                    background:rgba(255,255,255,0.15);
                    backdrop-filter:blur(8px);
                    -webkit-backdrop-filter:blur(8px);
                    border:1px solid rgba(255,255,255,0.2);
                    padding:6px 16px; border-radius:20px;
                    font-size:0.82rem; font-weight:500; color:#fff;
                ">🔒 Fully Offline</span>
                <span style="
                    background:rgba(255,255,255,0.15);
                    backdrop-filter:blur(8px);
                    -webkit-backdrop-filter:blur(8px);
                    border:1px solid rgba(255,255,255,0.2);
                    padding:6px 16px; border-radius:20px;
                    font-size:0.82rem; font-weight:500; color:#fff;
                ">⚡ Edge AI</span>
                <span style="
                    background:rgba(255,255,255,0.15);
                    backdrop-filter:blur(8px);
                    -webkit-backdrop-filter:blur(8px);
                    border:1px solid rgba(255,255,255,0.2);
                    padding:6px 16px; border-radius:20px;
                    font-size:0.82rem; font-weight:500; color:#fff;
                ">📚 RAG Pipeline</span>
            </div>
        </div>
    """)
    st.markdown(hero_html, unsafe_allow_html=True)

    # ── About card (glassmorphism) ────────────────────────────
    about_html = clean_html("""
        <div style="
            max-width:800px; margin:0 auto 24px auto;
            background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,250,252,0.9));
            backdrop-filter:blur(12px);
            -webkit-backdrop-filter:blur(12px);
            border-radius:18px;
            padding:32px 36px;
            border:1px solid rgba(59,107,245,0.08);
            box-shadow: 0 4px 24px rgba(59,107,245,0.06), 0 1px 3px rgba(0,0,0,0.04);
            animation: slide-up 0.5s ease-out;
        ">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
                <div style="
                    width:40px; height:40px; border-radius:12px;
                    background:linear-gradient(135deg,#3B6BF5,#6366f1);
                    display:flex; align-items:center; justify-content:center;
                ">
                    <span style="font-size:20px;">📖</span>
                </div>
                <h3 style="color:#1A1A2E; margin:0; font-size:1.15rem; font-weight:700;">
                    About the Project
                </h3>
            </div>
            <p style="color:#4B5563; line-height:1.8; font-size:0.93rem; margin:0 0 12px 0;">
                Scholar Minds is an AI-powered Research Paper Question-Answering system built using
                <span style="background:linear-gradient(135deg,#3B6BF5,#6366f1); -webkit-background-clip:text;
                -webkit-text-fill-color:transparent; font-weight:700;">Retrieval-Augmented Generation (RAG)</span>.
                Upload research papers, ask questions in natural language, and get context-aware answers
                with citations — <b>completely offline</b>.
            </p>
            <p style="color:#4B5563; line-height:1.8; font-size:0.93rem; margin:0;">
                Built for edge AI deployment using
                <span style="background:#F0FDF4; color:#059669; padding:2px 10px; border-radius:6px;
                font-weight:600; font-size:0.85rem;">NVIDIA Jetson</span> +
                <span style="background:#EEF2FF; color:#3B6BF5; padding:2px 10px; border-radius:6px;
                font-weight:600; font-size:0.85rem;">Ollama</span>
            </p>
        </div>
    """)
    st.markdown(about_html, unsafe_allow_html=True)

    # ── RAG Pipeline (horizontal flow) ───────────────────────
    pipeline = [
        ("📄", "Upload", "Drop your PDF", "#EEF2FF", "#3B6BF5"),
        ("📝", "Extract", "Parse text & pages", "#F0FDF4", "#059669"),
        ("✂️", "Chunk", "Split into passages", "#FFF7ED", "#EA580C"),
        ("🧮", "Embed", "Generate vectors", "#FDF2F8", "#DB2777"),
        ("💾", "Store", "Index in ChromaDB", "#F5F3FF", "#7C3AED"),
        ("🔍", "Retrieve", "Semantic search", "#ECFDF5", "#10B981"),
        ("🤖", "Generate", "LLM via Ollama", "#EFF6FF", "#2563EB"),
        ("✅", "Answer", "Cited response", "#F0FDF4", "#16A34A"),
    ]

    pipeline_cards = []
    for icon, title, desc, bg, color in pipeline:
        card = f"""
        <div class="pipeline-card" style="
            background: {bg};
            border-color: {color}20;
            --hover-shadow: 0 6px 16px {color}25;
        ">
            <div style="font-size: 28px; margin-bottom: 8px;">{icon}</div>
            <div style="font-weight: 700; color: {color}; font-size: 0.85rem; margin-bottom: 3px;">
                {title}
            </div>
            <div style="font-size: 0.72rem; color: #6B7280; line-height: 1.4;">
                {desc}
            </div>
            <div style="
                position: absolute; bottom: -4px; left: 50%;
                transform: translateX(-50%);
                width: 24px; height: 3px; border-radius: 2px;
                background: linear-gradient(90deg, {color}, {color}60);
            "></div>
        </div>
        """
        pipeline_cards.append(card)

    pipeline_grid_content = "\n".join(pipeline_cards)
    pipeline_html = clean_html(f"""
        <div style="
            max-width:800px; margin:0 auto 24px auto;
            background:#FFFFFF;
            border-radius:18px;
            padding:32px 28px;
            border:1px solid rgba(59,107,245,0.08);
            box-shadow: 0 4px 24px rgba(59,107,245,0.06), 0 1px 3px rgba(0,0,0,0.04);
            animation: slide-up 0.6s ease-out;
        ">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
                <div style="
                    width:40px; height:40px; border-radius:12px;
                    background:linear-gradient(135deg,#8B5CF6,#a78bfa);
                    display:flex; align-items:center; justify-content:center;
                ">
                    <span style="font-size:20px;">🧠</span>
                </div>
                <h3 style="color:#1A1A2E; margin:0; font-size:1.15rem; font-weight:700;">
                    How It Works
                </h3>
            </div>

            <div class="pipeline-grid">
                {pipeline_grid_content}
            </div>
        </div>
    """)
    st.markdown(pipeline_html, unsafe_allow_html=True)

    # ── Tech Stack (pill badges) ─────────────────────────────
    tech = [
        ("Python", "#3776AB", "#EBF5FF"),
        ("Streamlit", "#FF4B4B", "#FFF1F1"),
        ("Ollama", "#1A1A2E", "#F3F4F6"),
        ("ChromaDB", "#FF6F00", "#FFF8F0"),
        ("PyMuPDF", "#E74C3C", "#FFF1F0"),
        ("LLaMA 3.2", "#6366f1", "#EEF2FF"),
        ("Nomic Embed", "#059669", "#ECFDF5"),
        ("NVIDIA Jetson", "#76B900", "#F7FEE7"),
    ]

    tech_badges = []
    for name, color, bg in tech:
        badge = f"""
        <div class="tech-badge" style="
            background: {bg};
            border-color: {color}30;
            --hover-shadow: 0 4px 12px {color}20;
        ">
            <div style="
                width: 8px; height: 8px; border-radius: 50%;
                background: {color};
            "></div>
            <span style="font-weight: 600; color: {color}; font-size: 0.88rem;">
                {name}
            </span>
        </div>
        """
        tech_badges.append(badge)

    tech_badges_content = "\n".join(tech_badges)
    tech_html = clean_html(f"""
        <div style="
            max-width:800px; margin:0 auto 24px auto;
            background:#FFFFFF;
            border-radius:18px;
            padding:32px 36px;
            border:1px solid rgba(59,107,245,0.08);
            box-shadow: 0 4px 24px rgba(59,107,245,0.06), 0 1px 3px rgba(0,0,0,0.04);
            animation: slide-up 0.7s ease-out;
        ">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:20px;">
                <div style="
                    width:40px; height:40px; border-radius:12px;
                    background:linear-gradient(135deg,#10B981,#34d399);
                    display:flex; align-items:center; justify-content:center;
                ">
                    <span style="font-size:20px;">🛠️</span>
                </div>
                <h3 style="color:#1A1A2E; margin:0; font-size:1.15rem; font-weight:700;">
                    Tech Stack
                </h3>
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:10px;">
                {tech_badges_content}
            </div>
        </div>
    """)
    st.markdown(tech_html, unsafe_allow_html=True)

    # ── Team — Neural Ninjas ─────────────────────────────────
    team = [
        ("Shreyash Omar", "Developer", "#3B6BF5", "#6366f1"),
        ("Aniket Sahu", "Developer", "#10B981", "#059669"),
        ("Priyaansh Pandey", "Developer", "#F59E0B", "#EA580C"),
        ("Devansh Shukla", "Developer", "#8B5CF6", "#7C3AED"),
    ]

    team_cards = []
    for name, role, c1, c2 in team:
        initials = name[0]
        card = f"""
        <div class="team-card">
            <div style="
                width: 56px; height: 56px; border-radius: 50%;
                background: linear-gradient(135deg, {c1}, {c2});
                margin: 0 auto 10px auto;
                display: flex; align-items: center; justify-content: center;
                box-shadow: 0 4px 16px {c1}30;
                border: 3px solid #fff;
            ">
                <span style="color: #fff; font-weight: 700; font-size: 1.2rem;">
                    {initials}
                </span>
            </div>
            <div style="font-weight: 600; color: #1A1A2E; font-size: 0.88rem;
                line-height: 1.3; margin-bottom: 4px;">
                {name}
            </div>
            <div style="
                display: inline-block;
                background: linear-gradient(135deg, {c1}15, {c2}15);
                color: {c1}; font-weight: 600;
                font-size: 0.7rem; padding: 3px 10px;
                border-radius: 8px;
            ">{role}</div>
        </div>
        """
        team_cards.append(card)

    team_cards_content = "\n".join(team_cards)
    team_html = clean_html(f"""
        <div style="
            max-width:800px; margin:0 auto 24px auto;
            background:#FFFFFF;
            border-radius:18px;
            padding:32px 36px;
            border:1px solid rgba(59,107,245,0.08);
            box-shadow: 0 4px 24px rgba(59,107,245,0.06), 0 1px 3px rgba(0,0,0,0.04);
            animation: slide-up 0.8s ease-out;
        ">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
                <div style="
                    width:40px; height:40px; border-radius:12px;
                    background:linear-gradient(135deg,#F59E0B,#FBBF24);
                    display:flex; align-items:center; justify-content:center;
                ">
                    <span style="font-size:20px;">👨‍💻</span>
                </div>
                <div>
                    <h3 style="color:#1A1A2E; margin:0; font-size:1.15rem; font-weight:700;">
                        Team — Neural Ninjas
                    </h3>
                    <p style="color:#9CA3AF; margin:0; font-size:0.78rem;">
                        The minds behind Scholar Minds
                    </p>
                </div>
            </div>
            <div class="team-grid">
                {team_cards_content}
            </div>
        </div>
    """)
    st.markdown(team_html, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────
    footer_html = clean_html("""
        <div style="max-width:800px; margin:0 auto; padding:20px 0 8px 0; text-align:center;">
            <div style="
                width:80px; height:3px; border-radius:2px;
                background:linear-gradient(90deg,#3B6BF5,#8B5CF6,#F59E0B);
                margin:0 auto 16px auto;
            "></div>
            <p style="font-size:0.82rem; color:#9CA3AF; margin:0;">
                Made with <span style="color:#EF4444;">❤️</span> by
                <span style="
                    background:linear-gradient(135deg,#3B6BF5,#6366f1);
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;
                    font-weight:700;
                ">Neural Ninjas</span>
                &nbsp;·&nbsp; Licensed under the <b>MIT License</b>
            </p>
        </div>
    """)
    st.markdown(footer_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    init_state()

    # Inject CSS
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

    # Render sidebar
    render_sidebar()

    # Route to active page
    router = {
        "home": page_home,
        "ask": page_ask,
        "upload": page_upload,
        "library": page_library,
        "saved": page_saved,
        "about": page_about,
    }
    router.get(st.session_state.page, page_home)()


if __name__ == "__main__":
    main()
