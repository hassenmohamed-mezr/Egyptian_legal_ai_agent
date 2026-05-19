"""
Lex AI – Legal Assistant
Phase 1: RAG System with Ollama + Hybrid Search
Connects to local FastAPI backend for retrieval and generation.
"""

import streamlit as st
import requests
import time
from datetime import datetime

# ─── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Lex AI – المساعد القانوني",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── API Configuration ────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"

# ─── CSS: Premium Dark Legal Theme (كما هو موجود، لم يتغير) ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #0A0C10;
    --bg-secondary:  #111318;
    --bg-card:       #161A23;
    --bg-card-hover: #1C2130;
    --border:        #252A38;
    --border-accent: #2E3650;
    --gold:          #C9A84C;
    --gold-light:    #E8C97A;
    --gold-dim:      #7A6330;
    --text-primary:  #E8EAF0;
    --text-secondary:#8A91A8;
    --text-muted:    #4A5068;
    --green:         #2ECC71;
    --green-dim:     #1A7A44;
    --red:           #E74C3C;
    --blue:          #4A90D9;
    --score-high:    #2ECC71;
    --score-mid:     #F39C12;
    --score-low:     #E74C3C;
}

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main Container ── */
.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.2rem;
}

/* ── Logo / Brand ── */
.brand-container {
    text-align: center;
    padding: 1.5rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.brand-logo {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 3px;
    line-height: 1;
    text-shadow: 0 0 30px rgba(201,168,76,0.3);
}
.brand-subtitle {
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── Sidebar Sections ── */
.sidebar-section {
    margin-bottom: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.sidebar-section-title {
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--gold-dim);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Status Badge ── */
.status-badge {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: rgba(46,204,113,0.08);
    border: 1px solid rgba(46,204,113,0.2);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.8rem;
    color: var(--green);
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse-green 2s infinite;
    flex-shrink: 0;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(46,204,113,0.4); }
    50%       { box-shadow: 0 0 0 6px rgba(46,204,113,0); }
}
.status-badge-error {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: rgba(231,76,60,0.08);
    border: 1px solid rgba(231,76,60,0.2);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.8rem;
    color: var(--red);
}
.mode-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(201,168,76,0.08);
    border: 1px solid var(--gold-dim);
    border-radius: 20px;
    padding: 0.35rem 0.8rem;
    font-size: 0.75rem;
    color: var(--gold);
    font-family: 'JetBrains Mono', monospace;
}

/* ── Page Header ── */
.page-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--gold);
    margin: 0 0 0.4rem;
    direction: rtl;
}
.page-desc {
    color: var(--text-secondary);
    font-size: 0.9rem;
    direction: rtl;
}

/* ── Search Box ── */
.search-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border-accent);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.3s;
}
.search-wrapper:focus-within {
    border-color: var(--gold-dim);
    box-shadow: 0 0 0 3px rgba(201,168,76,0.06);
}
.stTextArea textarea {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    font-size: 1rem !important;
    direction: rtl;
    text-align: right;
    padding: 1rem !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: var(--gold-dim) !important;
    box-shadow: none !important;
}
.stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
    font-size: 0.9rem;
}

/* ── Search Button ── */
.stButton > button {
    background: linear-gradient(135deg, #C9A84C, #A07830) !important;
    color: #0A0C10 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(201,168,76,0.25) !important;
    background: linear-gradient(135deg, #E8C97A, #C9A84C) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result Card ── */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    transition: all 0.25s;
    direction: rtl;
    text-align: right;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--gold), transparent);
    border-radius: 0 14px 14px 0;
}
.result-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-accent);
    transform: translateX(-2px);
    box-shadow: 4px 0 20px rgba(0,0,0,0.3);
}
.answer-text {
    font-size: 1rem;
    line-height: 1.8;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 1.2rem;
    border-right: 3px solid var(--gold);
    margin-bottom: 1.2rem;
}
.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.8rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.card-article-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--gold-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
}
.card-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0.3rem 0 0.8rem;
    line-height: 1.5;
}
.card-snippet {
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.8;
    border-top: 1px solid var(--border);
    padding-top: 0.8rem;
    margin-top: 0.5rem;
}
.card-full-text {
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 2;
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    border: 1px solid var(--border);
    white-space: pre-wrap;
}
.score-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    border-radius: 20px;
    padding: 0.25rem 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
}
.score-high { background: rgba(46,204,113,0.12); color: #2ECC71; border: 1px solid rgba(46,204,113,0.25); }
.score-mid  { background: rgba(243,156,18,0.12);  color: #F39C12; border: 1px solid rgba(243,156,18,0.25); }
.score-low  { background: rgba(231,76,60,0.12);   color: #E74C3C; border: 1px solid rgba(231,76,60,0.25); }
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: rgba(201,168,76,0.1);
    border: 1px solid var(--gold-dim);
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--gold);
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
}
.results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
    direction: rtl;
}
.results-count {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}
.results-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-secondary);
    letter-spacing: 1px;
}
.history-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;
    direction: rtl;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.history-item:hover {
    background: var(--bg-card-hover);
    border-color: var(--gold-dim);
    color: var(--gold);
}
.history-time {
    font-size: 0.65rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    display: block;
    margin-bottom: 0.2rem;
}
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-muted);
    direction: rtl;
}
.empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.4;
}
.empty-text {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}
.empty-subtext {
    font-size: 0.8rem;
    color: var(--text-muted);
}
.error-box {
    background: rgba(231,76,60,0.08);
    border: 1px solid rgba(231,76,60,0.3);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    direction: rtl;
    text-align: right;
    margin-bottom: 1.5rem;
}
.error-title { color: var(--red); font-weight: 600; margin-bottom: 0.4rem; }
.error-detail { color: var(--text-secondary); font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; }
.stats-row {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1.5rem;
    direction: rtl;
    flex-wrap: wrap;
}
.stat-chip {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.stat-chip span { color: var(--gold); font-family: 'JetBrains Mono', monospace; font-weight: 600; }
hr { border-color: var(--border) !important; }
.stSpinner > div { border-top-color: var(--gold) !important; }
.streamlit-expanderHeader {
    background: var(--bg-secondary) !important;
    border-radius: 8px !important;
    color: var(--gold) !important;
    font-size: 0.82rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border-radius: 0 0 8px 8px !important;
}
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
.stSelectbox label { color: var(--text-secondary) !important; font-size: 0.8rem !important; }
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
.stNumberInput label { color: var(--text-secondary) !important; font-size: 0.8rem !important; }
.stCheckbox label { color: var(--text-secondary) !important; font-size: 0.82rem !important; }
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border-accent) !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 1rem !important;
}
.stDownloadButton > button:hover {
    border-color: var(--gold-dim) !important;
    color: var(--gold) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Helper: Check API health ────────────────────────────────────────────────
@st.cache_resource(ttl=5)
def check_api_status():
    """Return (is_healthy, error_message)"""
    try:
        # Try a lightweight health check (simple GET)
        resp = requests.get(f"{API_BASE_URL}/", timeout=2)
        # Actually root might return 404, but that's fine as long as server responds
        if resp.status_code < 500:
            return True, None
        return False, f"خادم API أجاب برمز {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"لا يمكن الاتصال بالخادم على {API_BASE_URL}"
    except Exception as e:
        return False, str(e)

# ─── Session State Init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []          # list of {"query": str, "time": str, "count": int}
if "last_result" not in st.session_state:
    st.session_state.last_result = None    # last API response dict
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "current_query" not in st.session_state:
    st.session_state.current_query = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-container">
        <div class="brand-logo">⚖ LEX AI</div>
        <div class="brand-subtitle">Legal Intelligence System</div>
    </div>
    """, unsafe_allow_html=True)

    # API Status
    st.markdown('<div class="sidebar-section-title">حالة النظام</div>', unsafe_allow_html=True)
    api_ok, api_error = check_api_status()
    if not api_ok:
        st.markdown(f"""
        <div class="status-badge-error">
            <div style="width:8px;height:8px;border-radius:50%;background:#E74C3C;flex-shrink:0;"></div>
            API غير متصل — {api_error}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-badge">
            <div class="status-dot"></div>
            AI Service — متصل
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div class="mode-chip">🧠 Qwen2.5 3B + RAG</div>
    <br>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Settings
    st.markdown('<div class="sidebar-section-title">إعدادات البحث</div>', unsafe_allow_html=True)

    top_k = st.number_input(
        "عدد المصادر",
        min_value=1, max_value=15, value=6, step=1,
        help="عدد المواد القانونية التي سيتم استرجاعها لتكوين الإجابة"
    )

    st.markdown("---")

    # History
    st.markdown('<div class="sidebar-section-title">سجل الاستعلامات</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown('<div style="color:#4A5068;font-size:0.8rem;text-align:right;direction:rtl;">لا توجد استعلامات سابقة.</div>', unsafe_allow_html=True)
    else:
        for i, item in enumerate(reversed(st.session_state.history[-8:])):
            if st.button(
                f"🔍 {item['query'][:35]}{'…' if len(item['query']) > 35 else ''}",
                key=f"hist_{i}",
                use_container_width=True
            ):
                st.session_state.current_query = item["query"]
                st.rerun()
    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 مسح السجل", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="color:#4A5068;font-size:0.7rem;text-align:center;margin-top:1rem;line-height:1.8;">
        Lex AI – Phase 1<br>
        قانون العمل المصري<br>
        <span style="color:#2E3650;">Hybrid RAG · FAISS · BM25 · LLM</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# Page Header
st.markdown("""
<div class="page-header">
    <div class="page-title">⚖️ المساعد القانوني — قانون العمل المصري</div>
    <div class="page-desc">اسأل عن أي موضوع قانوني · نظام RAG مع استرجاع ذكي وتوليد إجابات</div>
</div>
""", unsafe_allow_html=True)

# API Error Banner (if not connected)
if not api_ok:
    st.markdown(f"""
    <div class="error-box">
        <div class="error-title">⚠️ الخادم الخلفي غير متصل</div>
        <div class="error-detail">{api_error}</div>
        <br>
        <div style="color:#8A91A8;font-size:0.82rem;">
            تأكد من تشغيل خادم FastAPI باستخدام الأمر:<br>
            <code>uvicorn api.routes:app --reload</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Search Input ──────────────────────────────────────────────────────────────
st.markdown('<div class="search-wrapper">', unsafe_allow_html=True)

query_input = st.text_area(
    label="",
    value=st.session_state.current_query,
    placeholder="اسأل عن موضوع قانوني مثل: متى يجوز وقف العامل عن العمل؟",
    height=100,
    key="query_box",
    label_visibility="collapsed"
)

col_btn, col_clear, col_spacer = st.columns([2, 1, 4])
with col_btn:
    search_clicked = st.button("🔍  بحث في القانون", use_container_width=True, disabled=(not api_ok))
with col_clear:
    if st.button("✕  مسح", use_container_width=True):
        st.session_state.current_query = ""
        st.session_state.last_result = None
        st.session_state.last_query = ""
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ── Run Search (call API) ─────────────────────────────────────────────────────
if search_clicked and query_input.strip() and api_ok:
    with st.spinner("⚖️ جارٍ البحث والتحليل بواسطة الذكاء الاصطناعي..."):
        try:
            payload = {"query": query_input.strip(), "top_k": int(top_k)}
            start_time = time.time()
            response = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
            elapsed = time.time() - start_time
            if response.status_code == 200:
                data = response.json()
                st.session_state.last_result = data
                st.session_state.last_query = query_input.strip()
                st.session_state.current_query = query_input.strip()
                # Add to history
                ts = datetime.now().strftime("%H:%M")
                if not st.session_state.history or st.session_state.history[-1]["query"] != query_input.strip():
                    st.session_state.history.append({
                        "query": query_input.strip(),
                        "time": ts,
                        "count": len(data.get("sources", []))
                    })
                st.rerun()
            else:
                st.error(f"خطأ من الخادم: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            st.error(f"فشل الاتصال بالخادم: {str(e)}")

# ── Display Result ───────────────────────────────────────────────────────────
result = st.session_state.last_result
query = st.session_state.last_query

if result and query:
    answer = result.get("answer", "لا توجد إجابة.")
    sources = result.get("sources", [])

    # Stats row
    st.markdown(f"""
    <div class="results-header">
        <div class="results-title">الإجابة القانونية</div>
        <div class="results-count">{len(sources)} مصدر · "{query[:40]}{'…' if len(query) > 40 else ''}"</div>
    </div>
    """, unsafe_allow_html=True)

    # Display answer in a beautiful box
    st.markdown(f"""
    <div class="answer-text">{answer}</div>
    """, unsafe_allow_html=True)

    # Sources section
    if sources:
        st.markdown("""
        <div style="margin: 1rem 0 0.5rem;">
            <span style="font-size:0.85rem;color:var(--gold);letter-spacing:2px;">📚 المصادر القانونية المستخدمة</span>
        </div>
        """, unsafe_allow_html=True)

        for rank, src in enumerate(sources, 1):
            article_id = src.get("article_id", "—")
            article_title = src.get("article_title", "بدون عنوان")
            score = src.get("score", 0.0)
            text_content = src.get("text", "")

            if score >= 0.7:
                score_cls = "score-high"
                score_icon = "●"
            elif score >= 0.4:
                score_cls = "score-mid"
                score_icon = "●"
            else:
                score_cls = "score-low"
                score_icon = "●"

            st.markdown(f"""
            <div class="result-card">
                <div class="card-header">
                    <div style="display:flex;align-items:center;gap:0.6rem;">
                        <div class="rank-badge">{rank}</div>
                        <div>
                            <div class="card-article-num">المادة {article_id}</div>
                        </div>
                    </div>
                    <div class="score-badge {score_cls}">{score_icon} {score:.4f}</div>
                </div>
                <div class="card-title">{article_title}</div>
            </div>
            """, unsafe_allow_html=True)

            if text_content:
                # معاينة قصيرة
                preview = text_content[:200] + "…" if len(text_content) > 200 else text_content
                st.markdown(f'<div class="card-snippet">{preview}</div>', unsafe_allow_html=True)
                # النص الكامل في expander
                with st.expander(f"📖 عرض النص الكامل للمادة {article_id}"):
                    st.markdown(f'<div class="card-full-text">{text_content}</div>', unsafe_allow_html=True)
        # Download sources as JSON
        import json
        export_data = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        st.download_button(
            label="📥 تصدير النتائج (JSON)",
            data=json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"lex_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.warning("لم يتم العثور على مصادر مرتبطة بالإجابة.")

elif query:
    # Search ran but no result object
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <div class="empty-text">لم يتم العثور على نتائج</div>
        <div class="empty-subtext">حاول صياغة السؤال بطريقة مختلفة</div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Initial empty state
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⚖️</div>
        <div class="empty-text">اكتب سؤالك القانوني للبدء</div>
        <div class="empty-subtext">
            سيقوم النظام بالبحث في قاعدة بيانات قانون العمل المصري<br>
            ثم توليد إجابة مستندة إلى المصادر باستخدام نموذج الذكاء الاصطناعي.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<br><br>
<div style="text-align:center;color:#2E3650;font-size:0.7rem;letter-spacing:2px;border-top:1px solid #1C2130;padding-top:1.5rem;">
    LEX AI · RAG SYSTEM · EGYPTIAN LABOR LAW · Qwen2.5 3B · HYBRID FAISS+BM25
</div>
""", unsafe_allow_html=True)