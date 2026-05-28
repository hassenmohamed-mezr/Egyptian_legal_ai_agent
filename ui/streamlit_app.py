import streamlit as st
import requests
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Lex AI",
    page_icon="⚖️",
    layout="wide"
)

API_URL  = "http://localhost:8000/chat"
TIMEOUT  = 300   # 5 دقايق — يكفي أي pipeline

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"]  {
    background-color: #0e1117;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

.main            { padding-top: 2rem; }
.block-container { padding-top: 1rem; }

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #d4af37;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #9aa4b2;
    margin-bottom: 40px;
}

.narrative-badge {
    display: inline-block;
    background: #1f3a5f;
    color: #6ab0f5;
    border: 1px solid #3a6ea8;
    border-radius: 8px;
    padding: 4px 14px;
    font-size: 14px;
    margin-bottom: 14px;
    direction: rtl;
}

.questions-box {
    background: #161b22;
    border: 1px solid #3a6ea8;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    direction: rtl;
}

.questions-box ul {
    margin: 8px 0 0 0;
    padding-right: 20px;
}

.questions-box li {
    color: #c9d1d9;
    margin-bottom: 6px;
    font-size: 15px;
}

.answer-box {
    background: #161b22;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #30363d;
    line-height: 2;
    font-size: 17px;
    direction: rtl;
    text-align: right;
}

.source-box {
    background: #161b22;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #30363d;
    margin-bottom: 12px;
    direction: rtl;
}

.step-status {
    background: #161b22;
    border-radius: 10px;
    border: 1px solid #30363d;
    padding: 14px 20px;
    margin-bottom: 8px;
    direction: rtl;
    font-size: 15px;
}

.step-done    { border-color: #238636; color: #3fb950; }
.step-running { border-color: #d4af37; color: #d4af37; }
.step-pending { color: #484f58; }

.stTextArea textarea {
    background-color: #161b22 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #30363d !important;
    direction: rtl;
    text-align: right;
    font-size: 16px !important;
}

.stButton button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    border: none;
    background: #d4af37;
    color: black;
    font-size: 18px;
    font-weight: bold;
}

.stButton button:hover { background: #f0c94d; }

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚖️ Lex AI")
    st.markdown("---")
    st.subheader("الإعدادات")

    top_k = st.slider("عدد المصادر", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.subheader("حالة السيرفر")

    try:
        requests.get("http://localhost:8000", timeout=4)
        st.success("FastAPI متصل ✓")
    except Exception:
        st.error("FastAPI غير متصل ✗")

    st.markdown("---")
    st.subheader("آخر الأسئلة")

    if not st.session_state.history:
        st.caption("لا يوجد سجل")
    else:
        for item in reversed(st.session_state.history[-5:]):
            st.caption(f"• {item}")

# =========================================================
# HEADER
# =========================================================

st.markdown('<div class="title">⚖️ Lex AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">المساعد القانوني لقانون العمل المصري</div>',
    unsafe_allow_html=True,
)

# =========================================================
# INPUT
# =========================================================

query = st.text_area(
    "",
    placeholder="اكتب سؤالك القانوني أو اشرح موقفك هنا...",
    height=150,
)

# =========================================================
# SEARCH
# =========================================================

if st.button("بحث قانوني"):

    if not query.strip():
        st.warning("اكتب سؤالاً أو اشرح موقفك أولاً")

    else:

        # ── Progress UI ───────────────────────────────────────
        progress_area = st.container()

        with progress_area:
            step1 = st.empty()
            step2 = st.empty()
            step3 = st.empty()
            step4 = st.empty()

        def set_step(n, label, state="running"):
            """state: running | done | pending"""
            icon  = {"running": "⏳", "done": "✅", "pending": "○"}[state]
            cls   = {"running": "step-running", "done": "step-done", "pending": "step-pending"}[state]
            slot  = [step1, step2, step3, step4][n - 1]
            slot.markdown(
                f'<div class="step-status {cls}">{icon} {label}</div>',
                unsafe_allow_html=True,
            )

        set_step(1, "تحليل نوع السؤال أو القصة...", "running")
        set_step(2, "البحث في قاعدة القانون...",    "pending")
        set_step(3, "ترتيب النتائج وتصفيتها...",    "pending")
        set_step(4, "توليد الإجابة القانونية...",    "pending")

        try:
            response = requests.post(
                API_URL,
                json={"query": query, "top_k": top_k},
                timeout=TIMEOUT,
            )

            # ── بعد ما الـ API يرجع، حدّث الـ steps بناءً على الـ response ──
            if response.status_code == 200:
                data         = response.json()
                is_narrative = data.get("is_narrative", False)
                questions    = data.get("questions_used", [query])

                step1_label = (
                    "تم التعرف على القصة واستخراج الأسئلة"
                    if is_narrative
                    else "تم التعرف على السؤال المباشر"
                )
                set_step(1, step1_label,                         "done")
                set_step(2, "تم البحث في قاعدة القانون",         "done")
                set_step(3, "تم ترتيب النتائج وتصفيتها",         "done")
                set_step(4, "تم توليد الإجابة القانونية",         "done")

            else:
                set_step(1, "حدث خطأ في الاتصال", "running")
                st.error(f"Server Error: {response.status_code}")
                st.code(response.text)
                st.stop()

        except requests.exceptions.Timeout:
            set_step(1, "انتهت مهلة الاتصال — حاول مرة أخرى", "running")
            st.error(
                "الطلب استغرق وقتاً طويلاً. "
                "تأكد أن الـ GPU يعمل وأن Ollama نشيط."
            )
            st.stop()

        except requests.exceptions.ConnectionError:
            st.error("لا يمكن الاتصال بخادم FastAPI")
            st.stop()

        except Exception as e:
            st.error(str(e))
            st.stop()

        # ── حفظ في history ─────────────────────────────────────
        if query not in st.session_state.history:
            st.session_state.history.append(query)

        answer  = data.get("answer",  "لا توجد إجابة")
        sources = data.get("sources", [])

        st.markdown("---")

        # ── لو قصة: اعرض الـ badge والأسئلة المستخرجة ──────────
        if is_narrative:
            st.markdown(
                '<div class="narrative-badge">📖 تم تحليل السيناريو / القصة</div>',
                unsafe_allow_html=True,
            )

            if questions:
                items_html = "".join(f"<li>{q}</li>" for q in questions)
                st.markdown(
                    f"""
                    <div class="questions-box">
                    <b>الأسئلة القانونية المستخرجة من قصتك:</b>
                    <ul>{items_html}</ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── الإجابة ─────────────────────────────────────────────
        st.markdown("## الإجابة القانونية")
        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True,
        )

        # ── المصادر ─────────────────────────────────────────────
        st.markdown("## المواد القانونية")

        if not sources:
            st.info("لا توجد مصادر")
        else:
            for idx, src in enumerate(sources, start=1):

                article_id    = src.get("article_id", "؟")
                article_title = src.get("article_title", "")
                score         = src.get("score", 0.0)
                text          = src.get("text", "")
                preview       = text[:350] + "..." if len(text) > 350 else text

                with st.expander(
                    f"{idx}) المادة {article_id} | Score = {score:.3f}"
                ):
                    st.markdown(
                        f"""
                        <div class="source-box">
                        <b>{article_title}</b><br><br>
                        {preview}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.text_area(
                        "النص الكامل",
                        value=text,
                        height=220,
                        disabled=True,
                        key=f"src_{idx}",
                    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption(
    f"Lex AI • Egyptian Labor Law • {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)