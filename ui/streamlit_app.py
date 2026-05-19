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

API_URL = "http://localhost:8000/chat"

# =========================================================
# SIMPLE DARK STYLE
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"]  {
    background-color: #0e1117;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.main {
    padding-top: 2rem;
}

.block-container {
    padding-top: 1rem;
}

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

.small {
    color: #8b949e;
    font-size: 13px;
}

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

.stButton button:hover {
    background: #f0c94d;
}

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

    top_k = st.slider(
        "عدد المصادر",
        min_value=1,
        max_value=10,
        value=5
    )

    st.markdown("---")

    st.subheader("حالة السيرفر")

    try:
        test = requests.get("http://localhost:8000", timeout=5)

        st.success("FastAPI متصل")

    except:
        st.error("FastAPI غير متصل")

    st.markdown("---")

    st.subheader("آخر الأسئلة")

    if len(st.session_state.history) == 0:
        st.caption("لا يوجد سجل")

    else:
        for item in reversed(st.session_state.history[-5:]):
            st.caption(f"• {item}")

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">⚖️ Lex AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">المساعد القانوني لقانون العمل المصري</div>',
    unsafe_allow_html=True
)

# =========================================================
# INPUT
# =========================================================

query = st.text_area(
    "",
    placeholder="اكتب سؤالك القانوني هنا...",
    height=150
)

# =========================================================
# SEARCH
# =========================================================

if st.button("بحث قانوني"):

    if not query.strip():

        st.warning("اكتب سؤال أولاً")

    else:

        with st.spinner("جارٍ التحليل القانوني..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "query": query,
                        "top_k": top_k
                    },
                    timeout=60
                )

                if response.status_code != 200:

                    st.error(f"Server Error: {response.status_code}")
                    st.code(response.text)

                else:

                    data = response.json()

                    answer = data.get("answer", "لا توجد إجابة")
                    sources = data.get("sources", [])

                    # حفظ في history
                    if query not in st.session_state.history:
                        st.session_state.history.append(query)

                    # =================================================
                    # ANSWER
                    # =================================================

                    st.markdown("## الإجابة القانونية")

                    st.markdown(
                        f"""
                        <div class="answer-box">
                        {answer}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # =================================================
                    # SOURCES
                    # =================================================

                    st.markdown("## المواد القانونية")

                    if len(sources) == 0:

                        st.info("لا توجد مصادر")

                    else:

                        for idx, src in enumerate(sources, start=1):

                            article_id = src.get("article_id", "؟")
                            article_title = src.get("article_title", "")
                            score = src.get("score", 0.0)
                            text = src.get("text", "")

                            preview = (
                                text[:350] + "..."
                                if len(text) > 350
                                else text
                            )

                            with st.expander(
                                f"{idx}) المادة {article_id} | Score = {score:.3f}"
                            ):

                                st.markdown(
                                    f"""
                                    <div class="source-box">

                                    <b>{article_title}</b>

                                    <br><br>

                                    {preview}

                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                                st.text_area(
                                    "النص الكامل",
                                    value=text,
                                    height=220,
                                    disabled=True,
                                    key=f"src_{idx}"
                                )

            except requests.exceptions.ConnectionError:

                st.error("لا يمكن الاتصال بخادم FastAPI")

            except requests.exceptions.Timeout:

                st.error("الطلب استغرق وقتًا طويلًا")

            except Exception as e:

                st.error(str(e))

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    f"Lex AI • Egyptian Labor Law • {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)