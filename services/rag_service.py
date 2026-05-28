from typing import Any, Dict, List, Optional

from rag.hybrid_retriever import hybrid_search
from llm.ollama_client import generate_answer
from llm.query_rewriter import rewrite_query


# ─────────────────────────────────────────────
# FORMAT SOURCES
# ─────────────────────────────────────────────
def format_sources(
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    return [
        {
            "article_id":    item.get("article_id"),
            "article_title": item.get("article_title"),
            "chunk_id":      item.get("chunk_id"),
            "text":          item.get("text", ""),
            "score":         round(item.get("score", 0.0), 4),
        }
        for item in chunks
    ]


# ─────────────────────────────────────────────
# RETRIEVE CHUNKS  (multi-query + dedup)
# ─────────────────────────────────────────────
def retrieve_chunks(
    questions: List[str],
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    يبحث على كل سؤال ويدمج النتايج مع dedup بالـ article_id.
    بيحتفظ بأعلى score لكل مادة عبر كل الأسئلة.
    """
    seen: Dict[str, Dict[str, Any]] = {}   # article_id → best chunk

    for question in questions:
        results = hybrid_search(query=question, top_k=top_k)

        for chunk in results:
            aid = chunk.get("article_id")
            existing = seen.get(aid)

            if existing is None or chunk.get("score", 0) > existing.get("score", 0):
                seen[aid] = chunk

    # رتب حسب الـ score وخد top_k
    merged = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
    return merged[:top_k]


# ─────────────────────────────────────────────
# MAIN RAG PIPELINE
# ─────────────────────────────────────────────
def get_answer(
    query: str,
    top_k: int = 6,
    model: str = "qwen2.5:7b",
    host: Optional[str] = None,
) -> Dict[str, Any]:

    # ── 1. Rewrite / detect narrative ───────────────────────────────────
    rewrite_result = rewrite_query(
        text=query,
        model=model,
        host=host,
    )

    is_narrative   = rewrite_result["is_narrative"]
    questions      = rewrite_result["questions"]

    # ── 2. Retrieve ──────────────────────────────────────────────────────
    chunks = retrieve_chunks(questions=questions, top_k=top_k)

    sources = format_sources(chunks)

    if not chunks:
        return {
            "answer": (
                "لا توجد معلومات كافية في النصوص القانونية "
                "المتاحة للإجابة على هذا السؤال."
            ),
            "sources":       [],
            "query":         query,
            "is_narrative":  is_narrative,
            "questions_used": questions,
        }

    # ── 3. Generate ──────────────────────────────────────────────────────
    # نمرر النص الأصلي للـ LLM دايماً (قصة أو سؤال)
    answer = generate_answer(
        query=query,
        chunks=chunks,
        model=model,
        host=host,
        is_narrative=is_narrative,
    )

    return {
        "answer":         answer,
        "sources":        sources,
        "query":          query,
        "is_narrative":   is_narrative,
        "questions_used": questions,
    }