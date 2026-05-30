import numpy as np

from rag.bm25_store.preprocessing import tokenize
from rag.bm25_store.index import get_bm25_state


def search_bm25(query: str, top_k=10):
    _chunks, _bm25 = get_bm25_state()
    q = tokenize(query)
    scores = _bm25.get_scores(q)

    idxs = np.argsort(scores)[::-1][:top_k]

    results = []

    for i in idxs:
        c = _chunks[i]

        results.append({
            "chunk_id": c["chunk_id"],
            "article_id": c["article_id"],
            "article_title": c.get("article_title", ""),
            "text": c.get("text", ""),
            "score": float(scores[i]),
            "source": "bm25"
        })

    return results


def get_bm25_chunks():
    _chunks, _ = get_bm25_state()
    return _chunks
