import numpy as np
from rag.faiss_store import search_faiss
from rag.bm25_store import bm25_store
from rag.reranker import rerank as cross_encoder_rerank

def bm25_scale(x):
    x = np.array(x, dtype=np.float32)
    return np.log1p(x)

def hybrid_search(query: str, top_k: int = 8):
    # Retrieve more candidates from each source
    faiss_results = search_faiss(query, top_k=top_k * 5)
    bm25_results = bm25_store.search(query, top_k=top_k * 5)

    # Build score maps
    faiss_map = {r["chunk_id"]: r["score"] for r in faiss_results}
    bm25_map = {r["chunk_id"]: r["score"] for r in bm25_results}

    all_ids = list(set(faiss_map) | set(bm25_map))
    if not all_ids:
        return []

    # Normalize scores
    faiss_scores = np.array([faiss_map.get(i, 0.0) for i in all_ids])
    bm25_scores = np.array([bm25_map.get(i, 0.0) for i in all_ids])

    bm25_scores = bm25_scale(bm25_scores)
    faiss_scores = np.clip(faiss_scores, 0, 1)
    bm25_scores = bm25_scores / (bm25_scores.max() + 1e-6)

    final = faiss_scores + 0.7 * bm25_scores
    order = np.argsort(final)[::-1]

    # Deduplicate by article_id (first chunk per article)
    results = []
    seen_articles = set()
    for i in order:
        cid = all_ids[i]
        # Find the document metadata from either faiss_results or bm25_results
        doc = next((r for r in faiss_results if r["chunk_id"] == cid), None)
        if doc is None:
            doc = next((r for r in bm25_results if r["chunk_id"] == cid), None)
        if not doc:
            continue

        aid = doc["article_id"]
        if aid in seen_articles:
            continue
        seen_articles.add(aid)

        results.append({
            "chunk_id": cid,
            "article_id": aid,
            "article_title": doc["article_title"],
            "text": doc["text"],
            "score": float(final[i]),
            "faiss_score": float(faiss_map.get(cid, 0.0)),
            "bm25_score": float(bm25_map.get(cid, 0.0)),
            "source": "hybrid"
        })

    # Rerank the deduplicated candidates
    if results:
        results = cross_encoder_rerank(query, results, top_k=top_k)

    return results