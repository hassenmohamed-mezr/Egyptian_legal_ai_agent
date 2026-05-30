import numpy as np

from rag.faiss_store import search_faiss
from rag.bm25_store import search_bm25
from rag.reranker import rerank as cross_encoder_rerank
from rag.hybrid_retriever.scoring import normalize_scores
from rag.hybrid_retriever.article_index import _merge_article_text


def hybrid_search(query: str, top_k: int = 8):

    candidate_k = max(top_k * 4, 25)

    faiss_results = search_faiss(query, top_k=candidate_k)
    bm25_results  = search_bm25(query, top_k=candidate_k)

    doc_lookup = {}
    for r in faiss_results + bm25_results:
        doc_lookup[r["chunk_id"]] = r

    all_ids = list(doc_lookup.keys())
    if not all_ids:
        return []

    faiss_map = {r["chunk_id"]: r["score"] for r in faiss_results}
    bm25_map  = {r["chunk_id"]: r["score"] for r in bm25_results}

    faiss_scores = np.array([faiss_map.get(i, 0.0) for i in all_ids])
    bm25_scores  = np.array([bm25_map.get(i,  0.0) for i in all_ids])

    faiss_scores = normalize_scores(faiss_scores)

    bm25_scores = np.log1p(bm25_scores)
    bm25_scores = normalize_scores(bm25_scores)

    query_len = len(query.split())
    if query_len <= 3:
        faiss_weight, bm25_weight = 0.55, 0.45
    else:
        faiss_weight, bm25_weight = 0.72, 0.28

    final_scores = faiss_weight * faiss_scores + bm25_weight * bm25_scores

    order = np.argsort(final_scores)[::-1]

    article_best: dict = {}

    for idx in order:
        cid = all_ids[idx]
        doc = doc_lookup[cid]
        aid = doc["article_id"]

        if aid in article_best:
            continue

        article_best[aid] = {
            "chunk_id":     cid,
            "article_id":   aid,
            "article_title": doc.get("article_title", ""),
            "score":        float(final_scores[idx]),
            "faiss_score":  float(faiss_map.get(cid, 0.0)),
            "bm25_score":   float(bm25_map.get(cid,  0.0)),
        }

    results = []
    for aid, meta in article_best.items():
        full_text = _merge_article_text(aid)

        results.append({
            "chunk_id":      meta["chunk_id"],
            "article_id":    aid,
            "article_title": meta["article_title"],
            "text":          full_text,
            "score":         meta["score"],
            "faiss_score":   meta["faiss_score"],
            "bm25_score":    meta["bm25_score"],
            "source":        "hybrid",
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    filtered = [
        r for r in results
        if r["faiss_score"] >= 0.55 or r["bm25_score"] >= 1.5
    ]

    if not filtered:
        filtered = results[:12]

    reranked = cross_encoder_rerank(
        query=query,
        results=filtered,
        top_k=top_k,
    )

    return reranked
