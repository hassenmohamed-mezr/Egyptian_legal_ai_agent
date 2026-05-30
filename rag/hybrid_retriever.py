import numpy as np

from rag.faiss_store import search_faiss
from rag.bm25_store import search_bm25, get_bm25_chunks
from rag.reranker import rerank as cross_encoder_rerank


# =========================
# ARTICLE FULL-TEXT CACHE
# Built once from the loaded BM25 chunks (which already holds the full dataset)
# Key: article_id → list of chunks sorted by chunk_order
# =========================

def _build_article_index():
    index = {}
    for chunk in get_bm25_chunks():
        aid = chunk["article_id"]
        if aid not in index:
            index[aid] = []
        index[aid].append(chunk)
    for aid in index:
        index[aid].sort(key=lambda c: c.get("chunk_order", 0))
    return index


_ARTICLE_INDEX = None


def _get_article_index():
    global _ARTICLE_INDEX
    if _ARTICLE_INDEX is None:
        _ARTICLE_INDEX = _build_article_index()
    return _ARTICLE_INDEX


# =========================
# HELPERS
# =========================

def normalize_scores(scores):
    scores = np.array(scores, dtype=np.float32)
    if len(scores) == 0:
        return scores
    min_v = scores.min()
    max_v = scores.max()
    if max_v - min_v < 1e-6:
        return np.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v)


def _merge_article_text(article_id: str) -> str:
    """
    Returns the full concatenated text of all chunks for an article,
    in chunk_order. Falls back gracefully if article_id is unknown.
    """
    article_index = _get_article_index()
    chunks = article_index.get(article_id, [])
    if not chunks:
        return ""
    return "\n".join(c.get("text", "") for c in chunks)


# =========================
# MAIN
# =========================

def hybrid_search(query: str, top_k: int = 8):

    candidate_k = max(top_k * 4, 25)

    # =========================
    # RETRIEVAL
    # =========================

    faiss_results = search_faiss(query, top_k=candidate_k)
    bm25_results  = search_bm25(query, top_k=candidate_k)

    # =========================
    # BUILD LOOKUPS
    # =========================

    doc_lookup = {}
    for r in faiss_results + bm25_results:
        doc_lookup[r["chunk_id"]] = r

    all_ids = list(doc_lookup.keys())
    if not all_ids:
        return []

    # =========================
    # SCORE MAPS
    # =========================

    faiss_map = {r["chunk_id"]: r["score"] for r in faiss_results}
    bm25_map  = {r["chunk_id"]: r["score"] for r in bm25_results}

    # =========================
    # NORMALIZE SCORES
    # =========================

    faiss_scores = np.array([faiss_map.get(i, 0.0) for i in all_ids])
    bm25_scores  = np.array([bm25_map.get(i,  0.0) for i in all_ids])

    faiss_scores = normalize_scores(faiss_scores)

    bm25_scores = np.log1p(bm25_scores)        # log squash
    bm25_scores = normalize_scores(bm25_scores)

    # =========================
    # DYNAMIC FUSION WEIGHTS
    # =========================

    query_len = len(query.split())
    if query_len <= 3:
        faiss_weight, bm25_weight = 0.55, 0.45
    else:
        faiss_weight, bm25_weight = 0.72, 0.28

    final_scores = faiss_weight * faiss_scores + bm25_weight * bm25_scores

    # =========================
    # SORT + PER-ARTICLE BEST SCORE
    # Keep the BEST hybrid score for each article (first hit = highest score)
    # =========================

    order = np.argsort(final_scores)[::-1]

    # article_id → {best hybrid score, best faiss raw, best bm25 raw, repr chunk_id}
    article_best: dict = {}

    for idx in order:
        cid = all_ids[idx]
        doc = doc_lookup[cid]
        aid = doc["article_id"]

        if aid in article_best:
            continue                    # already recorded the best-scoring chunk

        article_best[aid] = {
            "chunk_id":     cid,
            "article_id":   aid,
            "article_title": doc.get("article_title", ""),
            "score":        float(final_scores[idx]),
            # raw (un-normalised) scores for filtering & reranker bonus
            "faiss_score":  float(faiss_map.get(cid, 0.0)),
            "bm25_score":   float(bm25_map.get(cid,  0.0)),
        }

    # =========================
    # BUILD MERGED RESULTS
    # Replace text with the FULL article (all chunks merged in order)
    # =========================

    results = []
    for aid, meta in article_best.items():
        full_text = _merge_article_text(aid)

        results.append({
            "chunk_id":      meta["chunk_id"],
            "article_id":    aid,
            "article_title": meta["article_title"],
            "text":          full_text,          # ← full article text
            "score":         meta["score"],
            "faiss_score":   meta["faiss_score"],
            "bm25_score":    meta["bm25_score"],
            "source":        "hybrid",
        })

    # Keep sorted by hybrid score
    results.sort(key=lambda x: x["score"], reverse=True)

    # =========================
    # PRE-RERANK FILTERING
    # =========================

    filtered = [
        r for r in results
        if r["faiss_score"] >= 0.55 or r["bm25_score"] >= 1.5
    ]

    if not filtered:
        filtered = results[:12]

    # =========================
    # RERANK  (sees full article text)
    # =========================

    reranked = cross_encoder_rerank(
        query=query,
        results=filtered,
        top_k=top_k,
    )

    return reranked