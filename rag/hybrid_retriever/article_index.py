from rag.bm25_store import get_bm25_chunks


_ARTICLE_INDEX = None


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


def _get_article_index():
    global _ARTICLE_INDEX
    if _ARTICLE_INDEX is None:
        _ARTICLE_INDEX = _build_article_index()
    return _ARTICLE_INDEX


def _merge_article_text(article_id: str) -> str:
    article_index = _get_article_index()
    chunks = article_index.get(article_id, [])
    if not chunks:
        return ""
    return "\n".join(c.get("text", "") for c in chunks)
