import json
import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "law_ar_rag_optimized.json"

AR_DIACRITICS = re.compile(
    r'[\u064B-\u065F\u0670\u06D6-\u06ED]'
)

ARABIC_STOPWORDS = {
    "في",
    "من",
    "على",
    "الى",
    "إلى",
    "عن",
    "ما",
    "متى",
    "هل",
    "ثم",
    "او",
    "أو",
    "و",
    "يا",
    "هو",
    "هي"
}


def normalize(text: str):

    text = str(text)

    text = re.sub(AR_DIACRITICS, "", text)

    text = (
        text
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def tokenize(text: str):

    text = normalize(text)

    tokens = re.findall(
        r"[\w\u0600-\u06FF]+",
        text
    )

    tokens = [
        t for t in tokens
        if t not in ARABIC_STOPWORDS
        and len(t) > 1
    ]

    return tokens


_chunks = []
_bm25 = None


def load_bm25():
    global _chunks, _bm25
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        _chunks = json.load(f)

    corpus = []

    for c in _chunks:
        text = " ".join([
            c.get("retrieval_text", c.get("text", "")),
            c.get("article_title", "")
        ])

        corpus.append(tokenize(text))

    _bm25 = BM25Okapi(corpus)


def search_bm25(query: str, top_k=10):
    global _chunks, _bm25
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
    global _chunks
    return _chunks


load_bm25()