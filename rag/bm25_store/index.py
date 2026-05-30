import json
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

from rag.bm25_store.preprocessing import tokenize


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "law_ar_rag_optimized.json"

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


def get_bm25_state():
    return _chunks, _bm25
