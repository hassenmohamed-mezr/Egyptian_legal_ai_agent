import json
import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "law_ar_rag_optimized.json"

AR_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]')


def normalize(text: str):
    text = str(text)
    text = re.sub(AR_DIACRITICS, "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str):
    text = normalize(text)
    return re.findall(r"[\w\u0600-\u06FF]+", text)


class BM25Store:
    def __init__(self):
        self.chunks = []
        self.bm25 = None

    def load(self):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        corpus = []

        for c in self.chunks:
            text = " ".join([
                c.get("normalized_text", c.get("text", "")),
                c.get("article_title", "")
            ])

            corpus.append(tokenize(text))

        self.bm25 = BM25Okapi(corpus)
        print(f"BM25 Loaded with {len(self.chunks)} documents")

    def search(self, query: str, top_k=10):
        q = tokenize(query)
        scores = self.bm25.get_scores(q)

        idxs = np.argsort(scores)[::-1][:top_k]

        results = []

        for i in idxs:
            c = self.chunks[i]

            results.append({
                "chunk_id": c["chunk_id"],
                "article_id": c["article_id"],
                "article_title": c.get("article_title", ""),
                "text": c.get("text", ""),
                "score": float(scores[i]),
                "source": "bm25"
            })

        return results


bm25_store = BM25Store()
bm25_store.load()