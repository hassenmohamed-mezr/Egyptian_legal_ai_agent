
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from rag.hybrid_retriever import hybrid_search

queries = [
    "متى يجوز وقف العامل عن العمل؟",
]

for q in queries:

    print("\n" + "=" * 80)
    print("QUERY:", q)

    results = hybrid_search(q)

    for r in results:

        print("ARTICLE:", r.get("article_title"))
        print("ARTICLE ID:", r.get("article_id"))
        print("SOURCE:", r.get("source", "unknown"))
        print("FINAL SCORE:", round(r.get("score", 0), 4))

        print("FAISS SCORE:", round(r.get("faiss_score", 0), 4))
        print("BM25 SCORE:", round(r.get("bm25_score", 0), 4))

        print("TEXT:")
        print(r.get("text", "")[:300])
        print("-" * 40)