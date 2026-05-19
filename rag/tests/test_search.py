
import sys
from pathlib import Path

# أضف المجلد الرئيسي للمشروع (حيث يحتوي على مجلد 'rag')
root_dir = Path(__file__).resolve().parent.parent.parent  # يصل إلى egyptian_legal_ai_agent
sys.path.insert(0, str(root_dir))

from rag.faiss_store import search_faiss


def print_results(query, results):

    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)

    for i, r in enumerate(results):

        data = r["data"]
        score = r["score"]

        print(f"\n[{i+1}] SCORE: {score}")
        print(f"ARTICLE: {data['article_id']} - {data['article_title']}")
        print(f"TEXT: {data['text'][:300]}...")


if __name__ == "__main__":

    queries = [
        "متى يجوز وقف العامل عن العمل؟",
        "الغرامة على صاحب العمل المخالف",
        "تظلم العامل من قرار الوقف",
        "مخالفة قوانين إلحاق العمالة بالخارج",
        "جزاءات العامل وخصم الأجر"
    ]

    for q in queries:

        results = search_faiss(q, top_k=3)

        print_results(q, results)