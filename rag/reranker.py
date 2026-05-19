from sentence_transformers import CrossEncoder
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device=device
)


def rerank(query, results, top_k=5):

    if not results:
        return []

    # limit input (IMPORTANT VRAM FIX)
    results = results[:20]

    pairs = [(query, r["text"][:512]) for r in results]

    scores = reranker.predict(pairs, batch_size=4)

    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)

    return results[:top_k]