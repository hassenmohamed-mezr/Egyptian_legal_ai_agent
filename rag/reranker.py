from sentence_transformers import CrossEncoder
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device=device,
    trust_remote_code=True
)

if device == "cuda":
    reranker.model.half()

torch.set_grad_enabled(False)


def rerank(query, results, top_k=5):

    if not results:
        return []

    # Keep enough candidates
    results = results[:15]

    pairs = [
        (
            query,
            r["text"][:1024]
        )
        for r in results
    ]

    with torch.inference_mode():

        scores = reranker.predict(
            pairs,
            batch_size=4,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    for r, s in zip(results, scores):

        rerank_score = float(s)

        # Hybrid rerank fusion
        hybrid_bonus = (
            0.15 * r.get("faiss_score", 0.0)
        )

        r["rerank_score"] = (
            rerank_score + hybrid_bonus
        )

    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results[:top_k]