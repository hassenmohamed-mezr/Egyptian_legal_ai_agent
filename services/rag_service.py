from typing import Any, Dict, List, Optional

from rag.hybrid_retriever import hybrid_search
from llm.ollama_client import OllamaClient


class RAGService:
    """Orchestrates retrieval and Ollama generation for legal question answering."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: Optional[str] = None,
        top_k: int = 6,
    ) -> None:
        self.client = OllamaClient(model=model, host=host)
        self.top_k = top_k

    def get_answer(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve relevant chunks and generate a legal answer."""
        effective_top_k = top_k if top_k is not None else self.top_k
        chunks = hybrid_search(query, top_k=effective_top_k)

        sources: List[Dict[str, Any]] = [
            {
                "article_id": item.get("article_id"),
                "article_title": item.get("article_title"),
                "chunk_id": item.get("chunk_id"),
                "text": item.get("text", ""), 
                "score": round(item.get("score", 0.0), 4),
            }
            for item in chunks
        ]

        if not chunks:
            return {
                "answer": "لا توجد معلومات كافية في النصوص القانونية المتاحة للإجابة على هذا السؤال.",
                "sources": [],
                "query": query,
            }

        answer = self.client.generate(query=query, chunks=chunks)
        return {
            "answer": answer,
            "sources": sources,
            "query": query,
        }

    def stream_answer(self, query: str, top_k: Optional[int] = None):
        """Stream the generated answer token by token."""
        effective_top_k = top_k if top_k is not None else self.top_k
        chunks = hybrid_search(query, top_k=effective_top_k)

        if not chunks:
            yield "لا توجد معلومات كافية في النصوص القانونية المتاحة للإجابة على هذا السؤال."
            return

        yield from self.client.stream_generate(query=query, chunks=chunks)