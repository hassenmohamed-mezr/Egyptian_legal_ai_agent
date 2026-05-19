from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.rag_service import RAGService

app = FastAPI(title="Egyptian Legal AI Agent")
rag_service = RAGService()


class ChatRequest(BaseModel):
    query: str
    top_k: int = 6


class SourceItem(BaseModel):
    article_id: Any
    article_title: str
    chunk_id: Any
    score: float
    text: Optional[str] = None 


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    query: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> Dict[str, Any]:
    try:
        return rag_service.get_answer(request.query, top_k=request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Streaming endpoint – returns Server‑Sent Events."""
    def event_generator():
        for token in rag_service.stream_answer(request.query, top_k=request.top_k):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")