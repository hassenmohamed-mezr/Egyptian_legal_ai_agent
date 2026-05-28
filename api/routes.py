from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.rag_service import get_answer


app = FastAPI(title="Egyptian Legal AI Agent")


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    top_k: int = 6


class SourceItem(BaseModel):
    article_id:    Any
    article_title: str
    chunk_id:      Any
    score:         float
    text:          Optional[str] = None


class ChatResponse(BaseModel):
    answer:         str
    sources:        List[SourceItem]
    query:          str
    is_narrative:   bool               = False
    questions_used: List[str]          = []


# ─────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> Dict[str, Any]:
    try:
        return get_answer(
            query=request.query,
            top_k=request.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))