import os
from typing import Any, Dict, List

from llm.query_rewriter.prompts import (
    _DETECT_SYSTEM,
    _DETECT_USER,
    _EXTRACT_SYSTEM,
    _EXTRACT_USER,
)
from llm.query_rewriter.ollama import _call_ollama
from llm.query_rewriter.parser import _parse_json_safe


def rewrite_query(
    text: str,
    model: str = "qwen2.5:7b",
    host: str | None = None,
) -> Dict[str, Any]:

    host = host or os.environ.get(
        "OLLAMA_HOST",
        "http://localhost:11434"
    )

    detect_raw = _call_ollama(
        system=_DETECT_SYSTEM,
        user=_DETECT_USER.format(text=text),
        model=model,
        host=host,
    )

    detect_data = _parse_json_safe(detect_raw)

    is_narrative = bool(
        detect_data.get("is_narrative", False)
    )

    if not is_narrative:

        return {
            "is_narrative": False,
            "original": text,
            "questions": [text],
        }

    extract_raw = _call_ollama(
        system=_EXTRACT_SYSTEM,
        user=_EXTRACT_USER.format(
            narrative=text
        ),
        model=model,
        host=host,
    )

    extract_data = _parse_json_safe(extract_raw)

    questions: List[str] = extract_data.get(
        "questions",
        []
    )

    cleaned_questions = []

    seen = set()

    for q in questions:

        if not isinstance(q, str):
            continue

        q = q.strip()

        if len(q) < 8:
            continue

        if q in seen:
            continue

        seen.add(q)

        cleaned_questions.append(q)

    if not cleaned_questions:
        cleaned_questions = [text]

    return {
        "is_narrative": True,
        "original": text,
        "questions": cleaned_questions[:15],
    }
