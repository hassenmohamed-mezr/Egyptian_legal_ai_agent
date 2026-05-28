import os
import time
from typing import Any, Dict, Optional, Sequence

import requests

from llm.prompts import (
    LEGAL_SYSTEM_PROMPT,
    LEGAL_USER_PROMPT_TEMPLATE,
    NARRATIVE_USER_PROMPT_TEMPLATE,
    build_legal_context,
)


# ─────────────────────────────────────────────
# Retry POST
# ─────────────────────────────────────────────
def post_to_ollama(
    payload: Dict,
    api_url: str,
    timeout: int = 300,
    retry_attempts: int = 1,
    retry_delay: int = 2,
) -> Dict:

    last_error = None

    for _ in range(retry_attempts + 1):
        try:
            response = requests.post(
                api_url,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as e:
            last_error = e
            time.sleep(retry_delay)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")

    raise RuntimeError(f"Ollama failed after retries: {last_error}")


# ─────────────────────────────────────────────
# MAIN GENERATE FUNCTION
# ─────────────────────────────────────────────
def generate_answer(
    query: str,
    chunks: Sequence[Dict[str, Any]],
    model: str = "qwen2.5:7b",
    host: Optional[str] = None,
    max_context_chars: int = 8000,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    retry_attempts: int = 1,
    retry_delay: int = 2,
    is_narrative: bool = False,        # ← جديد
) -> str:

    host = host or os.environ.get(
        "OLLAMA_HOST",
        "http://localhost:11434"
    )

    api_generate_url = f"{host.rstrip('/')}/api/generate"

    # Build context
    context = build_legal_context(chunks, max_chars=max_context_chars)

    # اختار الـ template المناسب
    if is_narrative:
        prompt = NARRATIVE_USER_PROMPT_TEMPLATE.format(
            narrative=query,
            context=context,
        )
    else:
        prompt = LEGAL_USER_PROMPT_TEMPLATE.format(
            query=query,
            context=context,
        )

    full_prompt = f"{LEGAL_SYSTEM_PROMPT}\n\n{prompt}"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    data = post_to_ollama(
        payload=payload,
        api_url=api_generate_url,
        retry_attempts=retry_attempts,
        retry_delay=retry_delay,
    )

    return data.get("response", "").strip()