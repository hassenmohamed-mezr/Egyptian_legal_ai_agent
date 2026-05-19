import json
import os
import time
from typing import Any, Dict, Iterator, Optional, Sequence

import requests

from llm.prompts import (
    LEGAL_SYSTEM_PROMPT,
    LEGAL_USER_PROMPT_TEMPLATE,
    build_legal_context,
)


class OllamaClient:
    """
    Lightweight RAG client for Ollama (optimized for latency + Streamlit).
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: Optional[str] = None,
        max_context_chars: int = 8000,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: int = 60,
        retry_attempts: int = 1,
        retry_delay: int = 2,
    ):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.base_url = self.host.rstrip("/")
        self.api_generate_url = f"{self.base_url}/api/generate"

        self.max_context_chars = max_context_chars
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    # ─────────────────────────────────────────────
    # Health check (خفيف جدًا)
    # ─────────────────────────────────────────────
    def _check_ollama(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
        except Exception:
            raise RuntimeError("Ollama server is not reachable. Run: ollama serve")

    # ─────────────────────────────────────────────
    # Retry POST (خفيف)
    # ─────────────────────────────────────────────
    def _post(self, payload: Dict) -> Dict:
        last_error = None

        for _ in range(self.retry_attempts + 1):
            try:
                r = requests.post(
                    self.api_generate_url,
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json()

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                time.sleep(self.retry_delay)
                continue

            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Ollama request failed: {e}")

        raise RuntimeError(f"Ollama failed after retries: {last_error}")

    # ─────────────────────────────────────────────
    # MAIN GENERATE
    # ─────────────────────────────────────────────
    def generate(
        self,
        query: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> str:

        self._check_ollama()

        context = build_legal_context(
            chunks,
            max_chars=self.max_context_chars
        )

        prompt = LEGAL_USER_PROMPT_TEMPLATE.format(
            query=query,
            context=context
        )

        full_prompt = f"{LEGAL_SYSTEM_PROMPT}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        data = self._post(payload)
        return data.get("response", "").strip()

    # ─────────────────────────────────────────────
    # STREAMING (اختياري لـ Streamlit UI)
    # ─────────────────────────────────────────────
    def stream_generate(
        self,
        query: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> Iterator[str]:

        self._check_ollama()

        context = build_legal_context(
            chunks,
            max_chars=self.max_context_chars
        )

        prompt = LEGAL_USER_PROMPT_TEMPLATE.format(
            query=query,
            context=context
        )

        full_prompt = f"{LEGAL_SYSTEM_PROMPT}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        with requests.post(
            self.api_generate_url,
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as r:

            r.raise_for_status()

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "error" in chunk:
                    raise RuntimeError(chunk["error"])

                token = chunk.get("response", "")
                if token:
                    yield token

                if chunk.get("done"):
                    break