import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence

import requests

from llm.prompts import LEGAL_SYSTEM_PROMPT, LEGAL_USER_PROMPT_TEMPLATE, build_legal_context


class OllamaClient:
    """
    Client for Ollama's native API (not OpenAI-compatible).
    Works with Qwen2.5:3B and other models.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: Optional[str] = None,
        max_context_chars: int = 8000,
        temperature: float = 0.0,
        max_tokens: int = 2024,
    ) -> None:
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.base_url = self.host.rstrip("/")
        self.api_generate_url = f"{self.base_url}/api/generate"
        self.max_context_chars = max_context_chars
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _check_ollama(self) -> None:
        """Quick check if Ollama is reachable and model is loaded."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            if not any(m.get("name") == self.model for m in models):
                # Model might not be loaded, but we'll let generate call handle it
                pass
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Is 'ollama serve' running?"
            ) from e

    def generate(
        self,
        query: str,
        chunks: Sequence[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response using Ollama's /api/generate endpoint.
        Returns the full generated text.
        """
        self._check_ollama()

        prompt = self._build_prompt(query, chunks)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            },
        }

        try:
            response = requests.post(self.api_generate_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Unable to connect to Ollama at {self.base_url}. Please run 'ollama serve'."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        if "error" in data:
            raise RuntimeError(f"Ollama error: {data['error']}")

        # Ollama returns {"response": "...", "done": true, ...}
        return data.get("response", "").strip()

    def stream_generate(
        self,
        query: str,
        chunks: Sequence[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Stream generation token by token using Ollama's streaming mode.
        Yields each piece of the response as it arrives.
        """
        self._check_ollama()

        prompt = self._build_prompt(query, chunks)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            },
        }

        try:
            with requests.post(self.api_generate_url, json=payload, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        raise RuntimeError(f"Ollama stream error: {chunk['error']}")
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Unable to connect to Ollama at {self.base_url}. Please run 'ollama serve'."
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama streaming request failed: {e}") from e

    def _build_prompt(self, query: str, chunks: Sequence[Dict[str, Any]]) -> str:
        """Combine system prompt, context, and user query into a single prompt."""
        context = build_legal_context(chunks, max_chars=self.max_context_chars)
        user_prompt = LEGAL_USER_PROMPT_TEMPLATE.format(query=query, context=context)
        # Some models work better when system prompt is prepended directly
        return f"{LEGAL_SYSTEM_PROMPT}\n\n{user_prompt}"