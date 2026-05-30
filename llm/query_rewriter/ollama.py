import time

import requests


def _call_ollama(
    system: str,
    user: str,
    model: str,
    host: str,
    temperature: float = 0.1,
    max_tokens: int = 700,
    timeout: int = 90,
) -> str:

    url = f"{host.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": f"{system}\n\n{user}",
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    for attempt in range(2):

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout
            )

            response.raise_for_status()

            return response.json().get(
                "response",
                ""
            ).strip()

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ):

            if attempt == 0:
                time.sleep(1.5)
            else:
                raise

        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Ollama query rewriter error: {e}"
            )

    return ""
