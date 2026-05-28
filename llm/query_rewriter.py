"""
query_rewriter.py
─────────────────
يأخد نص المستخدم (سؤال مباشر أو قصة/سيناريو) ويرجع:
  - is_narrative: True لو النص قصة أو سيناريو
  - questions: قائمة أسئلة قانونية مستخرجة للبحث
"""

import json
import os
import re
import time
from typing import Any, Dict, List

import requests


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

_DETECT_SYSTEM = """
أنت محلل نصوص قانونية. مهمتك تحديد نوع النص الذي يكتبه المستخدم.
أجب بـ JSON فقط بدون أي نص إضافي.
""".strip()

_DETECT_USER = """
النص التالي: هل هو سؤال قانوني مباشر أم قصة/سيناريو يحتاج تحليل؟

النص:
{text}

أجب بـ JSON فقط بهذا الشكل:
{{
  "is_narrative": true أو false,
  "reason": "سبب قصير جداً"
}}
""".strip()


_EXTRACT_SYSTEM = """
أنت محلل قانوني متخصص في قانون العمل المصري.
مهمتك: استخراج الأسئلة القانونية الصريحة من قصة أو سيناريو يرويه المستخدم.
أجب بـ JSON فقط بدون أي نص إضافي أو markdown.
""".strip()

_EXTRACT_USER = """
القصة/السيناريو:
{narrative}

استخرج من هذه القصة الأسئلة القانونية التي يجب البحث عنها في قانون العمل المصري.
- اجعل كل سؤال مستقلاً وواضحاً
- ركز على الحقوق والالتزامات والإجراءات القانونية
- لا تزيد عن 4 أسئلة
- رتبها من الأهم للأقل أهمية

أجب بـ JSON فقط:
{{
  "questions": [
    "السؤال الأول",
    "السؤال الثاني"
  ]
}}
""".strip()


# ─────────────────────────────────────────────
# Ollama helper (نفس نمط ollama_client)
# ─────────────────────────────────────────────

def _call_ollama(
    system: str,
    user: str,
    model: str,
    host: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
    timeout: int = 60,
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
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError):
            if attempt == 0:
                time.sleep(1)
            else:
                raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama rewriter error: {e}")

    return ""


def _parse_json_safe(text: str) -> Dict[str, Any]:
    """
    يحاول يعمل parse للـ JSON حتى لو فيه markdown fences.
    """
    # شيل ```json ... ```
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # حاول تستخرج أول {} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def rewrite_query(
    text: str,
    model: str = "qwen2.5:7b",
    host: str | None = None,
) -> Dict[str, Any]:
    """
    يحلل نص المستخدم ويرجع:
    {
        "is_narrative": bool,
        "original":     str,     # النص الأصلي
        "questions":    list[str] # أسئلة للبحث
    }

    لو is_narrative=False → questions = [text] (السؤال نفسه)
    لو is_narrative=True  → questions مستخرجة من القصة
    """
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    # ── 1. اكتشاف نوع النص ──────────────────────────────────────────────
    detect_raw = _call_ollama(
        system=_DETECT_SYSTEM,
        user=_DETECT_USER.format(text=text),
        model=model,
        host=host,
    )
    detect_data = _parse_json_safe(detect_raw)
    is_narrative = bool(detect_data.get("is_narrative", False))

    # ── 2. لو مش قصة → رجّع السؤال كما هو ─────────────────────────────
    if not is_narrative:
        return {
            "is_narrative": False,
            "original": text,
            "questions": [text],
        }

    # ── 3. استخراج الأسئلة من القصة ─────────────────────────────────────
    extract_raw = _call_ollama(
        system=_EXTRACT_SYSTEM,
        user=_EXTRACT_USER.format(narrative=text),
        model=model,
        host=host,
    )
    extract_data = _parse_json_safe(extract_raw)
    questions: List[str] = extract_data.get("questions", [])

    # fallback لو فشل الاستخراج
    if not questions:
        questions = [text]

    return {
        "is_narrative": True,
        "original": text,
        "questions": questions,
    }