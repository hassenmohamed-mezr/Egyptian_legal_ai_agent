# query_rewriter.py

import json
import os
import re
import time
from typing import Any, Dict, List

import requests


_DETECT_SYSTEM = """
أنت محلل ذكي لرسائل المستخدمين في الأنظمة القانونية.

مهمتك:
تحديد هل النص المكتوب:
1. سؤال قانوني مباشر
أم
2. قصة / شكوى / سيناريو / مشكلة واقعية تحتاج استخراج أسئلة قانونية منها.

قواعد صارمة:
- أجب بـ JSON فقط
- ممنوع أي شرح إضافي
- إذا احتوى النص على أحداث متعددة أو تفاصيل زمنية أو أشخاص أو تحقيقات أو مشاكل عملية فاعتبره narrative
""".strip()


_DETECT_USER = """
النص:

{text}

أجب بصيغة JSON فقط:

{{
  "is_narrative": true,
  "reason": "سبب مختصر"
}}
""".strip()


_EXTRACT_SYSTEM = """
أنت محلل قانوني متخصص في قانون العمل المصري.

مهمتك:
تحويل القصة أو السيناريو إلى أكبر عدد ممكن من الأسئلة القانونية الدقيقة والمهمة التي يجب البحث عنها.

قواعد صارمة:
1. استخرج جميع المشاكل القانونية الموجودة ضمنياً وصراحة.
2. لا تكتف بالسؤال المباشر فقط.
3. استخرج الأسئلة المتعلقة بـ:
   - التحقيق
   - الفصل
   - الإيقاف
   - الأجور
   - الخصومات
   - النقل
   - التعويض
   - الإنذارات
   - الإجراءات
   - مدد التظلم
   - حقوق العامل
   - صلاحيات صاحب العمل
4. لا تكرر نفس الفكرة بصياغات مختلفة.
5. الأسئلة يجب أن تكون قصيرة وواضحة وقابلة للبحث.
6. استخرج من 6 إلى 15 سؤالاً حسب تعقيد القصة.
7. أجب بـ JSON فقط.
""".strip()


_EXTRACT_USER = """
القصة أو السيناريو:

{narrative}

استخرج جميع الأسئلة القانونية المهمة المتعلقة بهذه القصة.

أجب بصيغة JSON فقط:

{{
  "questions": [
    "سؤال 1",
    "سؤال 2",
    "سؤال 3"
  ]
}}
""".strip()


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


def _parse_json_safe(text: str) -> Dict[str, Any]:

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text
    )

    cleaned = cleaned.replace(
        "```",
        ""
    ).strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            cleaned,
            re.DOTALL
        )

        if match:

            try:
                return json.loads(match.group())

            except json.JSONDecodeError:
                pass

    return {}


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