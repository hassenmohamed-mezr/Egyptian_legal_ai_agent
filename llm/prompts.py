from typing import Any, Dict, Sequence

LEGAL_SYSTEM_PROMPT = """
    أنت مساعد قانوني مصري متخصص في تحليل قانون العمل.

    القواعد:
    - اعتمد فقط على النصوص المقدمة.
    - مسموح بدمج أكثر من مادة قانونية للوصول للإجابة.
    - لا تقل "لا يوجد نص" إذا كانت هناك مواد جزئية مرتبطة.
    - إذا لم يوجد أي دعم إطلاقًا فقط عندها قل: لا يوجد نص صريح.
    - لا تخترع مواد.
    """

LEGAL_USER_PROMPT_TEMPLATE = """
    السؤال:
    {query}

    النصوص القانونية:
    {context}

    تعليمات:
    - استخرج كل المواد المرتبطة بالسؤال.
    - اربط بينها للوصول للحكم القانوني النهائي.
    - اشرح العلاقة بين المواد إذا لزم الأمر.
    - اذكر المواد المستخدمة فقط من النص.

    شكل الإجابة:
    1. الحكم القانوني
    2. المواد القانونية
    3. التفسير المختصر

    الإجابة:
"""

def build_legal_context(chunks, max_chars=2500):
    if not chunks:
        return "لا توجد نصوص."

    # فلترة + ترتيب قوي
    chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

    parts = []
    total = 0

    for c in chunks[:5]:  # ⛔ أهم تحسين: حد أقصى ثابت
        score = c.get("score", 0)
        if score < 0.3:
            continue

        text = c.get("text", "").strip()
        article_id = c.get("article_id", "?")

        block = f"[{article_id}] {text}\n"

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts) if parts else "لا يوجد نصوص كافية."