from typing import Any, Dict, Sequence

LEGAL_SYSTEM_PROMPT = (
    "أنت مساعد قانوني متخصص في القانون المصري. "
    "أجب بدقة باللغة العربية، واستند فقط إلى المعلومات الواردة في قسم \"المصادر القانونية\". "
    "إذا لم تُعثر على معلومات كافية للإجابة، أجب بالضبط: "
    "\"لا توجد معلومات كافية في النصوص القانونية المتاحة للإجابة على هذا السؤال.\" "
    "اذكر المواد والعناوين ذات الصلة بصيغة واضحة، مثل \"وفقاً للمادة 145 من قانون العمل...\". "
    "لا تستخدم أي معلومات خارج النصوص المقدمة."
)

LEGAL_USER_PROMPT_TEMPLATE = (
    "السؤال:\n{query}\n\n"
    "المصادر القانونية:\n{context}\n\n"
    "الإرشادات:\n"
    "- أجب بدقة وشمولية بناءً على النصوص المقدمة فقط.\n"
    "- أدرج المادة أو المواد المساندة بصيغة واضحة.\n"
    "- إذا لم تتوفر إجابة، أجب بالجملة التالية بالضبط:\n"
    "  لا توجد معلومات كافية في النصوص القانونية المتاحة للإجابة على هذا السؤال.\n"
    "- لا تذكر أي معلومات خارج السياق المرفق."
)


def build_legal_context(chunks: Sequence[Dict[str, Any]], max_chars: int = 8000) -> str:
    """Build a concise Arabic legal context block from retrieved chunks."""
    lines = []
    total_length = 0

    for chunk in chunks:
        article_id = chunk.get("article_id", "غير معروف")
        article_title = chunk.get("article_title", "بدون عنوان")
        text = str(chunk.get("text", "")).strip()
        block = f"المادة {article_id} - {article_title}\n{text}"
        block_length = len(block)

        if total_length + block_length > max_chars and lines:
            break

        lines.append(block)
        total_length += block_length

    return "\n\n".join(lines)