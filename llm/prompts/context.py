from typing import Any, Dict, Sequence


def build_legal_context(
    chunks: Sequence[Dict[str, Any]],
    max_chars: int = 5000,
) -> str:

    if not chunks:
        return "لا توجد نصوص قانونية متاحة."

    chunks = sorted(
        chunks,
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    parts = []
    total = 0
    seen_articles = set()

    for c in chunks:

        score = c.get("score", 0)

        if score < 0.25:
            continue

        article_id = c.get("article_id", "?")

        if article_id in seen_articles:
            continue

        seen_articles.add(article_id)

        title = c.get("article_title", "")
        text = c.get("text", "").strip()

        block = (
            f"{title}\n"
            f"{text}\n\n"
        )

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    if not parts:
        return "لا توجد نصوص قانونية كافية."

    return "\n".join(parts)
