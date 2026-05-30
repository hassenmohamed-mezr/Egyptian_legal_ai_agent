import json
import re
from typing import Any, Dict


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
