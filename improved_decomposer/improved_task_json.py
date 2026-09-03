import json
import re


try:
    import json_repair
except ImportError:  # pragma: no cover - json fallback is enough for tests
    json_repair = None


def strip_json_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return stripped


def parse_json_list(text: str) -> list[str]:
    text = strip_json_markdown_fence(text)
    if json_repair is not None:
        value = json_repair.loads(text)
    else:
        value = json.loads(text)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("model output is not a JSON list of strings")
    return value


def parse_json_list_or_object(text: str):
    text = strip_json_markdown_fence(text)
    if json_repair is not None:
        return json_repair.loads(text)
    return json.loads(text)
