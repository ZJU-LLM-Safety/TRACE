"""Method-agnostic result shaping, redaction, and environment helpers."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .bridge import DEFAULT_OUTPUT_ROOT


def redact_secrets(text: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", text)


def coerce_score(value: int | None) -> int:
    return value if isinstance(value, int) else 0


def with_strategy_meta(records: list[dict[str, Any]], strategy_meta: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for record in records:
        item = dict(record)
        item["strategy_meta"] = strategy_meta
        annotated.append(item)
    return annotated


def clean_input_output(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "input": record.get("input", ""),
            "output": record.get("output", []),
        }
        for record in records
    ]


def current_turn_actions(target: Any) -> list[dict[str, Any]]:
    if not getattr(target, "turn_results", None):
        return []
    turn_number = target.turn_results[-1]["turn"]
    return [action for action in target.actions if action.get("turn") == turn_number]


def rewrite_turn_numbers(records: list[dict[str, Any]], turn_number: int) -> list[dict[str, Any]]:
    rewritten = []
    for record in records:
        item = dict(record)
        item["turn"] = turn_number
        rewritten.append(item)
    return rewritten


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_output_path(output_dir: Path | None) -> Path:
    run_dir = output_dir or DEFAULT_OUTPUT_ROOT / timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "results.jsonl"


def write_json(path: Path, payload: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def apply_codex_env_defaults(args: argparse.Namespace) -> None:
    os.environ.pop("CODEX_HOME", None)
    if getattr(args, "responses_base_url", None):
        os.environ["OPENAI_BASE_URL"] = args.responses_base_url
    responses_env_key = getattr(args, "responses_env_key", "OPENAI_API_KEY")
    if responses_env_key != "OPENAI_API_KEY" and os.environ.get(responses_env_key):
        os.environ.setdefault("OPENAI_API_KEY", os.environ[responses_env_key])
    if getattr(args, "clean_proxy_env", False):
        _extend_no_proxy("localhost,127.0.0.1,::1,192.168.2.1")
    if getattr(args, "placeholder_api_key", False) and not os.environ.get(responses_env_key):
        os.environ[responses_env_key] = "placeholder"


def _extend_no_proxy(extra: str) -> None:
    for key in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(key, "")
        entries = [part.strip() for part in current.split(",") if part.strip()]
        for part in extra.split(","):
            part = part.strip()
            if part and part not in entries:
                entries.append(part)
        os.environ[key] = ",".join(entries)
