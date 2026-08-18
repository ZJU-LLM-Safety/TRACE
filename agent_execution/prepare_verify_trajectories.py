#!/usr/bin/env python3
"""Convert same-session Codex execution results into VRAP verify trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_actions(result: dict[str, Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for subtask_number, actions in enumerate(result.get("actions") or [], 1):
        for action in actions or []:
            copied = dict(action)
            copied.setdefault("subtask_number", subtask_number)
            flat.append(copied)
    return flat


def build_trajectory(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    item = result.get("item") or {}
    attacker_identity = str(item.get("attacker_identity") or "root")
    final_output = ""
    input_output = result.get("input_output") or []
    if input_output:
        final_output = str(input_output[-1].get("output") or "")
    return {
        "item": item,
        "run": {
            "attacker_identity": attacker_identity,
            "agent_user": attacker_identity,
            "source_result": str(result_path),
            "mode": result.get("mode"),
            "model": result.get("model"),
        },
        "model_output": {"text": final_output},
        "input_output": input_output,
        "actions": flatten_actions(result),
        "replay": {
            "dataset_item_id": item.get("id", item.get("unique_id")),
            "source_result": str(result_path),
            "steps": [],
        },
    }


def iter_result_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("item_*/result.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = iter_result_paths(args.input)
    if not paths:
        raise FileNotFoundError(f"no result.json files found under {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for path in paths:
            out.write(json.dumps(build_trajectory(read_result(path), path), ensure_ascii=False) + "\n")
    print(f"wrote {len(paths)} trajectories to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
