#!/usr/bin/env python3
"""Merge generated X-Teaming attack-plan files into one reusable library."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def numeric_suffix(value: Any) -> int:
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else 0


def behavior_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    behavior_id = str((record.get("behavior_details") or {}).get("BehaviorID") or "")
    match = re.match(r"^(\d+):subtask:(\d+)$", behavior_id)
    if match:
        return int(match.group(1)), int(match.group(2)), behavior_id
    return 10**9, 10**9, behavior_id


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [
        record
        for record in data
        if isinstance(record, dict)
        and (record.get("behavior_details") or {}).get("BehaviorID") is not None
    ]


def total_strategy_count(record: dict[str, Any]) -> int:
    strategies = record.get("attack_strategies") or {}
    if not isinstance(strategies, dict):
        return 0
    return sum(len(value or {}) for value in strategies.values() if isinstance(value, dict))


def merge_strategy_set(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    max_strategies: int | None,
) -> dict[str, Any]:
    merged = dict(existing)
    used_values = list(merged.values())
    for _, value in sorted(incoming.items(), key=lambda item: numeric_suffix(item[0])):
        if max_strategies is not None and len(merged) >= max_strategies:
            break
        if not isinstance(value, dict):
            continue
        if value in used_values:
            continue
        key = f"strategy_{len(merged) + 1}"
        merged[key] = value
        used_values.append(value)
    return merged


def merge_record(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    max_strategies_per_set: int | None,
) -> dict[str, Any]:
    if total_strategy_count(incoming) > total_strategy_count(existing):
        merged = {
            "behavior_number": incoming.get("behavior_number", existing.get("behavior_number")),
            "behavior_details": incoming.get("behavior_details") or existing.get("behavior_details"),
            "attack_strategies": {},
        }
        base_strategies = incoming.get("attack_strategies") or {}
        other_strategies = existing.get("attack_strategies") or {}
    else:
        merged = {
            "behavior_number": existing.get("behavior_number", incoming.get("behavior_number")),
            "behavior_details": existing.get("behavior_details") or incoming.get("behavior_details"),
            "attack_strategies": {},
        }
        base_strategies = existing.get("attack_strategies") or {}
        other_strategies = incoming.get("attack_strategies") or {}

    set_names = {
        key
        for strategies in (base_strategies, other_strategies)
        if isinstance(strategies, dict)
        for key in strategies
    }
    for set_name in sorted(set_names, key=numeric_suffix):
        base_set = base_strategies.get(set_name) if isinstance(base_strategies, dict) else None
        other_set = other_strategies.get(set_name) if isinstance(other_strategies, dict) else None
        if not isinstance(base_set, dict):
            base_set = {}
        if not isinstance(other_set, dict):
            other_set = {}
        merged["attack_strategies"][set_name] = merge_strategy_set(
            base_set,
            other_set,
            max_strategies_per_set,
        )
    return merged


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        behavior_id = str((record.get("behavior_details") or {}).get("BehaviorID"))
        strategies = record.get("attack_strategies") or {}
        counts = {
            set_name: len(set_value or {})
            for set_name, set_value in sorted(strategies.items(), key=lambda item: numeric_suffix(item[0]))
            if isinstance(set_value, dict)
        }
        rows.append(
            {
                "behavior_id": behavior_id,
                "total_strategies": sum(counts.values()),
                "sets": counts,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("xteaming_vrap_codex_outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--max-strategies-per-set", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    by_behavior_id: dict[str, dict[str, Any]] = {}
    sources = 0
    for path in sorted(args.root.glob("**/attack_plans*.json")):
        if path.resolve() == args.output.resolve():
            continue
        records = load_records(path)
        if not records:
            continue
        sources += 1
        for record in records:
            behavior_id = str((record.get("behavior_details") or {}).get("BehaviorID"))
            if behavior_id not in by_behavior_id:
                by_behavior_id[behavior_id] = record
            else:
                by_behavior_id[behavior_id] = merge_record(
                    by_behavior_id[behavior_id],
                    record,
                    args.max_strategies_per_set,
                )

    merged = sorted(by_behavior_id.values(), key=behavior_sort_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.summary:
        args.summary.write_text(
            json.dumps(summarize(merged), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"sources={sources} records={len(merged)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
