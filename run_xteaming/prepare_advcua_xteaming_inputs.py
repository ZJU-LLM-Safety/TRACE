"""Prepare benchmark subtasks as X-Teaming behavior-generation inputs.

The upstream X-Teaming plan generator expects a HarmBench-like CSV containing
Behavior, BehaviorID, and FunctionalCategory.  This helper converts selected
AdvCUA or NYUCTF subtasks into that shape and writes a config copy that points
the plan generator at the generated CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def parse_pairs(value: str | None) -> list[tuple[str, int]]:
    if not value:
        return []
    pairs: list[tuple[str, int]] = []
    for raw_pair in value.split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        item_id, subtask_index = raw_pair.split(":", 1)
        pairs.append((item_id, int(subtask_index)))
    return pairs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_item_id(item: dict[str, Any], fallback_index: int) -> Any:
    return item.get(
        "id",
        item.get("unique_id", item.get("challenge", fallback_index)),
    )


def item_keys(item: dict[str, Any], fallback_index: int) -> Iterable[str]:
    values = [
        fallback_index,
        source_item_id(item, fallback_index),
        item.get("id"),
        item.get("unique_id"),
        item.get("challenge"),
        item.get("path"),
    ]
    for value in values:
        if value is not None:
            yield str(value)


def all_subtask_pairs(dataset: list[dict[str, Any]]) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for index, item in enumerate(dataset):
        subtasks = item.get("sub_prompts") or []
        if not subtasks:
            pairs.append((str(source_item_id(item, index)), 0))
            continue
        pairs.extend(
            (str(source_item_id(item, index)), subtask_index)
            for subtask_index in range(len(subtasks))
        )
    return pairs


def selected_rows(
    dataset: list[dict[str, Any]],
    pairs: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    indexed_items = list(enumerate(dataset))
    items_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, item in indexed_items:
        for key in item_keys(item, index):
            items_by_id[key] = (index, item)

    if not pairs:
        pairs = all_subtask_pairs(dataset)

    rows: list[dict[str, Any]] = []
    for item_id, subtask_index in pairs:
        if item_id not in items_by_id:
            raise KeyError(f"Dataset item {item_id!r} not found")
        item_index, item = items_by_id[item_id]
        item_source_id = source_item_id(item, item_index)
        subtasks = item.get("sub_prompts") or []
        if subtasks:
            if subtask_index < 0 or subtask_index >= len(subtasks):
                raise IndexError(
                    f"item {item_id!r} has {len(subtasks)} subtasks, "
                    f"cannot select index {subtask_index}"
                )
            subtask = subtasks[subtask_index]
        elif subtask_index == 0:
            subtask = (
                item.get("harmful")
                or item.get("query")
                or item.get("prompt_to_os_agent")
            )
        else:
            raise IndexError(f"item {item_id!r} has no subtasks, only index 0 is valid")
        rows.append(
            {
                "Behavior": subtask,
                "BehaviorID": f"{item_source_id}:subtask:{subtask_index}",
                "FunctionalCategory": "standard",
                "SourceItemID": item_source_id,
                "SubtaskIndex": subtask_index,
                "FullPrompt": item.get("prompt_to_os_agent")
                or item.get("harmful")
                or item.get("query", ""),
                "AdvCUAItemID": item.get("id", item.get("unique_id", "")),
                "AdvCUASubtaskIndex": subtask_index,
                "AdvCUAPromptToOSAgent": item.get("prompt_to_os_agent", ""),
                "CTFItemIndex": item_index,
                "CTFChallenge": item.get("challenge", ""),
                "CTFCategory": item.get("category", ""),
                "CTFPath": item.get("path", ""),
                "CTFEvaluationType": item.get("evaluation_type", ""),
            }
        )
    return rows


def write_behaviors_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No behavior rows selected")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_config(
    *,
    source_config: Path,
    output_config: Path,
    behavior_csv: Path,
    strategies_dir: Path,
    num_behaviors: int,
) -> None:
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["attack_plan_generator"]["behavior_path"] = str(behavior_csv)
    config["attack_plan_generator"]["attack_plan_generation_dir"] = str(strategies_dir)
    config["attack_plan_generator"]["num_behaviors"] = num_behaviors
    output_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert AdvCUA or NYUCTF subtasks into X-Teaming behavior inputs."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/advcua_test.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--pairs",
        help=(
            "Comma-separated item:subtask selections. AdvCUA items use their id; "
            "NYUCTF items can use zero-based jsonl row index, challenge name, or path. "
            "If omitted, all subtasks are selected."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("xteaming_vrap_codex_outputs/advcua_xteaming_inputs"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = selected_rows(read_jsonl(args.dataset), parse_pairs(args.pairs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_csv = args.output_dir / "behaviors.csv"
    config_path = args.output_dir / "config.yaml"
    strategies_dir = args.output_dir / "strategies"
    write_behaviors_csv(behavior_csv, rows)
    write_config(
        source_config=args.config,
        output_config=config_path,
        behavior_csv=behavior_csv,
        strategies_dir=strategies_dir,
        num_behaviors=len(rows),
    )
    print(f"Wrote {len(rows)} behavior row(s) to {behavior_csv}", flush=True)
    print(f"Wrote X-Teaming config to {config_path}", flush=True)
    print(
        "Next: conda run -n x-teaming python generate_attack_plans_compat.py "
        f"-c {config_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
