"""Prepare AdvCUA subtasks as X-Teaming behavior-generation inputs.

The upstream X-Teaming plan generator expects a HarmBench-like CSV containing
Behavior, BehaviorID, and FunctionalCategory.  This helper converts selected
``data/advcua_test.jsonl`` subtasks into that shape and writes a config copy
that points ``generate_attack_plans.py`` at the generated CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def parse_pairs(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw_pair in value.split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        item_id, subtask_index = raw_pair.split(":", 1)
        pairs.append((int(item_id), int(subtask_index)))
    return pairs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def selected_rows(
    dataset: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    items_by_id = {
        item.get("id", item.get("unique_id")): item
        for item in dataset
    }
    rows: list[dict[str, Any]] = []
    for item_id, subtask_index in pairs:
        item = items_by_id[item_id]
        subtask = item["sub_prompts"][subtask_index]
        rows.append(
            {
                "Behavior": subtask,
                "BehaviorID": f"{item_id}:subtask:{subtask_index}",
                "FunctionalCategory": "standard",
                "AdvCUAItemID": item_id,
                "AdvCUASubtaskIndex": subtask_index,
                "AdvCUAPromptToOSAgent": item.get("prompt_to_os_agent", ""),
            }
        )
    return rows


def write_behaviors_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        description="Convert AdvCUA subtasks into X-Teaming behavior inputs."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/advcua_test.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--pairs", required=True)
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
        "Next: conda run -n x-teaming python generate_attack_plans.py "
        f"-c {config_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
