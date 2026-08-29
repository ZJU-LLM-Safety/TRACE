"""Prepare AdvCUA/CTF subtasks as experiment behavior-generation inputs.

The experiment plan generator consumes a HarmBench-like CSV containing
Behavior, BehaviorID, and FunctionalCategory. This helper converts selected
AdvCUA or CTF JSON/JSONL subtasks into that shape and writes a config copy that
points the plan generator at the generated CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def read_dataset(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("data", "items", "records"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
    raise ValueError(f"Unsupported dataset shape in {path}")


def source_item_id(item: dict[str, Any], fallback_index: int) -> Any:
    return item.get(
        "id",
        item.get("unique_id", item.get("challenge", fallback_index)),
    )


def selected_rows(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_index, item in enumerate(dataset):
        item_source_id = source_item_id(item, item_index)
        subtasks = item.get("sub_prompts") or []
        if subtasks:
            subtask_rows = [
                (subtask_index, str(subtask))
                for subtask_index, subtask in enumerate(subtasks)
            ]
        else:
            subtask_rows = [
                (
                    0,
                    str(
                        item.get("harmful")
                        or item.get("query")
                        or item.get("prompt_to_os_agent")
                        or item.get("prompt")
                        or ""
                    ),
                )
            ]
        for subtask_index, subtask in subtask_rows:
            rows.append(
                {
                    "Behavior": subtask,
                    "BehaviorID": f"{item_source_id}:subtask:{subtask_index}",
                    "FunctionalCategory": "standard",
                    "SourceItemID": item_source_id,
                    "SubtaskIndex": subtask_index,
                    "FullPrompt": item.get("prompt_to_os_agent")
                    or item.get("harmful")
                    or item.get("query")
                    or item.get("prompt", ""),
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
        description="Convert AdvCUA or CTF subtasks into experiment behavior inputs."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "config" / "config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_outputs/experiment_inputs"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = selected_rows(read_dataset(args.dataset))
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
    print(f"Wrote experiment config to {config_path}", flush=True)
    print(f"Next online run: python main.py -c {config_path}", flush=True)
    print(f"Optional batch plan generation: python generate_attack_plans_compat.py -c {config_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
