"""Run NYUCTF/CTF prompts and save reusable trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nyuctf_bridge_runner import add_common_args, read_dataset, run_agent_for_item, selected_items, item_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NYUCTF data with Codex/Claude Code inside a CTF Docker container.")
    add_common_args(parser)
    parser.add_argument("--output", type=Path, default=Path("results/nyuctf/nyuctf_run_trajectories.jsonl"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = read_dataset(args.dataset)
    items = selected_items(data, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as out:
        for index, item in enumerate(items, 1):
            print(f"[{index}/{len(items)}] running item id={item_id(item)}")
            trajectory = run_agent_for_item(item, args)
            out.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            out.flush()
            print(
                "  actions="
                f"{len(trajectory.get('actions') or [])} "
                f"rc={trajectory['model_output']['returncode']} "
                f"gold={trajectory['verification']['ok']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
