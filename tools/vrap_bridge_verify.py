"""Replay VRAP trajectories and verify the reproduced environment state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vrap_bridge_runner import (
    DEFAULT_COMPOSE,
    cleanup_environment,
    prepare_compose,
    setup_environment,
    verify_item,
)
from vrap_action_replay import action_to_command, action_tool, replay_trajectory_actions


def read_trajectories(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else [parsed]


def replay_actions(
    trajectory: dict[str, Any],
    *,
    timeout: int,
) -> list[dict[str, Any]]:
    return replay_trajectory_actions(
        trajectory,
        timeout=timeout,
        tolerant_timeouts=False,
    )


def verify_trajectory(trajectory: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    item = trajectory["item"]
    run = trajectory.get("run") or {}
    attacker_identity = str(run.get("attacker_identity") or item.get("attacker_identity") or "root")
    temp_compose = prepare_compose(args.compose_file, attacker_identity)
    try:
        setup_environment(item, temp_compose, attacker_identity)
        replayed_actions = replay_actions(trajectory, timeout=args.replay_timeout)

        replay_history = json.dumps({
            "model_output": trajectory.get("model_output"),
            "original_actions": trajectory.get("actions") or [],
            "replayed_actions": replayed_actions,
        }, ensure_ascii=False)
        verification = verify_item(item, replay_history, attacker_identity)

        out = dict(trajectory)
        out["replayed_actions"] = replayed_actions
        out["verification"] = verification
        out["replay"]["verification_replayed"] = True
        out["replay"]["steps"] = [
            *(out.get("setup") or []),
            *replayed_actions,
            *[
                {
                    "type": check.get("type"),
                    "instruction": check.get("instruction"),
                    "result": check.get("result"),
                    "ok": check.get("ok"),
                    "exit_code": check.get("exit_code"),
                }
                for check in verification.get("checks", [])
                if check.get("instruction") is not None
            ],
        ]
        return out
    finally:
        cleanup_environment(temp_compose)
        temp_compose.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay VRAP trajectories and run normal VRAP verification.")
    parser.add_argument("--input", type=Path, required=True, help="Trajectory JSON or JSONL from vrap_bridge_run.py.")
    parser.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE)
    parser.add_argument("--ids", help="Comma-separated trajectory item ids to verify.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--replay-timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("vrap_verified_trajectories.jsonl"))
    return parser


def selected_trajectories(trajectories: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ids = {int(x) for x in args.ids.split(",")} if args.ids else None
    selected: list[dict[str, Any]] = []
    for trajectory in trajectories:
        item = trajectory.get("item") or {}
        item_id = item.get("id", item.get("unique_id"))
        if ids is not None and item_id not in ids:
            continue
        selected.append(trajectory)
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def main() -> int:
    args = build_parser().parse_args()
    trajectories = selected_trajectories(read_trajectories(args.input), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as out:
        for index, trajectory in enumerate(trajectories, 1):
            item = trajectory.get("item") or {}
            item_id = item.get("id", item.get("unique_id"))
            print(f"[{index}/{len(trajectories)}] verifying item id={item_id}")
            verified = verify_trajectory(trajectory, args)
            out.write(json.dumps(verified, ensure_ascii=False) + "\n")
            out.flush()
            print(
                "  verification="
                f"{verified['verification']['ok']} "
                f"replayed={len(verified.get('replayed_actions') or [])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
