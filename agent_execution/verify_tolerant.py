"""Replay VRAP trajectories with per-action timeout recording."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TRACE_ROOT = Path(__file__).resolve().parents[1]
TRACE_TOOLS = TRACE_ROOT / "tools"
if str(TRACE_TOOLS) not in sys.path:
    sys.path.insert(0, str(TRACE_TOOLS))

from vrap_bridge_runner import (  # noqa: E402
    DEFAULT_COMPOSE,
    cleanup_environment,
    prepare_compose,
    setup_environment,
    verify_item,
)
from vrap_bridge_verify import read_trajectories, selected_trajectories  # noqa: E402
from vrap_action_replay import replay_trajectory_actions  # noqa: E402


def replay_actions_tolerant(
    trajectory: dict[str, Any],
    *,
    timeout: int,
) -> list[dict[str, Any]]:
    return replay_trajectory_actions(
        trajectory,
        timeout=timeout,
        tolerant_timeouts=True,
    )


def verify_trajectory_tolerant(trajectory: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    item = trajectory["item"]
    run = trajectory.get("run") or {}
    attacker_identity = str(run.get("attacker_identity") or item.get("attacker_identity") or "root")
    temp_compose = prepare_compose(args.compose_file, attacker_identity)
    try:
        setup_environment(item, temp_compose, attacker_identity)
        replayed_actions = replay_actions_tolerant(trajectory, timeout=args.replay_timeout)
        replay_history = json.dumps(
            {
                "model_output": trajectory.get("model_output"),
                "original_actions": trajectory.get("actions") or [],
                "replayed_actions": replayed_actions,
            },
            ensure_ascii=False,
        )
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
    parser = argparse.ArgumentParser(description="Replay VRAP trajectories with timeout-tolerant action handling.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE)
    parser.add_argument("--ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--replay-timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trajectories = selected_trajectories(read_trajectories(args.input), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as out:
        for index, trajectory in enumerate(trajectories, 1):
            item = trajectory.get("item") or {}
            item_id = item.get("id", item.get("unique_id"))
            print(f"[{index}/{len(trajectories)}] verifying item id={item_id}")
            verified = verify_trajectory_tolerant(trajectory, args)
            out.write(json.dumps(verified, ensure_ascii=False) + "\n")
            out.flush()
            timeouts = sum(1 for action in verified.get("replayed_actions") or [] if action.get("timeout"))
            print(
                "  verification="
                f"{verified['verification']['ok']} "
                f"replayed={len(verified.get('replayed_actions') or [])} "
                f"timeouts={timeouts}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
