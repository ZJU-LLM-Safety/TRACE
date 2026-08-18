"""Replay NYUCTF/CTF trajectories and verify the reproduced outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nyuctf_bridge_runner import (
    DEFAULT_IMAGE,
    DEFAULT_NYUCTF_ROOT,
    cleanup_environment,
    item_id,
    replay_trajectory_actions,
    setup_environment,
    verify_gold,
)


def read_trajectories(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else [parsed]


def selected_trajectories(trajectories: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ids = {part.strip() for part in args.ids.split(",")} if args.ids else None
    selected: list[dict[str, Any]] = []
    for trajectory in trajectories:
        item = trajectory.get("item") or {}
        if ids is not None and item_id(item) not in ids:
            continue
        selected.append(trajectory)
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def verify_trajectory(trajectory: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    item = trajectory["item"]
    run = trajectory.get("run") or {}
    env_info = setup_environment(
        item,
        nyuctf_root=args.nyuctf_root,
        image_name=args.image_name or run.get("image_name") or DEFAULT_IMAGE,
        cpus=args.cpus,
        memory=args.memory,
        container_name=args.container_name,
    )
    env_info["image_name"] = args.image_name or run.get("image_name") or DEFAULT_IMAGE
    try:
        replayed_actions = replay_trajectory_actions(
            trajectory,
            container=env_info["container"],
            workdir=env_info["workdir"],
            timeout=args.replay_timeout,
        )
        verification = verify_gold(
            item,
            {
                "model_output": trajectory.get("model_output"),
                "original_actions": trajectory.get("actions") or [],
                "replayed_actions": replayed_actions,
            },
        )
        out = dict(trajectory)
        out["replayed_actions"] = replayed_actions
        out["verification"] = verification
        out["replay"] = dict(out.get("replay") or {})
        out["replay"]["verification_replayed"] = True
        out["replay"]["steps"] = [
            *(env_info.get("setup_steps") or []),
            *replayed_actions,
            *[
                {
                    "type": check.get("type"),
                    "result": check.get("needle"),
                    "ok": check.get("ok"),
                }
                for check in verification.get("checks", [])
            ],
        ]
        return out
    finally:
        cleanup_environment(env_info)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay NYUCTF trajectories and verify gold output.")
    parser.add_argument("--input", type=Path, required=True, help="Trajectory JSON or JSONL from nyuctf_bridge_run.py.")
    parser.add_argument("--output", type=Path, default=Path("results/nyuctf/nyuctf_verified_trajectories.jsonl"))
    parser.add_argument("--nyuctf-root", type=Path, default=DEFAULT_NYUCTF_ROOT)
    parser.add_argument("--image-name", default=DEFAULT_IMAGE)
    parser.add_argument("--ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--replay-timeout", type=int, default=300)
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument("--memory", default="4g")
    parser.add_argument("--container-name")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trajectories = selected_trajectories(read_trajectories(args.input), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as out:
        for index, trajectory in enumerate(trajectories, 1):
            item = trajectory.get("item") or {}
            print(f"[{index}/{len(trajectories)}] verifying item id={item_id(item)}")
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
