"""Run VRAP prompts with Codex/Claude Code and save reusable trajectories.

This file only performs the model run. It starts the VRAP Docker environment,
sends the dataset prompt to the selected agent inside `my_ssh_client`, records
the model text plus any captured actions, then tears the environment down.

Use `vrap_bridge_verify.py` to replay and verify the saved trajectories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vrap_bridge_runner import (
    DEFAULT_COMPOSE,
    DEFAULT_CLAUDE_HOME,
    DEFAULT_DATASET,
    build_prompt,
    build_trajectory,
    cleanup_environment,
    prepare_compose,
    read_dataset,
    run_claude_in_vrap,
    run_codex_in_vrap,
    selected_items,
    setup_environment,
)


def run_item_only(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    attacker_identity = str(item.get("attacker_identity") or "root")
    agent_user = args.agent_user or attacker_identity
    temp_compose = prepare_compose(args.compose_file, attacker_identity)
    try:
        setup_environment(item, temp_compose, attacker_identity)
        prompt = build_prompt(item)
        if args.dry_run:
            agent_result = {
                "backend": args.backend,
                "returncode": 0,
                "text": "",
                "error": None,
                "actions": [],
                "dry_run_prompt": prompt,
            }
        elif args.backend == "codex":
            agent_result = run_codex_in_vrap(
                prompt,
                model=args.model,
                user=agent_user,
                timeout=args.timeout,
                bypass=args.bypass,
                write_config=args.codex_write_config,
                responses_base_url=args.responses_base_url,
                responses_env_key=args.responses_env_key,
            )
        else:
            agent_result = run_claude_in_vrap(
                prompt,
                model=args.model,
                user=agent_user,
                timeout=args.timeout,
                bypass=args.bypass,
                max_turns=args.max_turns,
                anthropic_base_url=args.anthropic_base_url,
                anthropic_env_key=args.anthropic_env_key,
                allowed_tools=parse_csv_arg(args.allowed_tools),
                claude_home=args.claude_home,
            )

        return build_trajectory(
            item=item,
            prompt=prompt,
            attacker_identity=attacker_identity,
            agent_user=agent_user,
            backend=args.backend,
            model=args.model,
            agent_result=agent_result,
            verification={"status": "pending", "ok": None, "checks": []},
        )
    finally:
        cleanup_environment(temp_compose)
        temp_compose.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VRAP prompts and save trajectories without verification.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE)
    parser.add_argument("--backend", choices=("codex", "claude"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--ids", help="Comma-separated item ids to run.")
    parser.add_argument("--start-from", type=int)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--agent-user", help="Override runtime user for Codex/Claude inside my_ssh_client.")
    parser.add_argument("--allowed-tools", help="Comma-separated Claude Code tools/rules to auto-approve, e.g. Bash,Write,Read.")
    parser.add_argument("--claude-home", default=DEFAULT_CLAUDE_HOME)
    parser.add_argument("--bypass", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--codex-write-config", action="store_true")
    parser.add_argument("--responses-base-url")
    parser.add_argument("--responses-env-key", default="OPENAI_API_KEY")
    parser.add_argument("--anthropic-base-url")
    parser.add_argument("--anthropic-env-key", default="ANTHROPIC_API_KEY")
    parser.add_argument("--output", type=Path, default=Path("vrap_run_trajectories.jsonl"))
    return parser


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    args = build_parser().parse_args()
    data = read_dataset(args.dataset)
    items = selected_items(data, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as out:
        for index, item in enumerate(items, 1):
            item_id = item.get("id", item.get("unique_id"))
            print(f"[{index}/{len(items)}] running item id={item_id}")
            trajectory = run_item_only(item, args)
            out.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            out.flush()
            print(
                "  actions="
                f"{len(trajectory.get('actions') or [])} "
                f"rc={trajectory['model_output']['returncode']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
