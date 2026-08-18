"""Run multi-turn VRAP conversations with one persistent Docker environment.

This keeps the existing single-item runner intact. The conversation runner
starts VRAP once per item, sends multiple prompts to Codex or Claude Code inside
the same `my_ssh_client` container, and records each turn in `input_output`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vrap_bridge_runner import (
    CLIENT_CONTAINER,
    DEFAULT_COMPOSE,
    DEFAULT_CLAUDE_HOME,
    DEFAULT_DATASET,
    build_input_output_pairs,
    build_prompt,
    cleanup_environment,
    prepare_compose,
    read_dataset,
    run_claude_in_vrap,
    run_codex_in_vrap,
    selected_items,
    setup_environment,
    verify_item,
)
from agent_cli_bridge import clean_codex_noise


def normalize_turns(raw: Any, item: dict[str, Any]) -> list[str]:
    if raw is None:
        return [build_prompt(item)]
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, list):
        raise ValueError("turns must be a string or a list")

    turns: list[str] = []
    for turn in raw:
        if isinstance(turn, str):
            turns.append(turn)
            continue
        if isinstance(turn, dict):
            value = turn.get("input") or turn.get("prompt") or turn.get("content")
            if isinstance(value, str):
                turns.append(value)
                continue
        raise ValueError(f"unsupported turn shape: {turn!r}")
    return turns


def read_turns(args: argparse.Namespace, item: dict[str, Any]) -> list[str]:
    if args.turns_json:
        return normalize_turns(json.loads(args.turns_json), item)
    if args.turns_file:
        text = args.turns_file.read_text(encoding="utf-8")
        if args.turns_file.suffix == ".jsonl":
            return normalize_turns(
                [json.loads(line) for line in text.splitlines() if line.strip()],
                item,
            )
        return normalize_turns(json.loads(text), item)
    return normalize_turns(None, item)


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def run_turn(
    prompt: str,
    *,
    args: argparse.Namespace,
    agent_user: str,
    resume_id: str | None,
) -> dict[str, Any]:
    should_resume = args.agent_session == "resume" and bool(resume_id)
    if args.backend == "codex":
        return run_codex_in_vrap(
            prompt,
            model=args.model,
            user=agent_user,
            timeout=args.timeout,
            bypass=args.bypass,
            write_config=args.codex_write_config,
            responses_base_url=args.responses_base_url,
            responses_env_key=args.responses_env_key,
            resume_thread_id=resume_id if should_resume else None,
        )

    return run_claude_in_vrap(
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
        resume_session_id=resume_id if should_resume else None,
        fork_session=args.fork_session and should_resume,
    )


def run_conversation_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    attacker_identity = str(item.get("attacker_identity") or "root")
    agent_user = args.agent_user or attacker_identity
    turns = read_turns(args, item)
    temp_compose = prepare_compose(args.compose_file, attacker_identity)

    input_output: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []
    turn_results: list[dict[str, Any]] = []
    resume_id: str | None = None

    try:
        setup_environment(item, temp_compose, attacker_identity)
        for index, prompt in enumerate(turns, 1):
            result = run_turn(
                prompt,
                args=args,
                agent_user=agent_user,
                resume_id=resume_id,
            )
            if args.backend == "codex":
                result, _removed = clean_codex_noise(result)
            next_resume_id = result.get("thread_id") or resume_id
            actions = []
            for action_index, action in enumerate(result.get("actions") or [], 1):
                turn_action = dict(action)
                turn_action["turn"] = index
                turn_action["turn_action_index"] = action_index
                actions.append(turn_action)
                all_actions.append(turn_action)

            pair = build_input_output_pairs(
                prompt,
                str(result.get("text") or ""),
                actions,
            )[0]
            input_output.append(pair)
            turn_results.append({
                "turn": index,
                "returncode": result.get("returncode"),
                "error": result.get("error"),
                "session_id_before": resume_id,
                "session_id_after": next_resume_id,
                "action_count": len(actions),
            })
            resume_id = next_resume_id

        history = json.dumps(
            {"input_output": input_output, "actions": all_actions},
            ensure_ascii=False,
        )
        verification = (
            verify_item(item, history, attacker_identity)
            if args.verify
            else {"status": "pending", "ok": None, "checks": []}
        )
        return {
            "item": item,
            "run": {
                "backend": args.backend,
                "model": args.model,
                "attacker_identity": attacker_identity,
                "agent_user": agent_user,
                "container": CLIENT_CONTAINER,
                "mode": "conversation",
                "agent_session": args.agent_session,
            },
            "prompt_to_model": turns[-1] if turns else "",
            "input_output": input_output,
            "turns": turn_results,
            "model_output": {
                "text": input_output[-1]["output"][0]["content"]
                if input_output and input_output[-1]["output"]
                and input_output[-1]["output"][0].get("type") == "text"
                else "",
                "returncode": turn_results[-1]["returncode"] if turn_results else None,
                "error": turn_results[-1]["error"] if turn_results else None,
            },
            "actions": all_actions,
            "verification": verification,
            "replay": {
                "compose_file": str(DEFAULT_COMPOSE),
                "dataset_item_id": item.get("id", item.get("unique_id")),
                "steps": all_actions,
            },
        }
    finally:
        if not args.keep_container:
            cleanup_environment(temp_compose)
            temp_compose.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-turn VRAP conversations.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE)
    parser.add_argument("--backend", choices=("codex", "claude"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--ids", help="Comma-separated item ids to run.")
    parser.add_argument("--start-from", type=int)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--turns-file", type=Path, help="JSON/JSONL turns: strings or objects with input/prompt/content.")
    parser.add_argument("--turns-json", help="Inline JSON turns: string or list of strings/objects.")
    parser.add_argument("--agent-session", choices=("none", "resume"), default="resume")
    parser.add_argument("--fork-session", action="store_true", help="Claude only: fork when resuming instead of continuing the same session.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--agent-user", help="Override runtime user for Codex/Claude inside my_ssh_client.")
    parser.add_argument("--allowed-tools", help="Comma-separated Claude Code tools/rules to auto-approve, e.g. Bash,Write,Read.")
    parser.add_argument("--claude-home", default=DEFAULT_CLAUDE_HOME)
    parser.add_argument("--bypass", action="store_true")
    parser.add_argument("--codex-write-config", action="store_true")
    parser.add_argument("--responses-base-url")
    parser.add_argument("--responses-env-key", default="OPENAI_API_KEY")
    parser.add_argument("--anthropic-base-url")
    parser.add_argument("--anthropic-env-key", default="ANTHROPIC_API_KEY")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("vrap_conversation_trajectories.jsonl"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = read_dataset(args.dataset)
    items = selected_items(data, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as out:
        for index, item in enumerate(items, 1):
            item_id = item.get("id", item.get("unique_id"))
            print(f"[{index}/{len(items)}] running conversation item id={item_id}")
            trajectory = run_conversation_item(item, args)
            out.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            out.flush()
            print(
                "  turns="
                f"{len(trajectory.get('input_output') or [])} "
                f"actions={len(trajectory.get('actions') or [])} "
                f"verification={trajectory['verification'].get('ok')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
