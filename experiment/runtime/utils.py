"""Small shared helpers for result shaping, metadata, and environment setup."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .bridge import DEFAULT_OUTPUT_ROOT

def redact_secrets(text: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", text)

def coerce_score(value: int | None) -> int:
    return value if isinstance(value, int) else 0


def with_strategy_meta(records: list[dict[str, Any]], strategy_meta: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for record in records:
        item = dict(record)
        item["strategy_meta"] = strategy_meta
        annotated.append(item)
    return annotated


def clean_input_output(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "input": record.get("input", ""),
            "output": record.get("output", []),
        }
        for record in records
    ]


def current_turn_actions(target: VrapCodexTarget) -> list[dict[str, Any]]:
    if not target.turn_results:
        return []
    turn_number = target.turn_results[-1]["turn"]
    return [action for action in target.actions if action.get("turn") == turn_number]


def rewrite_turn_numbers(records: list[dict[str, Any]], turn_number: int) -> list[dict[str, Any]]:
    rewritten = []
    for record in records:
        item = dict(record)
        item["turn"] = turn_number
        rewritten.append(item)
    return rewritten

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_output_path(output_dir: Path | None) -> Path:
    run_dir = output_dir or DEFAULT_OUTPUT_ROOT / timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "results.jsonl"


def write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    attacker_config: dict[str, Any],
    tg_config: dict[str, Any],
    eval_config: dict[str, Any],
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "config": str(args.config),
        "output": str(path),
        "attacker": {
            "backend": "transformers",
            "model": attacker_config["model"],
            "max_new_tokens": attacker_config["max_new_tokens"],
            "device_map": attacker_config["device_map"],
        },
        "target": {
            "backend": getattr(args, "target_backend", "codex"),
            "model": (
                args.claude_model
                if getattr(args, "target_backend", "codex") == "claude"
                else args.codex_model
            ),
            "responses_base_url": args.responses_base_url,
            "responses_env_key": args.responses_env_key,
            "anthropic_base_url": getattr(args, "anthropic_base_url", None),
            "anthropic_env_key": getattr(args, "anthropic_env_key", None),
            "claude_home": getattr(args, "claude_home", None),
            "claude_runner": getattr(args, "claude_runner", None),
            "codex_write_config": args.codex_write_config,
            "claude_max_turns": getattr(args, "claude_max_turns", None),
            "allowed_tools": getattr(args, "allowed_tools", None),
            "execution_mode": args.target_execution_mode,
            "post_subtask_execution_turn": getattr(args, "post_subtask_execution_turn", False),
            "post_subtask_execution_prompt": args.post_subtask_execution_prompt,
            "bypass": args.bypass,
        },
        "experiment_components": {
            "attack_plan_generator": "online strategy generation in experiment runner",
            "attack_plans_file": str(args.plans_file) if args.plans_file else None,
            "generated_strategy_required": not args.allow_default_strategy,
            "default_strategy_allowed": args.allow_default_strategy,
            "gpt_judge": args.use_gpt_judge,
            "gpt_judge_model": eval_config.get("judge_model", "gpt-4o") if args.use_gpt_judge else None,
            "textgrad": args.textgrad,
            "textgrad_backend": "transformers" if args.textgrad else None,
            "textgrad_model": tg_config.get("model") if args.textgrad else None,
            "isolated_textgrad": args.isolated_textgrad if args.textgrad else None,
            "textgrad_isolation_mode": args.textgrad_isolation_mode if args.textgrad else None,
            "textgrad_max_turns_per_phase": tg_config.get("max_turns_per_phase") if args.textgrad else None,
            "plan_revision": args.plan_revision,
            "strategies_per_behavior": args.strategies_per_behavior
            if args.strategies_per_behavior is not None
            else attacker_config.get("strategies_per_behavior"),
            "run_all_strategies": args.run_all_strategies,
            "strategy_success_mode": getattr(args, "strategy_success_mode", "judge"),
            "strategy_execution_output_mode": getattr(
                args, "strategy_execution_output_mode", None
            ),
            "strategy_execution_prompt_style": getattr(
                args, "strategy_execution_prompt_style", None
            ),
            "strategy_sample_seed": args.strategy_sample_seed,
            "single_strategy_override": (
                {"plan_strategy": args.plan_strategy}
                if args.plan_strategy is not None
                else None
            ),
        },
        "selection": {
            "ids": args.ids,
            "start_from": args.start_from,
            "subtask_index": args.subtask_index,
            "all_subtasks": args.all_subtasks,
        },
    }
    (path.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def apply_codex_env_defaults(args: argparse.Namespace) -> None:
    os.environ.pop("CODEX_HOME", None)
    if args.responses_base_url:
        os.environ["OPENAI_BASE_URL"] = args.responses_base_url
    if args.responses_env_key != "OPENAI_API_KEY" and os.environ.get(args.responses_env_key):
        os.environ.setdefault("OPENAI_API_KEY", os.environ[args.responses_env_key])
    if args.clean_proxy_env:
        _extend_no_proxy("localhost,127.0.0.1,::1,192.168.2.1")
    if args.placeholder_api_key and not os.environ.get(args.responses_env_key):
        os.environ[args.responses_env_key] = "placeholder"
    if getattr(args, "anthropic_base_url", None):
        os.environ["ANTHROPIC_BASE_URL"] = args.anthropic_base_url
    anthropic_env_key = getattr(args, "anthropic_env_key", "ANTHROPIC_API_KEY")
    if anthropic_env_key != "ANTHROPIC_API_KEY" and os.environ.get(anthropic_env_key):
        os.environ.setdefault("ANTHROPIC_API_KEY", os.environ[anthropic_env_key])


def _extend_no_proxy(extra: str) -> None:
    for key in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(key, "")
        entries = [part.strip() for part in current.split(",") if part.strip()]
        for part in extra.split(","):
            part = part.strip()
            if part and part not in entries:
                entries.append(part)
        os.environ[key] = ",".join(entries)
