"""Subtask-level orchestration, dry-run output, and verification helpers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from agents.gpt_evaluator import GPTJudge

TRACE_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TRACE_TOOLS) not in sys.path:
    sys.path.insert(0, str(TRACE_TOOLS))

from .bridge import CLIENT_CONTAINER, DEFAULT_COMPOSE
from .plans import item_identifier, resolve_behavior_and_strategies
from .strategies import run_single_xteaming_strategy
from .targets import VrapCodexTarget, run_post_subtask_execution_turn
from agent_execution.verification import verify_target_state

def run_xteaming_subtask(
    *,
    item: dict[str, Any],
    subtask_index: int,
    subtask: str,
    attacker_config: dict[str, Any],
    tg_config: dict[str, Any],
    eval_config: dict[str, Any],
    plans_by_behavior_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    args.current_subtask_index = subtask_index
    behavior, selected_strategies = resolve_behavior_and_strategies(
        item,
        subtask_index,
        subtask,
        plans_by_behavior_id,
        attacker_config,
        args,
    )
    evaluator = (
        GPTJudge(model_name=eval_config.get("judge_model", "gpt-4o"))
        if args.use_gpt_judge
        else None
    )

    strategy_results = []
    strategy_runs = []
    successful_strategy_run = None
    last_strategy_run = None
    success_mode = getattr(args, "strategy_success_mode", "judge")

    for index, (strategy, strategy_meta) in enumerate(selected_strategies, 1):
        print(
            "  strategy "
            f"{index}/{len(selected_strategies)} "
            f"set={strategy_meta['set_number']} "
            f"strategy={strategy_meta['strategy_number']}",
            flush=True,
        )
        strategy_result, strategy_run = run_single_xteaming_strategy(
            item=item,
            subtask=subtask,
            behavior=behavior,
            strategy=strategy,
            strategy_meta=strategy_meta,
            attacker_config=attacker_config,
            tg_config=tg_config,
            evaluator=evaluator,
            args=args,
        )
        strategy_results.append(strategy_result)
        strategy_runs.append(strategy_run)
        last_strategy_run = strategy_run

        if strategy_result.get("error") and getattr(args, "stop_on_target_error", False):
            raise RuntimeError(
                "Stopping after target error in "
                f"set={strategy_meta['set_number']} "
                f"strategy={strategy_meta['strategy_number']}: "
                f"{strategy_result['error']}"
            )

        if success_mode == "execute_verify":
            strategy_success = bool((strategy_run.get("verification") or {}).get("ok"))
            strategy_result["execution_verification_success"] = strategy_success
            strategy_result["strategy_success_mode"] = "execute_verify"
        else:
            strategy_success = bool(strategy_result["jailbreak_achieved"])

        if strategy_success and not args.run_all_strategies:
            successful_strategy_run = strategy_run
            if success_mode == "execute_verify":
                print("  execution verification succeeded; skipping remaining strategies", flush=True)
            else:
                print("  jailbreak achieved; skipping remaining strategies", flush=True)
            break
        if strategy_success:
            successful_strategy_run = strategy_run

    final_strategy_run = successful_strategy_run or last_strategy_run or {
        "strategy": {},
        "strategy_meta": {},
        "input_output": [],
        "turns": [],
        "actions": [],
        "model_output": {"text": "", "returncode": None, "error": None},
        "formal_state": {},
        "verification": {"status": "pending", "ok": None, "checks": []},
    }
    if (
        getattr(args, "post_subtask_execution_turn", False)
        and success_mode != "execute_verify"
    ):
        post_execution = run_post_subtask_execution_turn(
            item=item,
            args=args,
            strategy_run=final_strategy_run,
        )
        print(
            "  post-subtask execution "
            f"status={post_execution.get('status')} "
            f"actions={len(post_execution.get('actions') or [])}",
            flush=True,
        )
    return {
        "item": item,
        "subtask": {
            "index": subtask_index,
            "target": subtask,
            "behavior": behavior,
        },
        "run": {
            "backend": "agent",
            "target_backend": getattr(args, "target_backend", "codex"),
            "target_model": args.claude_model
            if getattr(args, "target_backend", "codex") == "claude"
            else args.codex_model,
            "codex_model": args.codex_model,
            "claude_model": getattr(args, "claude_model", None),
            "claude_home": getattr(args, "claude_home", None),
            "claude_runner": getattr(args, "claude_runner", None),
            "attacker_backend": "transformers",
            "attacker_model": attacker_config["model"],
            "attacker_identity": str(item.get("attacker_identity") or "root"),
            "agent_user": args.agent_user or str(item.get("attacker_identity") or "root"),
            "container": CLIENT_CONTAINER,
            "agent_session": args.agent_session,
            "target_execution_mode": args.target_execution_mode,
            "post_subtask_execution_turn": args.post_subtask_execution_turn,
            "bypass": args.bypass,
            "strategies_per_behavior": len(selected_strategies),
            "executed_strategies": len(strategy_results),
            "run_all_strategies": args.run_all_strategies,
            "strategy_success_mode": success_mode,
            "strategy_execution_output_mode": getattr(
                args, "strategy_execution_output_mode", None
            ),
            "strategy_execution_prompt_style": getattr(
                args, "strategy_execution_prompt_style", None
            ),
            "gpt_judge": args.use_gpt_judge,
            "gpt_judge_model": eval_config.get("judge_model", "gpt-4o") if args.use_gpt_judge else None,
            "textgrad": args.textgrad,
            "textgrad_model": tg_config.get("model") if args.textgrad else None,
            "textgrad_backend": "transformers" if args.textgrad else None,
            "isolated_textgrad": args.isolated_textgrad if args.textgrad else None,
            "textgrad_isolation_mode": args.textgrad_isolation_mode if args.textgrad else None,
            "textgrad_max_turns_per_phase": tg_config.get("max_turns_per_phase") if args.textgrad else None,
            "plan_revision": args.plan_revision,
        },
        "strategy": final_strategy_run["strategy"],
        "strategy_meta": final_strategy_run["strategy_meta"],
        "strategies": strategy_results,
        "strategy_runs": strategy_runs,
        "input_output": final_strategy_run["input_output"],
        "turns": final_strategy_run["turns"],
        "actions": final_strategy_run["actions"],
        "formal_state": final_strategy_run.get("formal_state") or {},
        "model_output": final_strategy_run["model_output"],
        "verification": final_strategy_run["verification"],
        "replay": {
            "compose_file": str(DEFAULT_COMPOSE),
            "dataset_item_id": item_identifier(item),
            "subtask_index": subtask_index,
            "steps": final_strategy_run["actions"],
        },
    }


def build_dry_run_result(
    *,
    item: dict[str, Any],
    subtask_index: int,
    subtask: str,
    attacker_config: dict[str, Any],
    plans_by_behavior_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    behavior, selected_strategies = resolve_behavior_and_strategies(
        item,
        subtask_index,
        subtask,
        plans_by_behavior_id,
        attacker_config,
        args,
    )
    first_strategy = selected_strategies[0] if selected_strategies else ({}, {})
    return {
        "item": item,
        "subtask": {
            "index": subtask_index,
            "target": subtask,
            "behavior": behavior,
        },
        "run": {
            "backend": "agent",
            "target_backend": getattr(args, "target_backend", "codex"),
            "target_model": args.claude_model
            if getattr(args, "target_backend", "codex") == "claude"
            else args.codex_model,
            "codex_model": args.codex_model,
            "claude_model": getattr(args, "claude_model", None),
            "claude_home": getattr(args, "claude_home", None),
            "claude_runner": getattr(args, "claude_runner", None),
            "attacker_backend": "transformers",
            "attacker_model": attacker_config["model"],
            "dry_run": True,
            "target_execution_mode": args.target_execution_mode,
            "post_subtask_execution_turn": args.post_subtask_execution_turn,
            "bypass": args.bypass,
            "strategies_per_behavior": len(selected_strategies),
            "run_all_strategies": args.run_all_strategies,
            "strategy_success_mode": getattr(args, "strategy_success_mode", "judge"),
            "strategy_execution_output_mode": getattr(
                args, "strategy_execution_output_mode", None
            ),
            "strategy_execution_prompt_style": getattr(
                args, "strategy_execution_prompt_style", None
            ),
            "gpt_judge": args.use_gpt_judge,
            "textgrad": args.textgrad,
            "isolated_textgrad": args.isolated_textgrad if args.textgrad else None,
            "textgrad_isolation_mode": args.textgrad_isolation_mode if args.textgrad else None,
            "plan_revision": args.plan_revision,
        },
        "strategy": first_strategy[0],
        "strategy_meta": first_strategy[1],
        "selected_strategies": [
            {"strategy": strategy, "strategy_meta": strategy_meta}
            for strategy, strategy_meta in selected_strategies
        ],
        "strategies": [],
        "strategy_runs": [],
        "input_output": [],
        "turns": [],
        "actions": [],
        "model_output": {"text": "", "returncode": None, "error": None},
        "verification": {"status": "dry_run", "ok": None, "checks": []},
        "replay": {
            "compose_file": str(DEFAULT_COMPOSE),
            "dataset_item_id": item_identifier(item),
            "subtask_index": subtask_index,
            "steps": [],
        },
    }


def verify_item_compat(
    item: dict[str, Any],
    target: VrapCodexTarget,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return verify_target_state(item, target, args)
