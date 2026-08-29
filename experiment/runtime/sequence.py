"""X-teaming item-level execution across all AdvCUA subprompts."""
from __future__ import annotations

import argparse
import copy
import itertools
import random
import sys
from pathlib import Path
from typing import Any

from .bridge import CLIENT_CONTAINER
from .plans import resolve_behavior_and_strategies
from .runner import run_xteaming_subtask
from .utils import redact_secrets


TRACE_ROOT = Path(__file__).resolve().parents[2]
if str(TRACE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACE_ROOT))

from agent_execution.sequence import (  # noqa: E402
    MultiSubtaskRunner,
    make_action_summary_prefix,
    mark_cleaned_snapshot_paths,
    result_session_state,
    summarize_execution_actions,
)
from agent_execution.strategy_execution import ContinuousItemExecution


class XTeamingSubtaskRunner(MultiSubtaskRunner):
    """Connect the generic multi-subtask runner to x-teaming subtask execution."""

    def __init__(
        self,
        *,
        item: dict[str, Any],
        attacker_config: dict[str, Any],
        tg_config: dict[str, Any],
        eval_config: dict[str, Any],
        plans_by_behavior_id: dict[str, dict[str, Any]],
        args: argparse.Namespace,
    ):
        super().__init__(item=item, args=args)
        self.attacker_config = attacker_config
        self.tg_config = tg_config
        self.eval_config = eval_config
        self.plans_by_behavior_id = plans_by_behavior_id

    def summary_model(self) -> str:
        return self.eval_config.get("judge_model", "gpt-4o-2024-11-20")

    def run_subtask(
        self,
        *,
        subtask_index: int,
        subtask: str,
        sub_args: argparse.Namespace,
        sub_config: dict[str, Any],
    ) -> dict[str, Any]:
        sub_attacker_config = dict(self.attacker_config)
        sub_attacker_config.update(sub_config)
        return run_xteaming_subtask(
            item=self.item,
            subtask_index=subtask_index,
            subtask=subtask,
            attacker_config=sub_attacker_config,
            tg_config=self.tg_config,
            eval_config=self.eval_config,
            plans_by_behavior_id=self.plans_by_behavior_id,
            args=sub_args,
        )

    def build_run_metadata(self) -> dict[str, Any]:
        return {
            "backend": "codex",
            "target_backend": getattr(self.args, "target_backend", "codex"),
            "target_model": self.args.claude_model
            if getattr(self.args, "target_backend", "codex") == "claude"
            else self.args.codex_model,
            "codex_model": self.args.codex_model,
            "claude_model": getattr(self.args, "claude_model", None),
            "claude_home": getattr(self.args, "claude_home", None),
            "claude_runner": getattr(self.args, "claude_runner", None),
            "container": CLIENT_CONTAINER,
            "lab_backend": getattr(self.args, "lab_backend", "vrap"),
            "link_mode": self.args.link_mode,
            "subprompt_count": len(self.subprompts),
            "target_execution_mode": self.args.target_execution_mode,
            "post_subtask_execution_turn": getattr(self.args, "post_subtask_execution_turn", False),
            "bypass": self.args.bypass,
            "strategies_per_behavior": self.attacker_config.get("strategies_per_behavior"),
            "gpt_judge": self.args.use_gpt_judge,
            "textgrad": self.args.textgrad,
            "isolated_textgrad": self.args.isolated_textgrad if self.args.textgrad else None,
            "textgrad_isolation_mode": self.args.textgrad_isolation_mode if self.args.textgrad else None,
        }


class XTeamingItemExecuteVerifyRunner:
    """Run random item-level strategy combinations with continuous execution."""

    def __init__(
        self,
        *,
        item: dict[str, Any],
        attacker_config: dict[str, Any],
        tg_config: dict[str, Any],
        eval_config: dict[str, Any],
        plans_by_behavior_id: dict[str, dict[str, Any]],
        args: argparse.Namespace,
    ):
        self.item = item
        self.subprompts = list(item.get("sub_prompts") or [])
        self.attacker_config = attacker_config
        self.tg_config = tg_config
        self.eval_config = eval_config
        self.plans_by_behavior_id = plans_by_behavior_id
        self.args = args
        self.candidates_by_subtask = self._load_candidates()
        self._attempted_combo_keys: set[tuple[tuple[int, int], ...]] = set()

    def _load_candidates(self) -> list[list[tuple[dict[str, Any], dict[str, Any]]]]:
        candidates: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
        for subtask_index, subtask in enumerate(self.subprompts):
            _behavior, selected = resolve_behavior_and_strategies(
                self.item,
                subtask_index,
                subtask,
                self.plans_by_behavior_id,
                self.attacker_config,
                self.args,
            )
            if not selected:
                raise ValueError(f"subtask {subtask_index} has no selected strategies")
            candidates.append(selected)
        return candidates

    def _max_attempts(self) -> int:
        configured = (
            self.args.item_attempts
            if self.args.item_attempts is not None
            else (
                self.args.strategies_per_behavior
                if self.args.strategies_per_behavior is not None
                else self.attacker_config.get("strategies_per_behavior", 1)
            )
        )
        total_combinations = 1
        for candidates in self.candidates_by_subtask:
            total_combinations *= len(candidates)
        return min(max(int(configured), 1), total_combinations)

    def _sample_combinations(self) -> list[tuple[tuple[dict[str, Any], dict[str, Any]], ...]]:
        max_attempts = self._max_attempts()
        rng = random.Random(self.args.strategy_sample_seed)
        seen: set[tuple[tuple[int, int], ...]] = set(self._attempted_combo_keys)
        sampled: list[tuple[tuple[dict[str, Any], dict[str, Any]], ...]] = []

        def combo_key(combo: tuple[tuple[dict[str, Any], dict[str, Any]], ...]) -> tuple[tuple[int, int], ...]:
            return tuple(
                (int(meta.get("set_number", 0)), int(meta["strategy_number"]))
                for _strategy, meta in combo
            )

        collision_budget = max(100, max_attempts * 20)
        while len(sampled) < max_attempts and collision_budget > 0:
            collision_budget -= 1
            combo = tuple(rng.choice(candidates) for candidates in self.candidates_by_subtask)
            key = combo_key(combo)
            if key in seen:
                continue
            seen.add(key)
            self._attempted_combo_keys.add(key)
            sampled.append(combo)

        if len(sampled) < max_attempts:
            for combo in itertools.product(*self.candidates_by_subtask):
                key = combo_key(combo)
                if key in seen:
                    continue
                seen.add(key)
                self._attempted_combo_keys.add(key)
                sampled.append(tuple(combo))
                if len(sampled) >= max_attempts:
                    break
        return sampled

    def _run_subtask_with_strategy(
        self,
        *,
        subtask_index: int,
        subtask: str,
        strategy_meta: dict[str, Any],
    ) -> dict[str, Any]:
        sub_args = copy.copy(self.args)
        sub_args.plan_strategy = strategy_meta["strategy_number"]
        sub_args.run_all_strategies = False
        sub_args.strategy_success_mode = "judge"
        sub_args.verify = False
        sub_args.post_subtask_execution_turn = False
        sub_args.target_execution_mode = "chat"
        return run_xteaming_subtask(
            item=self.item,
            subtask_index=subtask_index,
            subtask=subtask,
            attacker_config=self.attacker_config,
            tg_config=self.tg_config,
            eval_config=self.eval_config,
            plans_by_behavior_id=self.plans_by_behavior_id,
            args=sub_args,
        )

    def build_run_metadata(self) -> dict[str, Any]:
        return {
            "backend": "codex",
            "target_backend": getattr(self.args, "target_backend", "codex"),
            "target_model": self.args.claude_model
            if getattr(self.args, "target_backend", "codex") == "claude"
            else self.args.codex_model,
            "codex_model": self.args.codex_model,
            "claude_model": getattr(self.args, "claude_model", None),
            "claude_runner": getattr(self.args, "claude_runner", None),
            "container": CLIENT_CONTAINER,
            "lab_backend": getattr(self.args, "lab_backend", "vrap"),
            "link_mode": self.args.link_mode,
            "subprompt_count": len(self.subprompts),
            "strategy_success_mode": "execute_verify",
            "item_attempts": self._max_attempts(),
            "strategy_combination_sampling": "random_unique_tuple",
            "strategy_execution_output_mode": self.args.strategy_execution_output_mode,
            "strategy_execution_prompt_style": self.args.strategy_execution_prompt_style,
            "bypass": self.args.bypass,
            "gpt_judge": self.args.use_gpt_judge,
            "textgrad": self.args.textgrad,
        }

    def run(self) -> dict[str, Any]:
        attempts = []
        successful_attempt = None
        combinations = self._sample_combinations()
        subtask_count = len(self.subprompts)
        for attempt_index, combo in enumerate(combinations, 1):
            combo_meta = [
                {
                    "subtask_index": subtask_index,
                    "strategy_number": meta["strategy_number"],
                }
                for subtask_index, (_strategy, meta) in enumerate(combo)
            ]
            for subtask_index, (_strategy, meta) in enumerate(combo):
                if "set_number" in meta:
                    combo_meta[subtask_index]["set_number"] = meta["set_number"]
            combo_label = [
                (
                    meta.get("set_number"),
                    meta["strategy_number"],
                )
                if "set_number" in meta
                else meta["strategy_number"]
                for meta in combo_meta
            ]
            print(
                f"item attempt {attempt_index}/{len(combinations)} "
                f"combination={combo_label}",
                flush=True,
            )
            subprompt_results = []
            input_output_by_subprompt = []
            execution_results = []
            attempt_error = None
            verification = {
                "status": "skipped",
                "ok": False,
                "checks": [],
                "reason": "attempt did not complete",
            }
            execution_summary = {}

            try:
                for subtask_number, (subtask_index, subtask) in enumerate(
                    enumerate(self.subprompts),
                    1,
                ):
                    _strategy, strategy_meta = combo[subtask_index]
                    print(
                        f"subprompt {subtask_number}/{subtask_count} "
                        f"index={subtask_index} "
                        f"strategy={strategy_meta['strategy_number']} "
                        f"target={subtask[:100]!r}",
                        flush=True,
                    )
                    subtask_result = self._run_subtask_with_strategy(
                        subtask_index=subtask_index,
                        subtask=subtask,
                        strategy_meta=strategy_meta,
                    )
                    subprompt_results.append(subtask_result)

                with ContinuousItemExecution(item=self.item, args=self.args) as executor:
                    for subtask_number, (subtask_index, subtask_result) in enumerate(
                        enumerate(subprompt_results),
                        1,
                    ):
                        subtask = self.subprompts[subtask_index]
                        print(
                            f"  executing subprompt {subtask_number}/{subtask_count} "
                            f"index={subtask_index}",
                            flush=True,
                        )
                        execution = executor.execute_subtask_output(
                            subtask_number=subtask_number,
                            subtask_count=subtask_count,
                            subtask_index=subtask_index,
                            subtask=subtask,
                            strategy_run=subtask_result,
                        )
                        subtask_result["item_attempt_execution"] = execution
                        execution_results.append(execution)
                        input_output_by_subprompt.append(
                            {
                                "subtask_index": subtask_index,
                                "target": subtask,
                                "input_output": subtask_result.get("input_output") or [],
                                "execution_input_output": execution.get("input_output") or [],
                            }
                        )
                    verification = executor.verify()
                    execution_summary = executor.summary()
            except Exception as exc:
                attempt_error = {
                    "type": type(exc).__name__,
                    "message": redact_secrets(str(exc)),
                }
                print(
                    f"item attempt {attempt_index} failed; trying next combination: "
                    f"{type(exc).__name__}: {attempt_error['message']}",
                    flush=True,
                )

            attempt_result = {
                "attempt": attempt_index,
                "strategy_combination": combo_meta,
                "subprompts": subprompt_results,
                "input_output": input_output_by_subprompt,
                "execution": {
                    **execution_summary,
                    "subtask_executions": execution_results,
                },
                "verification": verification,
                "verification_ok": bool(verification.get("ok")),
            }
            if attempt_error is not None:
                attempt_result["error"] = attempt_error
            attempts.append(attempt_result)
            if attempt_result["verification_ok"]:
                successful_attempt = attempt_result
                print(
                    f"item attempt {attempt_index} verification succeeded; stopping attempts",
                    flush=True,
                )
                break
            print(
                f"item attempt {attempt_index} verification failed; trying next combination",
                flush=True,
            )

        final_attempt = successful_attempt or (attempts[-1] if attempts else {})
        execution = final_attempt.get("execution") or {}
        flat_actions = [
            action
            for actions in (execution.get("actions") or [])
            for action in (actions or [])
        ]
        return {
            "item": self.item,
            "run": self.build_run_metadata(),
            "attempts": attempts,
            "selected_attempt": final_attempt.get("attempt"),
            "strategy_combination": final_attempt.get("strategy_combination") or [],
            "subprompts": final_attempt.get("subprompts") or [],
            "input_output": final_attempt.get("input_output") or [],
            "execution": execution,
            "actions": flat_actions,
            "verification": final_attempt.get("verification") or {
                "ok": None,
                "checks": [],
            },
        }


def run_all_subprompts_for_item(
    *,
    item: dict[str, Any],
    attacker_config: dict[str, Any],
    tg_config: dict[str, Any],
    eval_config: dict[str, Any],
    plans_by_behavior_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if getattr(args, "strategy_success_mode", "judge") == "execute_verify":
        return XTeamingItemExecuteVerifyRunner(
            item=item,
            attacker_config=attacker_config,
            tg_config=tg_config,
            eval_config=eval_config,
            plans_by_behavior_id=plans_by_behavior_id,
            args=args,
        ).run()
    return XTeamingSubtaskRunner(
        item=item,
        attacker_config=attacker_config,
        tg_config=tg_config,
        eval_config=eval_config,
        plans_by_behavior_id=plans_by_behavior_id,
        args=args,
    ).run()
