"""Execute one completed strategy output and verify the live environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bridge import (
    DEFAULT_CONTAINER_CLAUDE_HOME,
    cleanup_environment,
    prepare_compose,
    run_claude_in_vrap,
    run_codex_in_vrap,
    setup_environment,
    verify_item,
)
from .utils import redact_secrets, with_strategy_meta
from .scripts.codex_execute_item_subtasks_same_session import (
    build_generated_execution_prompt,
    build_live_verify_history,
    select_execution_prompt_builder,
    subtask_source_context,
)


class ContinuousItemExecution:
    """Execute multiple subtask outputs in one live environment/session."""

    def __init__(self, *, item: dict[str, Any], args: argparse.Namespace):
        self.item = item
        self.args = args
        self.attacker_identity = str(item.get("attacker_identity") or "root")
        self.agent_user = args.agent_user or self.attacker_identity
        self.backend = getattr(args, "target_backend", "codex")
        self.temp_compose = None
        self.session_id: str | None = None
        self.returncodes: list[int | None] = []
        self.result_row: dict[str, Any] = {
            "item": item,
            "backend": self.backend,
            "model": getattr(args, "claude_model", None)
            if self.backend == "claude"
            else getattr(args, "codex_model", None),
            "mode": f"same_{self.backend}_session_same_vrap_container",
            "output_mode": getattr(args, "strategy_execution_output_mode", "all"),
            "prompt_style": getattr(args, "strategy_execution_prompt_style", "rule"),
            "input_output": [],
            "actions": [],
        }

    def __enter__(self) -> "ContinuousItemExecution":
        self.temp_compose = prepare_compose(self.args.compose_file, self.attacker_identity)
        setup_environment(self.item, self.temp_compose, self.attacker_identity)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.temp_compose is not None and not getattr(self.args, "keep_container", False):
            cleanup_environment(self.temp_compose)
            self.temp_compose.unlink(missing_ok=True)

    def execute_subtask_output(
        self,
        *,
        subtask_number: int,
        subtask_count: int,
        subtask_index: int,
        subtask: str,
        strategy_run: dict[str, Any],
    ) -> dict[str, Any]:
        prompt, generated_parts, source_kind = build_strategy_execution_prompt(
            item=self.item,
            subtask_number=subtask_number,
            subtask_count=subtask_count,
            subtask_index=subtask_index,
            subtask=subtask,
            strategy_run=strategy_run,
            args=self.args,
        )
        session_before = self.session_id
        result = run_strategy_execution_prompt(
            prompt,
            args=self.args,
            backend=self.backend,
            agent_user=self.agent_user,
            resume_session_id=self.session_id,
        )
        self.session_id = result.get("thread_id") or self.session_id
        actions = result.get("actions") or []
        self.returncodes.append(result.get("returncode"))
        if result.get("returncode") not in (None, 0):
            raise RuntimeError(
                "strategy execution failed "
                f"returncode={result.get('returncode')} "
                f"error={redact_secrets(str(result.get('error') or ''))}"
            )

        safe_result = json.loads(redact_secrets(json.dumps(result, ensure_ascii=False)))
        entry = {
            "subtask_index": subtask_index,
            "target": subtask,
            "input": prompt,
            "prompt_style": getattr(self.args, "strategy_execution_prompt_style", "rule"),
            "prompt_parts": generated_parts,
            "source_kind": source_kind,
            "output": result.get("text") or "",
            "actions": actions,
            "turn": {
                "returncode": result.get("returncode"),
                "error": redact_secrets(str(result.get("error") or "")) or None,
                "session_id_before": session_before,
                "session_id_after": self.session_id,
                "action_count": len(actions),
                "command": result.get("command"),
            },
        }
        self.result_row["input_output"].append(entry)
        self.result_row["actions"].append(actions)

        strategy_meta = dict(strategy_run.get("strategy_meta") or {})
        execution_actions = with_strategy_meta(actions, strategy_meta)
        for action in execution_actions:
            action["source"] = "item_attempt_execution"
            action["subtask_index"] = subtask_index
        execution = {
            "status": "completed",
            "source_kind": source_kind,
            "source_output_mode": getattr(self.args, "strategy_execution_output_mode", "all"),
            "prompt_style": getattr(self.args, "strategy_execution_prompt_style", "rule"),
            "prompt": prompt,
            "prompt_parts": generated_parts,
            "input_output": [entry],
            "actions": execution_actions,
            "raw_result": safe_result,
        }
        print(
            "    execution subtask "
            f"{subtask_number}/{subtask_count} "
            f"actions={len(execution_actions)} "
            f"session_before={session_before} session_after={self.session_id}",
            flush=True,
        )
        return execution

    def verify(self) -> dict[str, Any]:
        verification = verify_item(
            self.item,
            build_live_verify_history(self.result_row),
            self.attacker_identity,
        )
        safe_verification = json.loads(
            redact_secrets(json.dumps(verification, ensure_ascii=False))
        )
        self.result_row["live_verification"] = safe_verification
        print(
            "  item live_verification="
            f"{safe_verification.get('ok')} "
            f"checks={len(safe_verification.get('checks') or [])}",
            flush=True,
        )
        return safe_verification

    def summary(self) -> dict[str, Any]:
        action_lists = self.result_row.get("actions") or []
        flat_actions = [
            action
            for actions in action_lists
            for action in (actions or [])
        ]
        return {
            **self.result_row,
            "summary": {
                "subtask_count": len(self.result_row.get("input_output") or []),
                "total_action_count": len(flat_actions),
                "action_counts": [len(actions or []) for actions in action_lists],
                "returncodes": list(self.returncodes),
                "final_session_id": self.session_id,
                "live_verification_ok": (
                    self.result_row.get("live_verification") or {}
                ).get("ok"),
            },
        }


def build_strategy_execution_prompt(
    *,
    item: dict[str, Any],
    subtask_number: int,
    subtask_count: int,
    subtask_index: int,
    subtask: str,
    strategy_run: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, dict[str, str] | None, str]:
    entry = {
        "subtask_index": subtask_index,
        "target": subtask,
        "input_output": strategy_run.get("input_output") or [],
    }
    output_mode = getattr(args, "strategy_execution_output_mode", "all")
    source_text, source_kind = subtask_source_context(entry, output_mode)
    if not source_text:
        raise ValueError(f"subtask {subtask_index} has no source text for execution")

    prompt_style = getattr(args, "strategy_execution_prompt_style", "rule")
    generated_parts = None
    if prompt_style == "auto":
        prompt, generated_parts = build_generated_execution_prompt(
            item=item,
            subtask_number=subtask_number,
            subtask_count=subtask_count,
            subtask_index=subtask_index,
            target=subtask,
            source_text=source_text,
            model=getattr(
                args,
                "strategy_execution_prompt_builder_model",
                "gpt-4o-2024-11-20",
            ),
            responses_base_url=getattr(args, "responses_base_url", None),
            responses_env_key=getattr(args, "responses_env_key", "OPENAI_API_KEY"),
            timeout=float(getattr(args, "strategy_execution_prompt_builder_timeout", 120)),
            include_item_prompt=bool(
                getattr(args, "strategy_execution_prompt_builder_include_item_prompt", False)
            ),
            omit_subtask_info=bool(
                getattr(args, "strategy_execution_omit_subtask_info", False)
            ),
        )
    else:
        prompt_builder = select_execution_prompt_builder(prompt_style)
        prompt = prompt_builder(
            subtask_number=subtask_number,
            subtask_count=subtask_count,
            subtask_index=subtask_index,
            target=subtask,
            source_text=source_text,
            omit_subtask_info=bool(
                getattr(args, "strategy_execution_omit_subtask_info", False)
            ),
        )
    return prompt, generated_parts, source_kind


def run_strategy_execution_prompt(
    prompt: str,
    *,
    args: argparse.Namespace,
    backend: str,
    agent_user: str,
    resume_session_id: str | None,
) -> dict[str, Any]:
    timeout = int(
        getattr(args, "strategy_execution_timeout", None)
        or getattr(args, "target_timeout", None)
        or getattr(args, "codex_timeout", 600)
    )
    if backend == "claude":
        return run_claude_in_vrap(
            prompt,
            model=getattr(args, "claude_model", None),
            user=agent_user,
            timeout=timeout,
            bypass=bool(getattr(args, "bypass", False)),
            max_turns=getattr(args, "claude_max_turns", None),
            anthropic_base_url=getattr(args, "anthropic_base_url", None),
            anthropic_env_key=getattr(args, "anthropic_env_key", "ANTHROPIC_API_KEY"),
            allowed_tools=_parse_csv_arg(getattr(args, "allowed_tools", None)),
            claude_home=getattr(args, "claude_home", DEFAULT_CONTAINER_CLAUDE_HOME),
            resume_session_id=resume_session_id,
            claude_runner=getattr(args, "claude_runner", "sdk"),
        )
    return run_codex_in_vrap(
        prompt,
        model=getattr(args, "codex_model", None),
        user=agent_user,
        timeout=timeout,
        bypass=bool(getattr(args, "bypass", False)),
        write_config=bool(getattr(args, "codex_write_config", True)),
        responses_base_url=getattr(args, "responses_base_url", None),
        responses_env_key=getattr(args, "responses_env_key", "OPENAI_API_KEY"),
        resume_thread_id=resume_session_id,
    )


def execute_strategy_output_and_verify(
    *,
    item: dict[str, Any],
    subtask_index: int,
    subtask: str,
    strategy_run: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the saved output from one x-teaming strategy, then live-verify it."""

    attacker_identity = str(item.get("attacker_identity") or "root")
    agent_user = args.agent_user or attacker_identity
    try:
        prompt, generated_parts, source_kind = build_strategy_execution_prompt(
            item=item,
            subtask_number=1,
            subtask_count=1,
            subtask_index=subtask_index,
            subtask=subtask,
            strategy_run=strategy_run,
            args=args,
        )
    except ValueError:
        return {
            "status": "skipped",
            "reason": "no_source_text",
            "live_verification": {"ok": False, "checks": []},
        }

    temp_compose = prepare_compose(args.compose_file, attacker_identity)
    backend = getattr(args, "target_backend", "codex")
    session_id = None
    try:
        setup_environment(item, temp_compose, attacker_identity)
        result = run_strategy_execution_prompt(
            prompt,
            args=args,
            backend=backend,
            agent_user=agent_user,
            resume_session_id=None,
        )
        session_id = result.get("thread_id")
        actions = result.get("actions") or []
        input_output = [
            {
                "subtask_index": subtask_index,
                "target": subtask,
                "input": prompt,
                "prompt_style": prompt_style,
                "prompt_parts": generated_parts,
                "source_kind": source_kind,
                "output": result.get("text") or "",
                "actions": actions,
                "turn": {
                    "returncode": result.get("returncode"),
                    "error": redact_secrets(str(result.get("error") or "")) or None,
                    "session_id_before": None,
                    "session_id_after": session_id,
                    "action_count": len(actions),
                    "command": result.get("command"),
                },
            }
        ]
        result_row = {
            "mode": "single_strategy_execute_verify",
            "backend": backend,
            "model": getattr(args, "claude_model", None)
            if backend == "claude"
            else getattr(args, "codex_model", None),
            "input_output": input_output,
            "actions": [actions],
        }
        live_verification = verify_item(
            item,
            build_live_verify_history(result_row),
            attacker_identity,
        )
    finally:
        if not getattr(args, "keep_container", False):
            cleanup_environment(temp_compose)
            temp_compose.unlink(missing_ok=True)

    safe_result = json.loads(redact_secrets(json.dumps(result, ensure_ascii=False)))
    safe_verification = json.loads(
        redact_secrets(json.dumps(live_verification, ensure_ascii=False))
    )
    strategy_meta = dict(strategy_run.get("strategy_meta") or {})
    execution_actions = with_strategy_meta(actions, strategy_meta)
    for action in execution_actions:
        action["source"] = "strategy_execution"
    return {
            "status": "completed",
            "source_kind": source_kind,
        "source_output_mode": getattr(args, "strategy_execution_output_mode", "all"),
        "prompt_style": getattr(args, "strategy_execution_prompt_style", "rule"),
        "prompt": prompt,
        "prompt_parts": generated_parts,
        "input_output": input_output,
        "actions": execution_actions,
        "raw_result": safe_result,
        "live_verification": safe_verification,
        "verification_ok": bool(safe_verification.get("ok")),
    }


def apply_strategy_execution_result(
    strategy_run: dict[str, Any],
    execution: dict[str, Any],
) -> None:
    """Attach execution artifacts to a strategy run without hiding x-teaming turns."""

    strategy_run["strategy_execution"] = execution
    strategy_run["execution_input_output"] = execution.get("input_output") or []
    strategy_run["execution_actions"] = execution.get("actions") or []
    strategy_run.setdefault("actions", []).extend(execution.get("actions") or [])
    strategy_run["verification"] = execution.get("live_verification") or {
        "ok": False,
        "checks": [],
    }


def _parse_csv_arg(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]
