"""Agent target wrappers plus container/session-state snapshot and replay helpers."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .bridge import (
    CLIENT_CONTAINER,
    DEFAULT_CONTAINER_CLAUDE_HOME,
    DEFAULT_CONTAINER_CODEX_HOME,
    build_input_output_pairs,
    clean_codex_noise,
    cleanup_environment,
    docker_exec,
    prepare_compose,
    run_claude_in_vrap,
    run_codex_in_vrap,
    setup_environment,
)
from .utils import (
    clean_input_output,
    current_turn_actions,
    redact_secrets,
    rewrite_turn_numbers,
    with_strategy_meta,
)
from .target_prompting import DEFAULT_POST_SUBTASK_EXECUTION_PROMPT, maybe_wrap_target_prompt
from vrap_action_replay import (
    replay_state_action,
    replay_state_actions,
    remove_container_path,
    shell_quote_single,
    write_container_text_file,
)

class VrapCodexTarget:
    """Target wrapper that preserves one agent session per subtask run.

    The historical class name is retained for x-teaming compatibility. The
    runtime is selected by ``args.target_backend`` and currently supports
    ``codex`` and ``claude``.
    """

    def __init__(self, item: dict[str, Any], args: argparse.Namespace):
        self.item = item
        self.args = args
        self.attacker_identity = str(item.get("attacker_identity") or "root")
        self.agent_user = args.agent_user or self.attacker_identity
        self.resume_id: str | None = None
        self.turn_results: list[dict[str, Any]] = []
        self.input_output: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.temp_compose: Path | None = None
        self.target_execution_prompt_applied = bool(
            getattr(args, "target_execution_prompt_applied", False)
        )
        self.backend = str(getattr(args, "target_backend", "codex") or "codex")

    def __enter__(self) -> "VrapCodexTarget":
        self.temp_compose = prepare_compose(self.args.compose_file, self.attacker_identity)
        setup_environment(self.item, self.temp_compose, self.attacker_identity)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.temp_compose is not None and not self.args.keep_container:
            cleanup_environment(self.temp_compose)
            self.temp_compose.unlink(missing_ok=True)

    def generate_response(self, attacker_message: str) -> str:
        target_prompt_mode = getattr(self.args, "target_execution_mode", "chat")
        apply_execution_prompt = (
            target_prompt_mode == "execute"
            and not self.target_execution_prompt_applied
        )
        prompt_to_codex, prompt_wrapped = maybe_wrap_target_prompt(
            attacker_message=attacker_message,
            item=self.item,
            subtask_index=getattr(self.args, "current_subtask_index", None),
            mode=target_prompt_mode if apply_execution_prompt else "chat",
        )
        result = self._run_agent(prompt_to_codex)
        if self.backend == "codex":
            result, _removed = clean_codex_noise(result)
        if prompt_wrapped:
            self.target_execution_prompt_applied = True
        next_resume_id = result.get("thread_id") or self.resume_id

        turn_number = len(self.turn_results) + 1
        turn_actions = []
        for action_index, action in enumerate(result.get("actions") or [], 1):
            turn_action = dict(action)
            turn_action["turn"] = turn_number
            turn_action["turn_action_index"] = action_index
            turn_actions.append(turn_action)
            self.actions.append(turn_action)

        output_text = str(result.get("text") or "")
        self.input_output.extend(
            build_input_output_pairs(attacker_message, output_text, turn_actions)
        )
        self.turn_results.append(
            {
                "turn": turn_number,
                "returncode": result.get("returncode"),
                "error": result.get("error"),
                "session_id_before": self.resume_id,
                "session_id_after": next_resume_id,
                "action_count": len(turn_actions),
                "command": result.get("command"),
                "target_execution_mode": target_prompt_mode,
                "target_prompt_wrapped": prompt_wrapped,
                "target_execution_prompt_applied_after": self.target_execution_prompt_applied,
                "agent_backend": self.backend,
                "agent_model": self.model_name,
                "agent_input_chars": len(prompt_to_codex),
                "codex_input_chars": len(prompt_to_codex),
            }
        )
        self.resume_id = next_resume_id
        return output_text

    @property
    def model_name(self) -> str | None:
        if self.backend == "claude":
            return getattr(self.args, "claude_model", None) or getattr(
                self.args, "codex_model", None
            )
        return getattr(self.args, "codex_model", None)

    @property
    def home_path(self) -> str:
        if self.backend == "claude":
            return (
                getattr(self.args, "claude_home", None)
                or DEFAULT_CONTAINER_CLAUDE_HOME
            )
        return DEFAULT_CONTAINER_CODEX_HOME

    def _run_agent(self, prompt: str) -> dict[str, Any]:
        should_resume = self.args.agent_session == "resume" and bool(self.resume_id)
        timeout = int(
            getattr(self.args, "target_timeout", None)
            or getattr(self.args, "codex_timeout", 600)
        )
        if self.backend == "codex":
            return run_codex_in_vrap(
                prompt,
                model=self.model_name,
                user=self.agent_user,
                timeout=timeout,
                bypass=self.args.bypass,
                write_config=self.args.codex_write_config,
                responses_base_url=self.args.responses_base_url,
                responses_env_key=self.args.responses_env_key,
                resume_thread_id=self.resume_id if should_resume else None,
            )
        if self.backend == "claude":
            return run_claude_in_vrap(
                prompt,
                model=self.model_name,
                user=self.agent_user,
                timeout=timeout,
                bypass=self.args.bypass,
                max_turns=getattr(self.args, "claude_max_turns", None),
                anthropic_base_url=getattr(self.args, "anthropic_base_url", None),
                anthropic_env_key=getattr(
                    self.args, "anthropic_env_key", "ANTHROPIC_API_KEY"
                ),
                allowed_tools=parse_csv_arg(getattr(self.args, "allowed_tools", None)),
                claude_home=self.home_path,
                resume_session_id=self.resume_id if should_resume else None,
                fork_session=bool(getattr(self.args, "fork_session", False)),
                claude_runner=getattr(self.args, "claude_runner", "sdk"),
            )
        raise ValueError(f"Unsupported target backend: {self.backend!r}")


def parse_csv_arg(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]

def run_docker_cmd(cmd: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def snapshot_agent_home(
    snapshot_root: Path,
    label: str,
    *,
    backend: str = "codex",
    home_path: str | None = None,
) -> Path | None:
    home = home_path or (
        DEFAULT_CONTAINER_CODEX_HOME
        if backend == "codex"
        else DEFAULT_CONTAINER_CLAUDE_HOME
    )
    snapshot = snapshot_root / f"{label}_{backend}_home"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    proc = run_docker_cmd(
        ["docker", "cp", f"{CLIENT_CONTAINER}:{home}", str(snapshot)],
        timeout=120,
    )
    if proc.returncode != 0:
        return None
    return snapshot


def snapshot_codex_home(snapshot_root: Path, label: str) -> Path | None:
    return snapshot_agent_home(snapshot_root, label, backend="codex")


def restore_agent_home(
    snapshot: Path | None,
    *,
    user: str,
    backend: str = "codex",
    home_path: str | None = None,
) -> dict[str, Any]:
    if snapshot is None:
        return {"status": "skipped", "reason": "no_agent_home_snapshot"}
    home = home_path or (
        DEFAULT_CONTAINER_CODEX_HOME
        if backend == "codex"
        else DEFAULT_CONTAINER_CLAUDE_HOME
    )
    parent = str(Path(home).parent)
    docker_exec(
        CLIENT_CONTAINER,
        [
            "bash",
            "-lc",
            f"rm -rf {shell_quote_single(home)} && mkdir -p {shell_quote_single(parent)}",
        ],
        user="root",
        timeout=60,
    )
    proc = run_docker_cmd(
        ["docker", "cp", str(snapshot), f"{CLIENT_CONTAINER}:{home}"],
        timeout=120,
    )
    chown = docker_exec(
        CLIENT_CONTAINER,
        ["chown", "-R", f"{user}:{user}", home],
        user="root",
        timeout=60,
    )
    return {
        "status": "ok" if proc.returncode == 0 and chown.returncode == 0 else "error",
        "error": redact_secrets((proc.stderr or chown.stderr or "").strip()),
    }


def restore_codex_home(snapshot: Path | None, *, user: str) -> dict[str, Any]:
    return restore_agent_home(snapshot, user=user, backend="codex")


def replay_action(action: dict[str, Any], *, user: str) -> dict[str, Any]:
    return replay_state_action(action, user=user)


def replay_actions(actions: list[dict[str, Any]], *, user: str) -> list[dict[str, Any]]:
    return replay_state_actions(actions, user=user)


def run_post_subtask_execution_turn(
    *,
    item: dict[str, Any],
    args: argparse.Namespace,
    strategy_run: dict[str, Any],
) -> dict[str, Any]:
    """Append one natural execution follow-up after a completed subtask run."""

    formal_state = dict(strategy_run.get("formal_state") or {})
    prompt = str(
        getattr(args, "post_subtask_execution_prompt", None)
        or DEFAULT_POST_SUBTASK_EXECUTION_PROMPT
    )
    if not formal_state.get("resume_id"):
        return {
            "status": "skipped",
            "reason": "no_formal_codex_session",
            "prompt": prompt,
        }
    formal_snapshot = formal_state.get("home_snapshot") or formal_state.get("codex_home_snapshot")
    if not formal_snapshot:
        return {
            "status": "skipped",
            "reason": "no_formal_agent_home_snapshot",
            "prompt": prompt,
        }

    post_args = copy.copy(args)
    post_args.target_execution_mode = "chat"
    official_turn = len(strategy_run.get("input_output") or []) + 1
    strategy_meta = dict(strategy_run.get("strategy_meta") or {})

    with VrapCodexTarget(item, post_args) as target:
        target.resume_id = formal_state.get("resume_id")
        target.target_execution_prompt_applied = bool(
            formal_state.get("target_execution_prompt_applied")
        )
        restore_result = restore_agent_home(
            formal_snapshot,
            user=target.agent_user,
            backend=target.backend,
            home_path=target.home_path,
        )
        replay_results = replay_actions(
            list(formal_state.get("actions") or []),
            user=target.agent_user,
        )
        response = target.generate_response(prompt)
        snapshot_parent = getattr(args, "snapshot_tmp_root", None) or Path.cwd()
        snapshot_parent.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_agent_home(
            snapshot_parent,
            f"post_subtask_turn_{official_turn}",
            backend=target.backend,
            home_path=target.home_path,
        )

        input_output = clean_input_output(target.input_output[-1:] if target.input_output else [])
        turns = rewrite_turn_numbers(target.turn_results[-1:] if target.turn_results else [], official_turn)
        actions = rewrite_turn_numbers(current_turn_actions(target), official_turn)
        for turn in turns:
            turn["post_subtask_execution_turn"] = True
            turn["replayed_action_count"] = len(formal_state.get("actions") or [])
            turn["restore_result"] = restore_result
            turn["replay_status"] = summarize_replay_results(replay_results)
        for action in actions:
            action["post_subtask_execution_turn"] = True

    durable_snapshot = None
    if snapshot is not None and getattr(args, "persist_codex_home_snapshots", False):
        store_root = Path(
            getattr(args, "codex_snapshot_store", None)
            or (getattr(args, "snapshot_tmp_root", None) or Path.cwd()) / "codex_home_snapshots"
        )
        store_root.mkdir(parents=True, exist_ok=True)
        label = f"post_subtask_turn_{official_turn}_{target.backend}_home"
        durable_snapshot = store_root / label
        if durable_snapshot.exists():
            shutil.rmtree(durable_snapshot)
        shutil.copytree(
            snapshot,
            durable_snapshot,
            ignore=shutil.ignore_patterns("tmp"),
            symlinks=True,
            ignore_dangling_symlinks=True,
        )

    wrapped_actions = with_strategy_meta(actions, strategy_meta)
    strategy_run.setdefault("input_output", []).extend(input_output)
    strategy_run.setdefault("turns", []).extend(with_strategy_meta(turns, strategy_meta))
    strategy_run.setdefault("actions", []).extend(wrapped_actions)
    strategy_run["post_subtask_execution"] = {
        "status": "completed",
        "prompt": prompt,
        "response": response,
        "input_output": input_output,
        "turns": turns,
        "actions": actions,
        "restore": restore_result,
        "replay_status": summarize_replay_results(replay_results),
        "replayed_action_count": len(formal_state.get("actions") or []),
    }
    strategy_run["model_output"] = {
        "text": response,
        "returncode": turns[-1].get("returncode") if turns else None,
        "error": turns[-1].get("error") if turns else None,
    }
    next_snapshot = str(durable_snapshot) if durable_snapshot is not None else None
    new_formal_state = {
        "resume_id": target.resume_id,
        "home_snapshot": next_snapshot,
        "codex_home_snapshot": next_snapshot,
        "agent_backend": target.backend,
        "actions": list(formal_state.get("actions") or []) + wrapped_actions,
        "target_execution_prompt_applied": bool(
            target.target_execution_prompt_applied
        ),
    }
    strategy_run["formal_state"] = new_formal_state
    if snapshot is not None and not getattr(args, "persist_codex_home_snapshots", False):
        shutil.rmtree(snapshot, ignore_errors=True)
    return strategy_run["post_subtask_execution"]


class ReplayTrialTarget:
    """Runs each candidate query in a fresh replayed execution environment."""

    def __init__(
        self,
        item: dict[str, Any],
        args: argparse.Namespace,
        accepted_prompts: list[str],
        trial_records: list[dict[str, Any]],
    ):
        self.item = item
        self.args = args
        self.accepted_prompts = accepted_prompts
        self.trial_records = trial_records

    def generate_response(self, attacker_message: str) -> str:
        trial_number = len(self.trial_records) + 1
        with VrapCodexTarget(self.item, self.args) as target:
            for accepted_prompt in self.accepted_prompts:
                target.generate_response(accepted_prompt)
            response = target.generate_response(attacker_message)
            turn_record = dict(target.turn_results[-1]) if target.turn_results else {}
            turn_record["replayed_turns"] = len(self.accepted_prompts)
            self.trial_records.append(
                {
                    "trial": trial_number,
                    "replayed_turns": len(self.accepted_prompts),
                    "input": attacker_message,
                    "output": response,
                    "input_output": clean_input_output(target.input_output[-1:] if target.input_output else []),
                    "turn": turn_record,
                    "actions": current_turn_actions(target),
                }
            )
            return response


class SnapshotPromotedTrialTarget:
    """Runs candidate trials from the current formal state and promotes winners."""

    def __init__(
        self,
        item: dict[str, Any],
        args: argparse.Namespace,
        formal_state: dict[str, Any],
        trial_records: list[dict[str, Any]],
        snapshot_root: Path,
    ):
        self.item = item
        self.args = args
        self.formal_state = formal_state
        self.trial_records = trial_records
        self.snapshot_root = snapshot_root
        self.last_trial_state: dict[str, Any] | None = None

    def generate_response(self, attacker_message: str) -> str:
        trial_number = len(self.trial_records) + 1
        with VrapCodexTarget(self.item, self.args) as target:
            target.resume_id = self.formal_state.get("resume_id")
            target.target_execution_prompt_applied = bool(
                self.formal_state.get("target_execution_prompt_applied")
            )
            formal_snapshot = self.formal_state.get("home_snapshot") or self.formal_state.get("codex_home_snapshot")
            restore_result = restore_agent_home(
                formal_snapshot,
                user=target.agent_user,
                backend=target.backend,
                home_path=target.home_path,
            )
            replay_results = replay_actions(
                list(self.formal_state.get("actions") or []),
                user=target.agent_user,
            )
            response = target.generate_response(attacker_message)
            snapshot = snapshot_agent_home(
                self.snapshot_root,
                f"trial_{trial_number}",
                backend=target.backend,
                home_path=target.home_path,
            )
            turn_record = dict(target.turn_results[-1]) if target.turn_results else {}
            current_actions = current_turn_actions(target)
            current_input_output = clean_input_output(
                target.input_output[-1:] if target.input_output else []
            )
            self.last_trial_state = {
                "trial": trial_number,
                "input": attacker_message,
                "output": response,
                "home_snapshot": snapshot,
                "codex_home_snapshot": snapshot,
                "agent_backend": target.backend,
                "session_id_after": target.resume_id,
                "target_execution_prompt_applied": target.target_execution_prompt_applied,
                "input_output": current_input_output,
                "turn": turn_record,
                "actions": current_actions,
                "restore_result": restore_result,
                "replay_results": replay_results,
                "replayed_action_count": len(self.formal_state.get("actions") or []),
            }
            self.trial_records.append(
                {
                    "trial": trial_number,
                    "state_mode": "snapshot_promote",
                    "input": attacker_message,
                    "output": response,
                    "input_output": current_input_output,
                    "turn": turn_record,
                    "actions": current_actions,
                    "session_id_before": turn_record.get("session_id_before"),
                    "session_id_after": target.resume_id,
                    "target_execution_prompt_applied": target.target_execution_prompt_applied,
                    "home_snapshot_saved": snapshot is not None,
                    "codex_home_snapshot_saved": snapshot is not None,
                    "agent_backend": target.backend,
                    "restore": restore_result,
                    "replayed_action_count": len(self.formal_state.get("actions") or []),
                    "replay_status": summarize_replay_results(replay_results),
                }
            )
            return response


def summarize_replay_results(replay_results: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"ok": 0, "error": 0, "skipped": 0}
    for result in replay_results:
        status = str(result.get("status") or "skipped")
        if status not in summary:
            status = "skipped"
        summary[status] += 1
    return summary


def run_official_accepted_turn(
    *,
    item: dict[str, Any],
    args: argparse.Namespace,
    accepted_prompts: list[str],
    attacker_message: str,
    official_turn: int,
    strategy_meta: dict[str, Any],
) -> dict[str, Any]:
    with VrapCodexTarget(item, args) as target:
        for accepted_prompt in accepted_prompts:
            target.generate_response(accepted_prompt)
        response = target.generate_response(attacker_message)
        input_output = clean_input_output(target.input_output[-1:] if target.input_output else [])
        turns = with_strategy_meta(
            rewrite_turn_numbers(target.turn_results[-1:] if target.turn_results else [], official_turn),
            strategy_meta,
        )
        for turn in turns:
            turn["replayed_turns"] = len(accepted_prompts)
        actions = with_strategy_meta(
            rewrite_turn_numbers(current_turn_actions(target), official_turn),
            strategy_meta,
        )
        return {
            "response": response,
            "input_output": input_output,
            "turns": turns,
            "actions": actions,
            "model_output": {
                "text": response,
                "returncode": target.turn_results[-1]["returncode"] if target.turn_results else None,
                "error": target.turn_results[-1]["error"] if target.turn_results else None,
            },
        }


# Backward-compatible aliases for callers that still use TextGrad terminology.
IsolatedTextGradTrialTarget = ReplayTrialTarget
SnapshotPromotedTextGradTrialTarget = SnapshotPromotedTrialTarget
