"""Shared VRAP action parsing and replay helpers."""
from __future__ import annotations

import re
import subprocess
from typing import Any

from vrap_bridge_runner import CLIENT_CONTAINER, docker_exec


def redact_secrets(text: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", text)


def action_to_command(action: dict[str, Any]) -> str | None:
    instruction = action.get("instruction")
    if action.get("type") == "command":
        if isinstance(instruction, str):
            return instruction
        if isinstance(instruction, list):
            return " ".join(str(part) for part in instruction)
        if isinstance(instruction, dict):
            for key in ("command", "cmd", "input", "script"):
                value = instruction.get(key)
                if isinstance(value, str):
                    return value
            args = instruction.get("arguments")
            if isinstance(args, dict) and isinstance(args.get("command"), str):
                return args["command"]
        return None

    if action.get("type") == "tool" and isinstance(instruction, dict):
        name = str(instruction.get("name") or instruction.get("tool") or "").lower()
        tool_input = instruction.get("input") or instruction.get("arguments") or {}
        if name in {"bash", "shell", "execute_bash"}:
            if isinstance(tool_input, str):
                return tool_input
            if isinstance(tool_input, dict) and isinstance(tool_input.get("command"), str):
                return tool_input["command"]
    return None


def action_tool(action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    instruction = action.get("instruction")
    if action.get("type") != "tool" or not isinstance(instruction, dict):
        return "", {}
    name = str(instruction.get("name") or instruction.get("tool") or "").lower()
    tool_input = instruction.get("input") or instruction.get("arguments") or {}
    return name, tool_input if isinstance(tool_input, dict) else {}


def _timeout_record(
    *,
    index: int,
    action: dict[str, Any],
    instruction: Any,
    timeout: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "index": index,
        "type": action.get("type"),
        "instruction": instruction,
        "replayed": True,
        "exit_code": None,
        "result": "",
        "error": f"Replay action timed out after {timeout}s: {error}",
        "timeout": True,
    }


def _completed_record(
    *,
    index: int,
    action: dict[str, Any],
    instruction: Any,
    proc: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    return {
        "index": index,
        "type": action.get("type"),
        "instruction": instruction,
        "replayed": True,
        "exit_code": proc.returncode,
        "result": proc.stdout,
        "error": proc.stderr.strip() or None,
    }


def replay_trajectory_action(
    action: dict[str, Any],
    *,
    index: int,
    user: str,
    timeout: int,
    tolerant_timeouts: bool = False,
    container: str = CLIENT_CONTAINER,
    workdir: str = "/work",
) -> dict[str, Any]:
    """Replay one trajectory action and return a verify-style replay record."""
    tool_name, tool_input = action_tool(action)
    if tool_name == "write":
        file_path = tool_input.get("file_path") or tool_input.get("path")
        content = tool_input.get("content")
        if isinstance(file_path, str) and isinstance(content, str):
            try:
                proc = docker_exec(
                    container,
                    ["bash", "-lc", 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"', "_", file_path],
                    user=user,
                    workdir=workdir,
                    input_text=content,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                if tolerant_timeouts:
                    return _timeout_record(
                        index=index,
                        action=action,
                        instruction=action.get("instruction"),
                        timeout=timeout,
                        error=exc,
                    )
                raise
            return _completed_record(
                index=index,
                action=action,
                instruction=action.get("instruction"),
                proc=proc,
            )

    if tool_name == "read":
        file_path = tool_input.get("file_path") or tool_input.get("path")
        if isinstance(file_path, str):
            try:
                proc = docker_exec(
                    container,
                    ["bash", "-lc", 'cat -- "$1"', "_", file_path],
                    user=user,
                    workdir=workdir,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                if tolerant_timeouts:
                    return _timeout_record(
                        index=index,
                        action=action,
                        instruction=action.get("instruction"),
                        timeout=timeout,
                        error=exc,
                    )
                raise
            return _completed_record(
                index=index,
                action=action,
                instruction=action.get("instruction"),
                proc=proc,
            )

    command = action_to_command(action)
    if not command:
        return {
            "index": index,
            "type": action.get("type"),
            "instruction": action.get("instruction"),
            "replayed": False,
            "reason": "unsupported action shape",
        }

    try:
        proc = docker_exec(
            container,
            ["bash", "-lc", command],
            user=user,
            workdir=workdir,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        if tolerant_timeouts:
            return _timeout_record(
                index=index,
                action=action,
                instruction=command,
                timeout=timeout,
                error=exc,
            )
        raise
    return _completed_record(index=index, action=action, instruction=command, proc=proc)


def replay_trajectory_actions(
    trajectory: dict[str, Any],
    *,
    timeout: int,
    tolerant_timeouts: bool = False,
    container: str = CLIENT_CONTAINER,
    workdir: str = "/work",
) -> list[dict[str, Any]]:
    run = trajectory.get("run") or {}
    user = str(run.get("agent_user") or run.get("attacker_identity") or "root")
    return [
        replay_trajectory_action(
            action,
            index=index,
            user=user,
            timeout=timeout,
            tolerant_timeouts=tolerant_timeouts,
            container=container,
            workdir=workdir,
        )
        for index, action in enumerate(trajectory.get("actions") or [], 1)
    ]


def shell_quote_single(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_container_text_file(
    file_path: str,
    content: str,
    *,
    user: str,
    container: str = CLIENT_CONTAINER,
) -> dict[str, Any]:
    script = (
        "from pathlib import Path\n"
        "import sys\n"
        "path_text, content = sys.stdin.read().split('\\0', 1)\n"
        "path = Path(path_text)\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_text(content, encoding='utf-8')\n"
    )
    proc = docker_exec(
        container,
        ["python3", "-c", script],
        user=user,
        input_text=f"{file_path}\0{content}",
        timeout=120,
    )
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "exit_code": proc.returncode,
        "error": redact_secrets(proc.stderr.strip() or ""),
    }


def remove_container_path(
    file_path: str,
    *,
    user: str,
    container: str = CLIENT_CONTAINER,
    workdir: str = "/work",
) -> dict[str, Any]:
    proc = docker_exec(
        container,
        ["bash", "-lc", f"rm -rf -- {shell_quote_single(file_path)}"],
        user=user,
        workdir=workdir,
        timeout=120,
    )
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "exit_code": proc.returncode,
        "error": redact_secrets(proc.stderr.strip() or ""),
    }


def replay_state_action(
    action: dict[str, Any],
    *,
    user: str,
    timeout: int = 600,
    container: str = CLIENT_CONTAINER,
    workdir: str = "/work",
) -> dict[str, Any]:
    """Replay one state-restoration action for session/promotion flows."""
    action_type = action.get("type")
    instruction = action.get("instruction")
    if action_type == "command":
        if isinstance(instruction, list):
            inner_cmd = [str(part) for part in instruction]
        elif isinstance(instruction, str):
            inner_cmd = ["bash", "-lc", instruction]
        else:
            return {"status": "skipped", "reason": "unsupported_command_instruction"}
        proc = docker_exec(
            container,
            inner_cmd,
            user=user,
            workdir=workdir,
            timeout=timeout,
        )
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "type": "command",
            "exit_code": proc.returncode,
            "error": redact_secrets(proc.stderr.strip() or ""),
        }

    if action_type == "tool" and isinstance(instruction, dict):
        tool_input = instruction.get("input") if isinstance(instruction.get("input"), dict) else {}
        name = instruction.get("name")
        file_path = tool_input.get("file_path")
        if name == "write" and isinstance(file_path, str):
            return write_container_text_file(
                file_path,
                str(tool_input.get("content") or ""),
                user=user,
                container=container,
            )

    if action_type == "file_event" and isinstance(instruction, dict):
        file_path = instruction.get("file_path") or instruction.get("path")
        kind = instruction.get("kind")
        if isinstance(file_path, str) and kind in {"delete", "remove"}:
            return remove_container_path(
                file_path,
                user=user,
                container=container,
                workdir=workdir,
            )

    return {"status": "skipped", "reason": f"unsupported_action_type:{action_type}"}


def replay_state_actions(
    actions: list[dict[str, Any]],
    *,
    user: str,
    timeout: int = 600,
    container: str = CLIENT_CONTAINER,
    workdir: str = "/work",
) -> list[dict[str, Any]]:
    replay_results = []
    for index, action in enumerate(actions, 1):
        result = replay_state_action(
            action,
            user=user,
            timeout=timeout,
            container=container,
            workdir=workdir,
        )
        result["replay_index"] = index
        replay_results.append(result)
    return replay_results
