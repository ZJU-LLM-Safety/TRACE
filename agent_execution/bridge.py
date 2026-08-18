"""Shared VRAP/Codex bridge imports and adapter defaults."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

TRACE_ROOT = Path(__file__).resolve().parents[1]
TRACE_TOOLS = TRACE_ROOT / "tools"
if str(TRACE_TOOLS) not in sys.path:
    sys.path.insert(0, str(TRACE_TOOLS))

from agent_cli_bridge import clean_codex_noise  # noqa: E402
from vrap_bridge_runner import (  # noqa: E402
    CLIENT_CONTAINER,
    DEFAULT_CLAUDE_HOME,
    DEFAULT_COMPOSE,
    SERVER_CONTAINER,
    build_input_output_pairs,
    cleanup_environment,
    docker_exec,
    prepare_compose,
    read_dataset,
    run_claude_in_vrap as _run_claude_in_vrap,
    run_codex_in_vrap as _run_codex_in_vrap,
    selected_items,
    setup_environment,
    verify_item,
)

DEFAULT_ATTACKER_MODEL = os.environ.get("QWEN32B", "Qwen/Qwen2.5-32B-Instruct")
DEFAULT_CODEX_MODEL = "deepseek-v4-pro"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-6"
DEFAULT_RESPONSES_BASE_URL = "http://192.168.2.1:38440/v1"
DEFAULT_ANTHROPIC_BASE_URL = "http://192.168.2.1:38445"
DEFAULT_OUTPUT_ROOT = Path("agent_execution_outputs")
DEFAULT_CONTAINER_CODEX_HOME = "/home/agent/codex_home"
DEFAULT_CONTAINER_CLAUDE_HOME = DEFAULT_CLAUDE_HOME

NETWORK_RETRY_SLEEP_SECONDS = 5
DEFAULT_CODEX_NETWORK_RETRIES = 9
DEFAULT_CODEX_SAFETY_RETRIES = 2
DEFAULT_CLAUDE_NETWORK_RETRIES = 9
DEFAULT_CLAUDE_SAFETY_RETRIES = 2

NETWORK_ERROR_PATTERNS = (
    "reconnecting",
    "stream disconnected",
    "error sending request",
    "connection reset",
    "connection refused",
    "connection aborted",
    "connection closed",
    "connection timed out",
    "timed out",
    "temporary failure",
    "dns",
    "network",
    "tls",
    "http2",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
)

NON_RETRYABLE_ERROR_PATTERNS = (
    "i can't help",
    "i can’t help",
    "policy",
)

SAFETY_BLOCK_PATTERNS = (
    "flagged for possible cybersecurity risk",
)


def run_codex_in_vrap(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run Codex with retries for transient network failures and safety blocks."""
    network_retries = _retry_count(
        "CODEX_NETWORK_RETRIES", DEFAULT_CODEX_NETWORK_RETRIES
    )
    safety_retries = _retry_count(
        "CODEX_SAFETY_RETRIES", DEFAULT_CODEX_SAFETY_RETRIES
    )
    return _run_with_agent_retries(
        "Codex",
        _run_codex_in_vrap,
        codex_retry_reason,
        network_retries,
        safety_retries,
        *args,
        **kwargs,
    )


def run_claude_in_vrap(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run Claude Code with retries for transient network failures and safety blocks."""
    network_retries = _retry_count(
        "CLAUDE_NETWORK_RETRIES", DEFAULT_CLAUDE_NETWORK_RETRIES
    )
    safety_retries = _retry_count(
        "CLAUDE_SAFETY_RETRIES", DEFAULT_CLAUDE_SAFETY_RETRIES
    )
    return _run_with_agent_retries(
        "Claude",
        _run_claude_in_vrap,
        claude_retry_reason,
        network_retries,
        safety_retries,
        *args,
        **kwargs,
    )


def _retry_count(name: str, default: int) -> int:
    return int(
        os.environ.get(
            f"AGENT_EXECUTION_{name}",
            os.environ.get(f"XTEAMING_{name}", default),
        )
    )


def _run_with_agent_retries(
    label: str,
    runner: Any,
    retry_reason_fn: Any,
    network_retries: int,
    safety_retries: int,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    max_retries = max(network_retries, safety_retries)
    attempts: list[dict[str, Any]] = []

    for attempt in range(max_retries + 1):
        result = runner(*args, **kwargs)
        retry_reason = retry_reason_fn(result)
        retry_limit = safety_retries if retry_reason == "safety_block" else network_retries
        retryable = retry_reason is not None and attempt < retry_limit
        attempts.append(
            {
                "attempt": attempt + 1,
                "returncode": result.get("returncode"),
                "retry_reason": retry_reason,
                "retryable": retryable,
                "error": result.get("error"),
            }
        )
        if not retryable:
            if len(attempts) > 1:
                result["retry_attempts"] = attempts
            return result

        print(
            f"{label} {retry_reason} detected; "
            f"sleeping {NETWORK_RETRY_SLEEP_SECONDS}s before retry "
            f"{attempt + 2}/{retry_limit + 1}.",
            flush=True,
        )
        time.sleep(NETWORK_RETRY_SLEEP_SECONDS)

    raise AssertionError("unreachable")


def is_retryable_codex_network_error(result: dict[str, Any]) -> bool:
    return codex_retry_reason(result) == "network_error"


def is_retryable_claude_network_error(result: dict[str, Any]) -> bool:
    return claude_retry_reason(result) == "network_error"


def codex_retry_reason(result: dict[str, Any]) -> str | None:
    return _agent_retry_reason(result)


def claude_retry_reason(result: dict[str, Any]) -> str | None:
    return _agent_retry_reason(result)


def _agent_retry_reason(result: dict[str, Any]) -> str | None:
    if result.get("returncode") == 0:
        return None
    combined = "\n".join(
        str(result.get(key) or "") for key in ("error", "text", "debug")
    ).lower()
    if not combined:
        return None
    if any(pattern in combined for pattern in SAFETY_BLOCK_PATTERNS):
        return "safety_block"
    if any(pattern in combined for pattern in NON_RETRYABLE_ERROR_PATTERNS):
        return None
    if any(pattern in combined for pattern in NETWORK_ERROR_PATTERNS):
        return "network_error"
    return None
