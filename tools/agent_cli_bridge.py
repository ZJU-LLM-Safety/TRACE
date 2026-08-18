"""Small Python bridge for running Codex and Claude Code from code.

The module supports two common layouts:

1. In-container/direct use: the Python process runs in a Docker container
   that already has `codex` and/or `claude` installed. Call `run_codex(...)`
   or `run_claude(...)` without `docker=...`.
2. Host-spawned container use: the Python process runs on the host and starts
   a container that has the CLI installed. Pass a `DockerOptions(image=...)`.

The defaults are intentionally conservative. For fully isolated benchmark
containers, enable the bypass flags explicitly through the options.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, is_dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Literal


Backend = Literal["codex", "claude"]

CODEX_NOISE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T[^\s]+Z ERROR codex_core::util: "
    r"ReasoningSummary(?:Delta|PartAdded) without active item$"
)

PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


CODEX_ENV_KEYS = (
    "CODEX_HOME",
    "CODEX_API_KEY_ENV",
    "CODEX_RESPONSES_BASE_URL",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "MOONBRIDGE_API_KEY",
    "ROUTER_API_KEY",
    *PROXY_ENV_KEYS,
)

CLAUDE_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "VICTIM_API_KEY",
    "VICTIM_AUTH_TOKEN",
    "VICTIM_BASE_URL",
    "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK",
    "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT",
    *PROXY_ENV_KEYS,
)


@dataclass(slots=True)
class DockerOptions:
    """Options for running an agent CLI inside a Docker image."""

    image: str
    cpus: float = 2.0
    memory: str = "4g"
    workdir: str = "/work"
    work_tmpfs_size: str = "2g"
    mount_cwd: bool = False
    add_host_gateway: bool = True
    override_entrypoint: bool = True
    user: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CodexOptions:
    """Codex CLI options."""

    model: str | None = None
    codex_home: str | None = None
    json_events: bool = True
    skip_git_repo_check: bool = True
    bypass_approvals_and_sandbox: bool = False
    disable_features: list[str] = field(default_factory=list)
    resume_thread_id: str | None = None
    write_responses_config: bool = False
    responses_base_url: str | None = None
    responses_env_key: str = "OPENAI_API_KEY"
    responses_provider_name: str = "custom"


@dataclass(slots=True)
class ClaudeOptions:
    """Claude Code SDK options."""

    model: str | None = None
    use_sdk: bool = True
    bypass_permissions: bool = False
    max_turns: int | None = None
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    system_prompt: str | None = None
    claude_home: str | None = None
    resume_session_id: str | None = None
    continue_conversation: bool = False
    fork_session: bool = False


@dataclass(slots=True)
class AgentResult:
    backend: Backend
    command: list[str]
    returncode: int
    text: str
    stdout: str
    stderr: str
    events: list[dict[str, Any]] = field(default_factory=list)
    parsed: Any | None = None
    thread_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_codex(
    prompt: str,
    *,
    options: CodexOptions | None = None,
    docker: DockerOptions | None = None,
    timeout: int = 600,
    cwd: str | Path | None = None,
) -> AgentResult:
    """Run one Codex turn and return parsed text/events.

    For multi-turn Codex sessions, pass `options.resume_thread_id` from the
    previous `AgentResult.thread_id`.
    """

    opts = options or CodexOptions()
    env = os.environ.copy()
    codex_config_home: Path | None = None
    if opts.codex_home:
        env["CODEX_HOME"] = opts.codex_home
    elif docker is None:
        env.setdefault("CODEX_HOME", _ensure_tmp_home("codex_home"))
    else:
        env.setdefault("CODEX_HOME", "/home/agent/codex_home")

    if opts.write_responses_config:
        if docker is not None:
            codex_config_home = Path(
                opts.codex_home or _ensure_tmp_home("codex_home")
            ).resolve()
            env["CODEX_HOME"] = "/codex_home"
            docker = _with_extra_docker_args(
                docker, ["-v", f"{codex_config_home}:/codex_home"]
            )
        else:
            codex_config_home = Path(env["CODEX_HOME"])
        _write_codex_responses_config(
            codex_config_home,
            model=opts.model,
            base_url=opts.responses_base_url,
            env_key=opts.responses_env_key,
            provider_name=opts.responses_provider_name,
        )

    cmd = ["codex", "exec"]
    if opts.resume_thread_id:
        cmd += ["resume", opts.resume_thread_id]
    if opts.model:
        cmd += ["--model", opts.model]
    if opts.json_events:
        cmd.append("--json")
    if opts.skip_git_repo_check:
        cmd.append("--skip-git-repo-check")
    if opts.bypass_approvals_and_sandbox:
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    for feature in opts.disable_features:
        cmd += ["--disable", feature]
    cmd.append("-")

    proc = _run(
        cmd,
        env=env,
        docker=docker,
        timeout=timeout,
        cwd=cwd,
        env_keys=CODEX_ENV_KEYS,
        input_text=prompt,
    )
    events = _parse_jsonl(proc.stdout) if opts.json_events else []
    text = _extract_codex_text(events) if events else proc.stdout.strip()
    thread_id = _extract_codex_thread_id(events)
    return AgentResult(
        backend="codex",
        command=proc.args if isinstance(proc.args, list) else cmd,
        returncode=proc.returncode,
        text=text,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        events=events,
        parsed=events,
        thread_id=thread_id,
    )


def run_claude(
    prompt: str,
    *,
    options: ClaudeOptions | None = None,
    docker: DockerOptions | None = None,
    timeout: int = 600,
    cwd: str | Path | None = None,
) -> AgentResult:
    """Run one Claude Code turn.

    The default path uses `claude-agent-sdk`, matching the
    Auto-research-red-teaming Claude Code runner. Set
    `ClaudeOptions(use_sdk=False)` to use the simple `claude -p` CLI fallback.
    """

    opts = options or ClaudeOptions()
    if not opts.use_sdk:
        return run_claude_cli(
            prompt,
            options=opts,
            docker=docker,
            timeout=timeout,
            cwd=cwd,
        )

    env = os.environ.copy()
    if opts.claude_home:
        env["HOME"] = opts.claude_home
    elif docker is None:
        env["HOME"] = _ensure_writable_home("claude_home")
    else:
        env.setdefault("HOME", "/home/agent")

    env.setdefault("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    env.setdefault("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "180000")

    if docker is not None:
        return _run_claude_sdk_in_docker(
            prompt, options=opts, docker=docker, env=env, timeout=timeout, cwd=cwd
        )

    try:
        return asyncio.run(
            asyncio.wait_for(
                _run_claude_sdk_local(prompt, options=opts, env=env, cwd=cwd),
                timeout=timeout,
            )
        )
    except asyncio.TimeoutError:
        return AgentResult(
            backend="claude",
            command=["claude-agent-sdk"],
            returncode=-1,
            text="",
            stdout="",
            stderr=f"claude-agent-sdk timed out after {timeout}s",
        )


def run_claude_cli(
    prompt: str,
    *,
    options: ClaudeOptions | None = None,
    docker: DockerOptions | None = None,
    timeout: int = 600,
    cwd: str | Path | None = None,
) -> AgentResult:
    """Run one Claude Code print-mode CLI turn and return parsed text/output."""

    opts = options or ClaudeOptions(use_sdk=False)
    env = os.environ.copy()
    if opts.claude_home:
        env["HOME"] = opts.claude_home
    elif docker is None:
        env["HOME"] = _ensure_writable_home("claude_home")
    else:
        env.setdefault("HOME", "/home/agent")

    env.setdefault("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    env.setdefault("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "180000")

    cmd = ["claude", "-p", prompt]
    if opts.model:
        cmd += ["--model", opts.model]
    cmd += ["--output-format", "json"]
    if opts.bypass_permissions:
        cmd.append("--dangerously-skip-permissions")
    if opts.max_turns is not None:
        cmd += ["--max-turns", str(opts.max_turns)]
    for tool in opts.allowed_tools:
        cmd += ["--allowedTools", tool]
    for tool in opts.disallowed_tools:
        cmd += ["--disallowedTools", tool]
    if opts.system_prompt:
        cmd += ["--append-system-prompt", opts.system_prompt]

    proc = _run(
        cmd,
        env=env,
        docker=docker,
        timeout=timeout,
        cwd=cwd,
        env_keys=CLAUDE_ENV_KEYS,
    )
    parsed = _try_json(proc.stdout)
    text = _extract_claude_text(parsed)
    if not text:
        text = (proc.stdout or "").strip()
    return AgentResult(
        backend="claude",
        command=proc.args if isinstance(proc.args, list) else cmd,
        returncode=proc.returncode,
        text=text,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        parsed=parsed,
    )


async def _run_claude_sdk_local(
    prompt: str,
    *,
    options: ClaudeOptions,
    env: dict[str, str],
    cwd: str | Path | None,
) -> AgentResult:
    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                ClaudeSDKClient,
                ResultMessage,
                TextBlock,
                ToolUseBlock,
            )
        except ImportError as exc:
            return AgentResult(
                backend="claude",
                command=["claude-agent-sdk"],
                returncode=127,
                text="",
                stdout="",
                stderr=(
                    "claude-agent-sdk is not installed. Install it in the "
                    "current Python environment or run with "
                    "ClaudeOptions(use_sdk=False). "
                    f"ImportError: {exc}"
                ),
            )

        stderr_lines: list[str] = []

        def stderr_sink(line: str) -> None:
            stderr_lines.append(line)

        kwargs: dict[str, Any] = {
            "cwd": str(cwd or Path.cwd()),
            "stderr": stderr_sink,
        }
        if options.model:
            kwargs["model"] = options.model
        if options.bypass_permissions:
            kwargs["permission_mode"] = "bypassPermissions"
        if options.max_turns is not None:
            kwargs["max_turns"] = options.max_turns
        if options.allowed_tools:
            kwargs["allowed_tools"] = list(options.allowed_tools)
        if options.disallowed_tools:
            kwargs["disallowed_tools"] = list(options.disallowed_tools)
        if options.system_prompt:
            kwargs["system_prompt"] = options.system_prompt
        if options.resume_session_id:
            kwargs["resume"] = options.resume_session_id
        if options.continue_conversation:
            kwargs["continue_conversation"] = True
        if options.fork_session:
            kwargs["fork_session"] = True

        final_text = ""
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tool_calls: list[dict[str, Any]] = []
        result_meta: dict[str, Any] = {}
        error: str | None = None

        try:
            async with ClaudeSDKClient(options=ClaudeAgentOptions(**kwargs)) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    mdict = _safe_model_dump(message)
                    if mdict is not None:
                        messages.append(mdict)
                    if isinstance(message, AssistantMessage):
                        for block in message.content or []:
                            if isinstance(block, TextBlock):
                                final_text = block.text or final_text
                            elif isinstance(block, ToolUseBlock):
                                tool_calls.append(_safe_model_dump(block) or {})
                    if isinstance(message, ResultMessage):
                        result_meta = _safe_model_dump(message) or {}
                        break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        parsed = {
            "messages": messages,
            "tool_calls": tool_calls,
            "result_meta": result_meta,
            "error": error,
        }
        stdout = json.dumps(parsed, ensure_ascii=False)
        return AgentResult(
            backend="claude",
            command=["claude-agent-sdk"],
            returncode=0 if error is None else 1,
            text=final_text,
            stdout=stdout,
            stderr="\n".join(stderr_lines),
            parsed=parsed,
            thread_id=str(result_meta.get("session_id") or "") or None,
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def _run_claude_sdk_in_docker(
    prompt: str,
    *,
    options: ClaudeOptions,
    docker: DockerOptions,
    env: dict[str, str],
    timeout: int,
    cwd: str | Path | None,
) -> AgentResult:
    with tempfile.TemporaryDirectory(prefix="claude_sdk_bridge_") as tmp:
        runner_path = Path(tmp) / "claude_sdk_runner.py"
        runner_path.write_text(_CLAUDE_SDK_RUNNER)
        docker_for_sdk = _with_extra_docker_args(
            docker,
            ["-v", f"{tmp}:/bridge:ro"],
        )
        cmd = _docker_command(
            ["python3", "/bridge/claude_sdk_runner.py"],
            docker=docker_for_sdk,
            env=env,
            env_keys=CLAUDE_ENV_KEYS + ("HOME",),
        )
        payload = {
            "prompt": prompt,
            "model": options.model,
            "bypass_permissions": options.bypass_permissions,
            "max_turns": options.max_turns,
            "allowed_tools": options.allowed_tools,
            "disallowed_tools": options.disallowed_tools,
            "system_prompt": options.system_prompt,
            "resume_session_id": options.resume_session_id,
            "continue_conversation": options.continue_conversation,
            "fork_session": options.fork_session,
            "cwd": docker.workdir,
        }
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload, ensure_ascii=False),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    parsed = _try_json(proc.stdout)
    text = _extract_claude_text(parsed)
    if isinstance(parsed, dict) and not text:
        text = str(parsed.get("final_text") or "")
    return AgentResult(
        backend="claude",
        command=proc.args if isinstance(proc.args, list) else cmd,
        returncode=proc.returncode,
        text=text,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        parsed=parsed,
        thread_id=(
            str((parsed.get("result_meta") or {}).get("session_id") or "")
            if isinstance(parsed, dict)
            else None
        ) or None,
    )


def run_agent(
    backend: Backend,
    prompt: str,
    *,
    codex_options: CodexOptions | None = None,
    claude_options: ClaudeOptions | None = None,
    docker: DockerOptions | None = None,
    timeout: int = 600,
    cwd: str | Path | None = None,
) -> AgentResult:
    """Dispatch to Codex or Claude Code."""

    if backend == "codex":
        return run_codex(
            prompt,
            options=codex_options,
            docker=docker,
            timeout=timeout,
            cwd=cwd,
        )
    if backend == "claude":
        return run_claude(
            prompt,
            options=claude_options,
            docker=docker,
            timeout=timeout,
            cwd=cwd,
        )
    raise ValueError(f"unknown backend: {backend!r}")


def _run(
    cmd: list[str],
    *,
    env: dict[str, str],
    docker: DockerOptions | None,
    timeout: int,
    cwd: str | Path | None,
    env_keys: Iterable[str],
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if docker is None:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    docker_cmd = _docker_command(cmd, docker=docker, env=env, env_keys=env_keys)
    return subprocess.run(
        docker_cmd,
        cwd=str(cwd) if cwd is not None else None,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _docker_command(
    inner_cmd: list[str],
    *,
    docker: DockerOptions,
    env: dict[str, str],
    env_keys: Iterable[str],
) -> list[str]:
    cmd = [
        "docker", "run", "--rm",
        "--cpus", str(docker.cpus),
        "--memory", docker.memory,
        "--workdir", docker.workdir,
    ]
    if docker.add_host_gateway:
        cmd += ["--add-host", "host.docker.internal:host-gateway"]
    if docker.user:
        cmd += ["--user", docker.user]
    if docker.mount_cwd:
        cmd += ["-v", f"{Path.cwd()}:{docker.workdir}"]
    else:
        cmd += ["--tmpfs", f"{docker.workdir}:rw,size={docker.work_tmpfs_size},mode=1777"]

    merged_env = {key: env[key] for key in env_keys if env.get(key)}
    merged_env.update(docker.extra_env)
    for key, value in merged_env.items():
        cmd += ["-e", f"{key}={value}"]

    cmd += docker.extra_args
    if docker.override_entrypoint and inner_cmd:
        cmd += ["--entrypoint", inner_cmd[0], docker.image]
        cmd += inner_cmd[1:]
    else:
        cmd.append(docker.image)
        cmd += inner_cmd
    return cmd


def _with_extra_docker_args(docker: DockerOptions, extra_args: list[str]) -> DockerOptions:
    return DockerOptions(
        image=docker.image,
        cpus=docker.cpus,
        memory=docker.memory,
        workdir=docker.workdir,
        work_tmpfs_size=docker.work_tmpfs_size,
        mount_cwd=docker.mount_cwd,
        add_host_gateway=docker.add_host_gateway,
        override_entrypoint=docker.override_entrypoint,
        user=docker.user,
        extra_env=dict(docker.extra_env),
        extra_args=[*docker.extra_args, *extra_args],
    )


def _make_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _make_jsonable(value.model_dump(mode="json"))
        except TypeError:
            return _make_jsonable(value.model_dump())
    if is_dataclass(value):
        return _make_jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        return {
            key: _make_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _safe_model_dump(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    dumped = _make_jsonable(obj)
    return dumped if isinstance(dumped, dict) else None


def _ensure_tmp_home(name: str) -> str:
    root = Path(tempfile.gettempdir()) / name
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _ensure_writable_home(fallback_name: str) -> str:
    home = Path(os.environ.get("HOME") or "")
    if home and _is_writable_dir(home):
        return str(home)
    return _ensure_tmp_home(fallback_name)


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".agent_cli_bridge_write_test"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _write_codex_responses_config(
    codex_home: Path,
    *,
    model: str | None,
    base_url: str | None,
    env_key: str,
    provider_name: str,
) -> Path:
    """Write a minimal Codex Responses provider config.

    This is useful in benchmark containers that route Codex through a local
    OpenAI-compatible Responses endpoint such as MoonBridge/OpenRouter.
    """

    codex_home.mkdir(parents=True, exist_ok=True)
    selected_model = model or os.environ.get("CODEX_MODEL") or "gpt-5"
    selected_base_url = (
        base_url
        or os.environ.get("CODEX_RESPONSES_BASE_URL")
        or "https://api.openai.com/v1"
    )
    config = f'''model = "{_toml_escape(selected_model)}"
model_provider = "{_toml_escape(provider_name)}"
web_search = "disabled"

[model_providers.{provider_name}]
name = "{_toml_escape(provider_name)}"
base_url = "{_toml_escape(selected_base_url)}"
env_key = "{_toml_escape(env_key)}"
wire_api = "responses"
'''
    path = codex_home / "config.toml"
    path.write_text(config)
    return path


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _parse_jsonl(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _try_json(stdout: str) -> Any | None:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_codex_thread_id(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("type") == "thread.started" and event.get("thread_id"):
            return str(event["thread_id"])
    return None


def _extract_codex_text(events: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") if isinstance(event.get("item"), dict) else event
        if item.get("type") == "agent_message" and item.get("text"):
            chunks.append(str(item["text"]))
    return "\n\n".join(chunks).strip()


def _extract_codex_error(events: list[dict[str, Any]]) -> str:
    messages: list[str] = []
    for event in events:
        if event.get("type") == "error" and event.get("message"):
            messages.append(str(event["message"]))
        if event.get("type") == "turn.failed":
            error = event.get("error")
            if isinstance(error, dict) and error.get("message"):
                messages.append(str(error["message"]))
            elif error:
                messages.append(str(error))
    return "\n".join(messages).strip()


def clean_codex_noise_from_new_records(path: Path, start_line: int) -> None:
    """Remove known Codex stderr noise from newly appended JSON/JSONL records."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rewritten: list[str] = []
    changed = False
    for index, line in enumerate(lines):
        if index < start_line or not line.strip():
            rewritten.append(line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            rewritten.append(line)
            continue
        if isinstance(record, dict):
            cleaned, removed = clean_codex_noise(record)
            changed = changed or removed > 0
            rewritten.append(json.dumps(cleaned, ensure_ascii=False))
        else:
            rewritten.append(line)
    if changed:
        path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")


def clean_codex_noise(value: Any) -> tuple[Any, int]:
    """Recursively remove noisy Codex ReasoningSummary stderr lines."""
    if isinstance(value, dict):
        removed_total = 0
        cleaned = {}
        for key, item in value.items():
            if key in {"error", "stderr"} and isinstance(item, str):
                cleaned_text, removed = clean_codex_noise_text(item)
                removed_total += removed
                cleaned[key] = cleaned_text if cleaned_text else None
            else:
                cleaned_item, removed = clean_codex_noise(item)
                removed_total += removed
                cleaned[key] = cleaned_item
        return cleaned, removed_total
    if isinstance(value, list):
        removed_total = 0
        cleaned_list = []
        for item in value:
            cleaned_item, removed = clean_codex_noise(item)
            removed_total += removed
            cleaned_list.append(cleaned_item)
        return cleaned_list, removed_total
    return value, 0


def clean_codex_noise_text(text: str) -> tuple[str, int]:
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if CODEX_NOISE_RE.match(line.strip()):
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept).strip(), removed


def _extract_claude_text(parsed: Any) -> str:
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, list):
        parts = [_extract_claude_text(item) for item in parsed]
        return "\n".join(part for part in parts if part).strip()
    if not isinstance(parsed, dict):
        return ""

    for key in ("result", "text", "content", "message"):
        value = parsed.get(key)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            extracted = _extract_claude_text(value)
            if extracted:
                return extracted
        if isinstance(value, dict):
            extracted = _extract_claude_text(value)
            if extracted:
                return extracted
    return ""


_CLAUDE_SDK_RUNNER = r'''
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)


def make_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return make_jsonable(value.model_dump(mode="json"))
        except TypeError:
            return make_jsonable(value.model_dump())
    if is_dataclass(value):
        return make_jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        return {
            key: make_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def safe_model_dump(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    dumped = make_jsonable(obj)
    return dumped if isinstance(dumped, dict) else None


async def main() -> int:
    payload = json.loads(sys.stdin.read())
    prompt = payload["prompt"]
    stderr_lines: list[str] = []

    def stderr_sink(line: str) -> None:
        stderr_lines.append(line)
        print(f"[claude-cli-stderr] {line}", file=sys.stderr, flush=True)

    kwargs: dict[str, Any] = {
        "cwd": payload.get("cwd") or os.getcwd(),
        "stderr": stderr_sink,
    }
    if payload.get("model"):
        kwargs["model"] = payload["model"]
    if payload.get("bypass_permissions"):
        kwargs["permission_mode"] = "bypassPermissions"
    if payload.get("max_turns") is not None:
        kwargs["max_turns"] = payload["max_turns"]
    if payload.get("allowed_tools"):
        kwargs["allowed_tools"] = list(payload["allowed_tools"])
    if payload.get("disallowed_tools"):
        kwargs["disallowed_tools"] = list(payload["disallowed_tools"])
    if payload.get("system_prompt"):
        kwargs["system_prompt"] = payload["system_prompt"]
    if payload.get("resume_session_id"):
        kwargs["resume"] = payload["resume_session_id"]
    if payload.get("continue_conversation"):
        kwargs["continue_conversation"] = True
    if payload.get("fork_session"):
        kwargs["fork_session"] = True

    final_text = ""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tool_calls: list[dict[str, Any]] = []
    result_meta: dict[str, Any] = {}
    error: str | None = None

    try:
        async with ClaudeSDKClient(options=ClaudeAgentOptions(**kwargs)) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                mdict = safe_model_dump(message)
                if mdict is not None:
                    messages.append(mdict)
                if isinstance(message, AssistantMessage):
                    for block in message.content or []:
                        if isinstance(block, TextBlock):
                            final_text = block.text or final_text
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append(safe_model_dump(block) or {})
                if isinstance(message, ResultMessage):
                    result_meta = safe_model_dump(message) or {}
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    out = {
        "final_text": final_text,
        "messages": messages,
        "tool_calls": tool_calls,
        "result_meta": result_meta,
        "stderr": "\n".join(stderr_lines),
        "error": error,
    }
    print(json.dumps(make_jsonable(out), ensure_ascii=False))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
'''


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Codex or Claude Code from Python-friendly CLI wrappers."
    )
    parser.add_argument("backend", choices=("codex", "claude"))
    parser.add_argument("prompt")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--docker-image")
    parser.add_argument("--mount-cwd", action="store_true")
    parser.add_argument("--memory", default="4g")
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument("--bypass", action="store_true")
    parser.add_argument("--codex-home")
    parser.add_argument("--claude-home")
    parser.add_argument("--claude-cli", action="store_true")
    parser.add_argument("--write-codex-config", action="store_true")
    parser.add_argument("--responses-base-url")
    parser.add_argument("--responses-env-key", default="OPENAI_API_KEY")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    docker = None
    if args.docker_image:
        docker = DockerOptions(
            image=args.docker_image,
            cpus=args.cpus,
            memory=args.memory,
            mount_cwd=args.mount_cwd,
        )

    if args.backend == "codex":
        result = run_codex(
            args.prompt,
            options=CodexOptions(
                model=args.model,
                codex_home=args.codex_home,
                bypass_approvals_and_sandbox=args.bypass,
                write_responses_config=args.write_codex_config,
                responses_base_url=args.responses_base_url,
                responses_env_key=args.responses_env_key,
            ),
            docker=docker,
            timeout=args.timeout,
        )
    else:
        result = run_claude(
            args.prompt,
            options=ClaudeOptions(
                model=args.model,
                use_sdk=not args.claude_cli,
                claude_home=args.claude_home,
                bypass_permissions=args.bypass,
            ),
            docker=docker,
            timeout=args.timeout,
        )

    payload = {
        "backend": result.backend,
        "ok": result.ok,
        "returncode": result.returncode,
        "text": result.text,
        "thread_id": result.thread_id,
        "stderr_tail": result.stderr[-2000:],
        "command": " ".join(shlex.quote(part) for part in result.command),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
