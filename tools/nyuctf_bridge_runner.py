"""Run NYUCTF/CTF dataset items with Codex or Claude Code in Docker.

This is independent dataset glue for the NYUCTF-style CTF tasks. It borrows
the operational shape of the SWE-agent CTF environment without importing the
SWE-agent package:

- start the challenge docker-compose when the item has ``compose: true``;
- start a disposable agent container, attach it to ``attacknet``;
- copy only the challenge files named by the dataset into the agent container;
- run Codex or Claude Code inside that container through CLI/SDK bridges;
- save replayable actions and verify by checking the known ``gold`` value.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import posixpath
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from agent_cli_bridge import (
    CLAUDE_ENV_KEYS,
    CODEX_ENV_KEYS,
    CodexOptions,
    _CLAUDE_SDK_RUNNER,
    _extract_claude_text,
    _extract_codex_error,
    _extract_codex_text,
    _extract_codex_thread_id,
    _parse_jsonl,
    _try_json,
    _write_codex_responses_config,
    clean_codex_noise,
)


Backend = Literal["codex", "claude"]

TRACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = TRACE_ROOT / "data/nyuctf/ctf_test.json"
DEFAULT_NYUCTF_ROOT = TRACE_ROOT / "data/nyuctf/.nyuctf"
DEFAULT_IMAGE = "attackenv:latest"
NETWORK_NAME = "attacknet"
DEFAULT_WORKDIR_PREFIX = "/"


def read_dataset(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"NYUCTF dataset must be a list: {path}")
    return parsed


def build_prompt(item: dict[str, Any], *, field: str = "harmful") -> str:
    if field not in item:
        raise ValueError(f"Missing prompt field {field!r} in item {item_id(item)!r}")
    return str(item[field])


def item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("unique_id") or item.get("challenge") or item.get("path") or "unknown")


def selected_items(data: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ids = {part.strip() for part in args.ids.split(",")} if getattr(args, "ids", None) else None
    start_index = int(getattr(args, "start_index", 0) or 0)
    selected: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if index < start_index:
            continue
        if ids is not None and item_id(item) not in ids:
            continue
        selected.append(item)
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def challenge_workdir(item: dict[str, Any]) -> str:
    name = str(item.get("challenge") or item.get("path") or "challenge")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "challenge"
    if prompt_references_absolute_challenge_file(item):
        return DEFAULT_WORKDIR_PREFIX + f"{safe}_workdir"
    return DEFAULT_WORKDIR_PREFIX + safe


def prompt_references_absolute_challenge_file(item: dict[str, Any]) -> bool:
    return any(
        prompt_references_absolute_file(item, str(file_name))
        for file_name in (item.get("files") or [])
    )


def prompt_references_absolute_file(item: dict[str, Any], file_name: str) -> bool:
    file_name = file_name.lstrip("./")
    if not file_name:
        return False
    files = [str(file_name).lstrip("./") for file_name in (item.get("files") or [])]
    if file_name not in files:
        return False
    prompts = [
        value
        for key, value in item.items()
        if isinstance(value, str) and key in {"harmful", "query", "prompt", "instruction"}
    ]
    return any(f"/{file_name}" in prompt for prompt in prompts)


def challenge_root(item: dict[str, Any], nyuctf_root: Path) -> Path:
    """Resolve a dataset item to the local NYUCTF challenge resource dir."""
    raw_path = str(item.get("path") or "")
    challenge = str(item.get("challenge") or Path(raw_path).name)
    candidates = [
        nyuctf_root / raw_path,
        nyuctf_root / "v20250206" / raw_path,
    ]
    if not raw_path.startswith("v20250206/"):
        candidates.append(nyuctf_root / "v20250206" / "test" / "2023" / "CSAW-Quals" / "pwn" / raw_path)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    matches: list[Path] = []
    for candidate in nyuctf_root.rglob(challenge):
        if candidate.is_dir() and (candidate / "docker-compose.yml").is_file():
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    if matches:
        files = item.get("files") or []
        for candidate in matches:
            if all((candidate / str(file_name)).exists() for file_name in files):
                return candidate
        return matches[0]
    raise FileNotFoundError(
        "Could not resolve NYUCTF challenge resources for "
        f"path={raw_path!r} challenge={challenge!r} under {nyuctf_root}"
    )


def compose_file_for(item: dict[str, Any], root: Path) -> Path | None:
    if not item.get("compose"):
        return None
    compose = root / "docker-compose.yml"
    if not compose.is_file():
        raise FileNotFoundError(f"compose=true but docker-compose.yml not found at {compose}")
    return compose


def run_checked(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def compose_cmd(compose_file: Path, *args: str) -> list[str]:
    return ["docker", "compose", "-f", str(compose_file), *args]


def ensure_network(name: str = NETWORK_NAME) -> None:
    inspect = run_checked(["docker", "network", "inspect", name], timeout=30)
    if inspect.returncode == 0:
        return
    create = run_checked(["docker", "network", "create", name], timeout=60)
    if create.returncode != 0:
        raise RuntimeError(f"docker network create failed: stdout={create.stdout!r} stderr={create.stderr!r}")


def unique_container_name(prefix: str, image_name: str) -> str:
    seed = f"{_dt.datetime.now().isoformat()}:{os.getpid()}:{image_name}"
    digest = hashlib.sha256(seed.encode()).hexdigest()[:10]
    safe_image = re.sub(r"[^A-Za-z0-9_.-]+", "-", image_name)
    return f"{prefix}-{safe_image}-{digest}"[:120]


def docker_exec(
    container: str,
    inner_cmd: list[str],
    *,
    user: str | None = None,
    workdir: str | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "exec", "-i"]
    if user:
        cmd += ["-u", user]
    if workdir:
        cmd += ["-w", workdir]
    for key, value in (env or {}).items():
        if value:
            cmd += ["-e", f"{key}={value}"]
    cmd.append(container)
    cmd += inner_cmd
    return subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def docker_cp(src: Path, container: str, dest: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "cp", str(src), f"{container}:{dest}"],
        capture_output=True,
        text=True,
        check=False,
    )


def start_challenge_compose(compose_file: Path | None) -> None:
    if compose_file is None:
        return
    ensure_network()
    down = run_checked(compose_cmd(compose_file, "down"), cwd=compose_file.parent, timeout=180)
    if down.returncode != 0:
        # docker compose down can fail for old/missing resources; up below is authoritative.
        pass
    up = run_checked(
        compose_cmd(compose_file, "up", "-d", "--force-recreate"),
        cwd=compose_file.parent,
        timeout=300,
    )
    if up.returncode != 0:
        raise RuntimeError(f"docker compose up failed: stdout={up.stdout[-2000:]!r} stderr={up.stderr[-4000:]!r}")
    time.sleep(3)


def stop_challenge_compose(compose_file: Path | None) -> None:
    if compose_file is None:
        return
    run_checked(compose_cmd(compose_file, "down"), cwd=compose_file.parent, timeout=180)


def start_agent_container(
    *,
    image_name: str,
    workdir: str,
    cpus: float,
    memory: str,
    container_name: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    ensure_network()
    name = container_name or unique_container_name("nyuctf-agent", image_name)
    cmd = [
        "docker", "run", "-d",
        "--name", name,
        "--network", NETWORK_NAME,
        "--cpus", str(cpus),
        "--memory", memory,
        "--add-host", "host.docker.internal:host-gateway",
    ]
    for key, value in (extra_env or {}).items():
        if value:
            cmd += ["-e", f"{key}={value}"]
    cmd += [image_name, "sleep", "infinity"]
    proc = run_checked(cmd, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"docker run failed: stdout={proc.stdout!r} stderr={proc.stderr!r}")
    mkdir = docker_exec(name, ["mkdir", "-p", workdir], user="root", timeout=30)
    if mkdir.returncode != 0:
        raise RuntimeError(f"failed to create workdir {workdir}: {mkdir.stderr or mkdir.stdout}")
    return name


def stop_agent_container(container: str) -> None:
    run_checked(["docker", "rm", "-f", container], timeout=60)


def copy_challenge_files(item: dict[str, Any], root: Path, container: str, workdir: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    files = item.get("files") or []
    docker_exec(container, ["mkdir", "-p", workdir], user="root", timeout=30)
    copied_destinations: list[str] = []
    for file_name_raw in files:
        file_name = str(file_name_raw)
        src = root / file_name
        if not src.exists():
            raise FileNotFoundError(f"challenge file not found: {src}")
        if prompt_references_absolute_file(item, file_name):
            dest = posixpath.normpath("/" + file_name.lstrip("./"))
        else:
            dest = posixpath.normpath(posixpath.join(workdir, file_name))
        parent = posixpath.dirname(dest) or "/"
        docker_exec(container, ["mkdir", "-p", parent], user="root", timeout=30)
        proc = docker_cp(src, container, dest)
        if proc.returncode != 0:
            raise RuntimeError(f"docker cp failed for {src}: stdout={proc.stdout!r} stderr={proc.stderr!r}")
        copied_destinations.append(dest)
        steps.append({
            "type": "setup_copy",
            "source": str(src),
            "destination": dest,
            "result": "copied challenge resource into agent container",
        })
    if copied_destinations:
        quoted = " ".join(shlex.quote(dest) for dest in copied_destinations)
        docker_exec(container, ["bash", "-lc", f"chown -R root:root -- {quoted}"], user="root", timeout=30)
    return steps


def setup_environment(
    item: dict[str, Any],
    *,
    nyuctf_root: Path,
    image_name: str,
    cpus: float,
    memory: str,
    container_name: str | None = None,
) -> dict[str, Any]:
    root = challenge_root(item, nyuctf_root)
    compose_file = compose_file_for(item, root)
    workdir = challenge_workdir(item)
    start_challenge_compose(compose_file)
    container = start_agent_container(
        image_name=image_name,
        workdir=workdir,
        cpus=cpus,
        memory=memory,
        container_name=container_name,
    )
    setup_steps = copy_challenge_files(item, root, container, workdir)
    return {
        "container": container,
        "challenge_root": root,
        "compose_file": compose_file,
        "workdir": workdir,
        "setup_steps": setup_steps,
    }


def cleanup_environment(env_info: dict[str, Any]) -> None:
    container = env_info.get("container")
    if container:
        stop_agent_container(str(container))
    stop_challenge_compose(env_info.get("compose_file"))


def collect_env(keys: tuple[str, ...]) -> dict[str, str]:
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def run_codex_in_container(
    prompt: str,
    *,
    container: str,
    workdir: str,
    model: str | None,
    user: str,
    timeout: int,
    bypass: bool,
    write_config: bool,
    responses_base_url: str | None,
    responses_env_key: str,
    resume_thread_id: str | None = None,
) -> dict[str, Any]:
    missing = missing_backend_reason(container, "codex")
    if missing:
        return {
            "backend": "codex",
            "returncode": 127,
            "text": "",
            "error": missing,
            "thread_id": None,
            "actions": [],
            "command": "codex exec",
        }

    env = collect_env(CODEX_ENV_KEYS)
    env.setdefault("CODEX_HOME", "/root/codex_home")
    docker_exec(container, ["mkdir", "-p", env["CODEX_HOME"]], user="root", timeout=30)

    if write_config:
        with tempfile.TemporaryDirectory(prefix="nyuctf_codex_home_") as tmp:
            config_path = _write_codex_responses_config(
                Path(tmp),
                model=model,
                base_url=responses_base_url,
                env_key=responses_env_key,
                provider_name="nyuctf",
            )
            docker_cp(config_path, container, f"{env['CODEX_HOME']}/config.toml")

    opts = CodexOptions(
        model=model,
        bypass_approvals_and_sandbox=bypass,
        json_events=True,
        skip_git_repo_check=True,
    )
    cmd = ["codex", "exec"]
    if resume_thread_id:
        cmd += ["resume", resume_thread_id]
    if opts.model:
        cmd += ["--model", opts.model]
    cmd += ["--json", "--skip-git-repo-check"]
    if opts.bypass_approvals_and_sandbox:
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    cmd.append("-")

    proc = docker_exec(
        container,
        cmd,
        user=user,
        workdir=workdir,
        env=env,
        input_text=prompt,
        timeout=timeout,
    )
    events = _parse_jsonl(proc.stdout)
    actions = extract_codex_actions(events, container=container, workdir=workdir)
    codex_error = proc.stderr.strip() or _extract_codex_error(events) or None
    return {
        "backend": "codex",
        "returncode": proc.returncode,
        "text": _extract_codex_text(events) if events else proc.stdout.strip(),
        "error": codex_error,
        "thread_id": _extract_codex_thread_id(events),
        "actions": actions,
        "command": " ".join(shlex.quote(part) for part in cmd),
    }


def run_claude_in_container(
    prompt: str,
    *,
    container: str,
    workdir: str,
    model: str | None,
    user: str,
    timeout: int,
    bypass: bool,
    max_turns: int | None,
    anthropic_base_url: str | None = None,
    anthropic_env_key: str = "ANTHROPIC_API_KEY",
    allowed_tools: list[str] | None = None,
    resume_session_id: str | None = None,
    fork_session: bool = False,
    claude_runner: str = "sdk",
) -> dict[str, Any]:
    if claude_runner == "cli":
        return run_claude_cli_in_container(
            prompt,
            container=container,
            workdir=workdir,
            model=model,
            user=user,
            timeout=timeout,
            bypass=bypass,
            max_turns=max_turns,
            anthropic_base_url=anthropic_base_url,
            anthropic_env_key=anthropic_env_key,
            allowed_tools=allowed_tools,
            resume_session_id=resume_session_id,
            continue_conversation=bool(resume_session_id),
            fork_session=fork_session,
        )
    if claude_runner != "sdk":
        raise ValueError(f"unsupported Claude runner: {claude_runner!r}")

    claude_python = find_claude_sdk_python(container)
    missing = missing_backend_reason(container, "claude", claude_python=claude_python)
    if missing:
        return {
            "backend": "claude",
            "runner": "sdk",
            "returncode": 127,
            "text": "",
            "error": missing,
            "thread_id": None,
            "actions": [],
            "debug": None,
            "command": "python3 /tmp/agent_cli_bridge/claude_sdk_runner.py",
        }

    env = collect_env(CLAUDE_ENV_KEYS)
    env.setdefault("HOME", "/root" if user == "root" else f"/home/{user}")
    env.setdefault("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    env.setdefault("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "180000")
    if anthropic_base_url:
        env["ANTHROPIC_BASE_URL"] = anthropic_base_url.rstrip("/")
    if anthropic_env_key != "ANTHROPIC_API_KEY" and os.environ.get(anthropic_env_key):
        env["ANTHROPIC_API_KEY"] = os.environ[anthropic_env_key]

    with tempfile.TemporaryDirectory(prefix="nyuctf_claude_sdk_") as tmp:
        runner_path = Path(tmp) / "claude_sdk_runner.py"
        runner_path.write_text(_CLAUDE_SDK_RUNNER)
        docker_exec(container, ["mkdir", "-p", "/tmp/agent_cli_bridge"], user="root", timeout=30)
        docker_cp(runner_path, container, "/tmp/agent_cli_bridge/claude_sdk_runner.py")
        payload = {
            "prompt": prompt,
            "model": model,
            "bypass_permissions": bypass,
            "max_turns": max_turns,
            "allowed_tools": allowed_tools or [],
            "disallowed_tools": [],
            "system_prompt": None,
            "resume_session_id": resume_session_id,
            "continue_conversation": bool(resume_session_id),
            "fork_session": fork_session,
            "cwd": workdir,
        }
        proc = docker_exec(
            container,
            [claude_python or "python3", "/tmp/agent_cli_bridge/claude_sdk_runner.py"],
            user=user,
            workdir=workdir,
            env=env,
            input_text=json.dumps(payload, ensure_ascii=False),
            timeout=timeout,
        )

    parsed = _try_json(proc.stdout)
    text = _extract_claude_text(parsed)
    if isinstance(parsed, dict) and not text:
        text = str(parsed.get("final_text") or "")
    return {
        "backend": "claude",
        "runner": "sdk",
        "returncode": proc.returncode,
        "text": text,
        "error": proc.stderr.strip() or None,
        "thread_id": (
            str((parsed.get("result_meta") or {}).get("session_id") or "")
            if isinstance(parsed, dict)
            else None
        ) or None,
        "actions": extract_claude_actions(parsed),
        "debug": summarize_claude_debug(parsed),
        "command": "python3 /tmp/agent_cli_bridge/claude_sdk_runner.py",
    }


def run_claude_cli_in_container(
    prompt: str,
    *,
    container: str,
    workdir: str,
    model: str | None,
    user: str,
    timeout: int,
    bypass: bool,
    max_turns: int | None,
    anthropic_base_url: str | None = None,
    anthropic_env_key: str = "ANTHROPIC_API_KEY",
    allowed_tools: list[str] | None = None,
    resume_session_id: str | None = None,
    continue_conversation: bool = False,
    fork_session: bool = False,
) -> dict[str, Any]:
    missing = missing_claude_cli_reason(container)
    if missing:
        return {
            "backend": "claude",
            "runner": "cli",
            "returncode": 127,
            "text": "",
            "error": missing,
            "thread_id": None,
            "actions": [],
            "debug": None,
            "command": "claude -p <prompt>",
        }

    env = collect_env(CLAUDE_ENV_KEYS)
    env.setdefault("HOME", "/root" if user == "root" else f"/home/{user}")
    env.setdefault("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    env.setdefault("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "180000")
    if anthropic_base_url:
        env["ANTHROPIC_BASE_URL"] = anthropic_base_url.rstrip("/")
    if anthropic_env_key != "ANTHROPIC_API_KEY" and os.environ.get(anthropic_env_key):
        env["ANTHROPIC_API_KEY"] = os.environ[anthropic_env_key]

    docker_exec(container, ["mkdir", "-p", env["HOME"]], user="root", timeout=30)

    cmd = ["claude", "-p", prompt]
    if model:
        cmd += ["--model", model]
    cmd += ["--output-format", "stream-json", "--verbose"]
    if bypass:
        cmd.append("--dangerously-skip-permissions")
    if max_turns is not None:
        cmd += ["--max-turns", str(max_turns)]
    for tool in allowed_tools or []:
        cmd += ["--allowedTools", tool]
    if resume_session_id:
        cmd += ["--resume", resume_session_id]
    if continue_conversation:
        cmd.append("--continue")
    if fork_session:
        cmd.append("--fork-session")

    proc = docker_exec(
        container,
        cmd,
        user=user,
        workdir=workdir,
        env=env,
        timeout=timeout,
    )
    parsed = parse_claude_cli_stream(proc.stdout)
    text = _extract_claude_text(parsed)
    if isinstance(parsed, dict) and not text:
        text = str((parsed.get("result_meta") or {}).get("result") or "")
    if not text:
        text = proc.stdout.strip()
    return {
        "backend": "claude",
        "runner": "cli",
        "returncode": proc.returncode,
        "text": text,
        "error": proc.stderr.strip() or None,
        "thread_id": (
            str((parsed.get("result_meta") or {}).get("session_id") or "")
            if isinstance(parsed, dict)
            else None
        ) or None,
        "actions": extract_claude_actions(parsed),
        "debug": summarize_claude_debug(parsed),
        "command": " ".join(shlex.quote(part) for part in cmd[:1] + ["-p", "<prompt>"] + cmd[3:]),
    }


def parse_claude_cli_stream(stdout: str) -> dict[str, Any]:
    events = _parse_jsonl(stdout)
    if not events:
        parsed = _try_json(stdout)
        if isinstance(parsed, dict):
            return parsed
        return {"final_text": stdout.strip(), "messages": [], "tool_calls": [], "result_meta": {}}

    messages: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    result_meta: dict[str, Any] = {}
    final_text = ""
    session_id: str | None = None

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("session_id"):
            session_id = str(event.get("session_id"))
        message = event.get("message")
        if isinstance(message, dict):
            messages.append(message)
            for block in _claude_content_blocks(message.get("content")):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text"):
                    final_text = str(block.get("text"))
                elif block.get("type") == "tool_use":
                    tool_calls.append(block)
        if event.get("type") == "result":
            result_meta = dict(event)
            if event.get("result"):
                final_text = str(event.get("result"))
            if event.get("session_id"):
                session_id = str(event.get("session_id"))

    if session_id and "session_id" not in result_meta:
        result_meta["session_id"] = session_id
    return {
        "final_text": final_text,
        "messages": messages,
        "tool_calls": tool_calls,
        "result_meta": result_meta,
        "events": events,
        "stderr": None,
        "error": None,
    }


def find_claude_sdk_python(container: str) -> str | None:
    candidates = ("/usr/bin/python3", "python3", "/opt/attackenv/bin/python3")
    probe = "import claude_agent_sdk"
    for candidate in candidates:
        proc = docker_exec(container, [candidate, "-c", probe], user="root", timeout=30)
        if proc.returncode == 0:
            return candidate
    return None


def missing_claude_cli_reason(container: str) -> str | None:
    proc = docker_exec(container, ["bash", "-lc", "command -v claude"], user="root", timeout=30)
    if proc.returncode != 0:
        return (
            "Claude Code CLI is not installed in the agent container. Use an "
            "image with Claude Code installed."
        )
    return None


def missing_backend_reason(container: str, backend: Backend, *, claude_python: str | None = None) -> str | None:
    if backend == "codex":
        proc = docker_exec(container, ["bash", "-lc", "command -v codex"], user="root", timeout=30)
        if proc.returncode != 0:
            return (
                "codex CLI is not installed in the agent container. "
                "Use an image with Codex installed, for example one built from "
                "docker/nyuctf-agent/Dockerfile."
            )
        return None

    proc = docker_exec(container, ["bash", "-lc", "command -v claude"], user="root", timeout=30)
    if proc.returncode != 0:
        return (
            "Claude Code CLI is not installed in the agent container. Use an "
            "image with Claude Code installed."
        )
    if claude_python is None:
        return (
            "claude-agent-sdk is not importable from python3 candidates in the "
            "agent container. In the NYUCTF attackenv Dockerfile it is usually "
            "installed in /usr/bin/python3 before /opt/attackenv is created; "
            "rebuild/tag the image, or install claude-agent-sdk into the active "
            "Python environment."
        )
    return None


def extract_codex_actions(
    events: list[dict[str, Any]],
    *,
    container: str,
    workdir: str,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") if isinstance(event.get("item"), dict) else event
        if not isinstance(item, dict):
            continue
        if item.get("type") == "command_execution":
            instruction = (
                item.get("command")
                or item.get("cmd")
                or item.get("input")
                or item.get("arguments")
                or {k: v for k, v in item.items() if k not in ("type", "id")}
            )
            result = (
                item.get("output")
                or item.get("aggregated_output")
                or item.get("result")
                or item.get("stderr")
                or ""
            )
            actions.append({
                "type": "command",
                "instruction": instruction,
                "result": result,
                "exit_code": item.get("exit_code"),
            })
        else:
            actions.extend(codex_file_event_to_actions(item, container=container, workdir=workdir))
    return actions


FILE_EVENT_RE = re.compile(r"(file|patch|edit|write|apply_patch)", re.I)


def codex_file_event_to_actions(
    item: dict[str, Any],
    *,
    container: str,
    workdir: str,
) -> list[dict[str, Any]]:
    marker = json.dumps(
        {
            "type": item.get("type"),
            "name": item.get("name"),
            "tool": item.get("tool"),
            "kind": item.get("kind"),
        },
        ensure_ascii=False,
    )
    if not FILE_EVENT_RE.search(marker):
        return []
    file_path = first_string_value(
        item,
        ("file_path", "filepath", "path", "target_path", "target_file", "filename", "file"),
    )
    content = first_string_value(item, ("content", "file_content", "new_content", "new_text", "text", "body"))
    if file_path and content is not None:
        return [{
            "type": "tool",
            "instruction": {"name": "write", "input": {"file_path": file_path, "content": content}},
            "result": item.get("result") or item.get("output") or "captured from Codex file event",
            "status": item.get("status"),
        }]
    if file_path:
        final_content = read_container_text_file(container, file_path, workdir=workdir)
        if final_content is not None:
            return [{
                "type": "tool",
                "instruction": {"name": "write", "input": {"file_path": file_path, "content": final_content}},
                "result": f"captured final file content ({len(final_content)} chars)",
                "status": item.get("status"),
            }]
    return [{
        "type": "file_event",
        "instruction": compact_file_event(item),
        "result": item.get("result") or item.get("output"),
        "status": item.get("status"),
        "replayable": False,
    }]


def first_string_value(value: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, dict):
        for key in keys:
            item = value.get(key)
            if isinstance(item, str):
                return item
        for item in value.values():
            found = first_string_value(item, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = first_string_value(item, keys)
            if found is not None:
                return found
    return None


def compact_file_event(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in ("type", "name", "tool", "kind", "path", "file_path", "arguments", "input")
        if item.get(key) is not None
    }


def read_container_text_file(container: str, file_path: str, *, workdir: str) -> str | None:
    proc = docker_exec(
        container,
        [
            "python3",
            "-c",
            (
                "from pathlib import Path\n"
                "import sys\n"
                "p=Path(sys.stdin.read())\n"
                "if not p.is_absolute(): p=Path.cwd()/p\n"
                "if not p.is_file(): raise SystemExit(2)\n"
                "sys.stdout.write(p.read_text(encoding='utf-8', errors='replace'))\n"
            ),
        ],
        user="root",
        workdir=workdir,
        input_text=file_path,
        timeout=30,
    )
    return proc.stdout if proc.returncode == 0 else None


def extract_claude_actions(parsed: Any) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    tool_results = _extract_claude_tool_results(parsed)
    actions: list[dict[str, Any]] = []
    for call in parsed.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        call_id = call.get("id") or call.get("tool_use_id")
        actions.append({
            "type": "tool",
            "instruction": {
                "name": call.get("name"),
                "input": call.get("input") or call.get("arguments") or {},
            },
            "result": call.get("result") or call.get("output") or tool_results.get(str(call_id)),
        })
    return actions


def summarize_claude_debug(parsed: Any) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    return {
        "stderr": parsed.get("stderr"),
        "error": parsed.get("error"),
        "result_meta": parsed.get("result_meta"),
        "message_count": len(parsed.get("messages") or []),
        "content_block_types": _claude_content_block_types(parsed),
        "tool_results": _extract_claude_tool_results(parsed),
    }


def _claude_content_block_types(parsed: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for message in parsed.get("messages") or []:
        if not isinstance(message, dict):
            continue
        for block in _claude_content_blocks(message.get("content")):
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type and block_type not in seen:
                seen.append(block_type)
    return seen


def _extract_claude_tool_results(parsed: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for message in parsed.get("messages") or []:
        if not isinstance(message, dict):
            continue
        for block in _claude_content_blocks(message.get("content")):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id") or block.get("id")
            if tool_use_id:
                results[str(tool_use_id)] = {
                    "content": block.get("content"),
                    "is_error": block.get("is_error"),
                }
    return results


def _claude_content_blocks(content: Any) -> list[Any]:
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        return [content]
    return []


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


def replay_trajectory_action(
    action: dict[str, Any],
    *,
    index: int,
    container: str,
    user: str,
    workdir: str,
    timeout: int,
) -> dict[str, Any]:
    tool_name, tool_input = action_tool(action)
    if tool_name == "write":
        file_path = tool_input.get("file_path") or tool_input.get("path")
        content = tool_input.get("content")
        if isinstance(file_path, str) and isinstance(content, str):
            proc = docker_exec(
                container,
                ["bash", "-lc", 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"', "_", file_path],
                user=user,
                workdir=workdir,
                input_text=content,
                timeout=timeout,
            )
            return replay_completed(index=index, action=action, instruction=action.get("instruction"), proc=proc)

    if tool_name == "read":
        file_path = tool_input.get("file_path") or tool_input.get("path")
        if isinstance(file_path, str):
            proc = docker_exec(
                container,
                ["bash", "-lc", 'cat -- "$1"', "_", file_path],
                user=user,
                workdir=workdir,
                timeout=timeout,
            )
            return replay_completed(index=index, action=action, instruction=action.get("instruction"), proc=proc)

    command = action_to_command(action)
    if not command:
        return {
            "index": index,
            "type": action.get("type"),
            "instruction": action.get("instruction"),
            "replayed": False,
            "reason": "unsupported action shape",
        }
    proc = docker_exec(
        container,
        ["bash", "-lc", command],
        user=user,
        workdir=workdir,
        timeout=timeout,
    )
    return replay_completed(index=index, action=action, instruction=command, proc=proc)


def replay_completed(
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


def replay_trajectory_actions(
    trajectory: dict[str, Any],
    *,
    container: str,
    workdir: str,
    timeout: int,
) -> list[dict[str, Any]]:
    run = trajectory.get("run") or {}
    user = str(run.get("agent_user") or "root")
    return [
        replay_trajectory_action(
            action,
            index=index,
            container=container,
            user=user,
            workdir=workdir,
            timeout=timeout,
        )
        for index, action in enumerate(trajectory.get("actions") or [], 1)
    ]


def verify_gold(item: dict[str, Any], history: Any) -> dict[str, Any]:
    gold = str(item.get("gold") or "")
    text = json.dumps(history, ensure_ascii=False)
    ok = bool(gold) and gold in text
    return {
        "ok": ok,
        "checks": [{
            "type": "gold_match",
            "ok": ok,
            "needle": gold,
        }],
    }


def build_input_output_pairs(
    prompt: str,
    output_text: str,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if output_text:
        output.append({"type": "text", "content": output_text})
    for index, action in enumerate(actions, 1):
        output.append({
            "type": "action",
            "index": index,
            "instruction": action.get("instruction"),
            "result": action.get("result"),
        })
    return [{"input": prompt, "output": output}]


def build_trajectory(
    *,
    item: dict[str, Any],
    prompt: str,
    backend: str,
    model: str | None,
    agent_result: dict[str, Any],
    env_info: dict[str, Any],
    verification: dict[str, Any],
    prompt_field: str,
) -> dict[str, Any]:
    if backend == "codex":
        agent_result, _removed = clean_codex_noise(agent_result)

    output_text = agent_result.get("text", "")
    actions = agent_result.get("actions") or []
    setup_steps = env_info.get("setup_steps") or []
    return {
        "item": item,
        "run": {
            "backend": backend,
            "model": model,
            "prompt_field": prompt_field,
            "claude_runner": agent_result.get("runner") if backend == "claude" else None,
            "agent_user": "root",
            "container": env_info.get("container"),
            "workdir": env_info.get("workdir"),
            "image_name": env_info.get("image_name"),
        },
        "prompt_to_model": prompt,
        "input_output": build_input_output_pairs(prompt, output_text, actions),
        "setup": setup_steps,
        "model_output": {
            "text": output_text,
            "returncode": agent_result.get("returncode"),
            "error": agent_result.get("error"),
            "debug": agent_result.get("debug"),
        },
        "actions": actions,
        "verification": verification,
        "replay": {
            "challenge_root": str(env_info.get("challenge_root")),
            "compose_file": str(env_info.get("compose_file")) if env_info.get("compose_file") else None,
            "dataset_item_id": item_id(item),
            "steps": [*setup_steps, *actions],
        },
    }


def run_agent_for_item(
    item: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    env_info = setup_environment(
        item,
        nyuctf_root=args.nyuctf_root,
        image_name=args.image_name,
        cpus=args.cpus,
        memory=args.memory,
        container_name=args.container_name,
    )
    env_info["image_name"] = args.image_name
    try:
        prompt = build_prompt(item, field=args.prompt_field)
        if args.dry_run:
            agent_result = {
                "backend": args.backend,
                "returncode": 0,
                "text": "",
                "error": None,
                "actions": [],
                "dry_run_prompt": prompt,
            }
            verification = {"ok": False, "checks": [], "dry_run": True}
        elif args.backend == "codex":
            agent_result = run_codex_in_container(
                prompt,
                container=env_info["container"],
                workdir=env_info["workdir"],
                model=args.model,
                user="root",
                timeout=args.timeout,
                bypass=args.bypass,
                write_config=args.codex_write_config,
                responses_base_url=args.responses_base_url,
                responses_env_key=args.responses_env_key,
            )
            verification = verify_gold(item, agent_result)
        else:
            agent_result = run_claude_in_container(
                prompt,
                container=env_info["container"],
                workdir=env_info["workdir"],
                model=args.model,
                user="root",
                timeout=args.timeout,
                bypass=args.bypass,
                max_turns=args.max_turns,
                anthropic_base_url=args.anthropic_base_url,
                anthropic_env_key=args.anthropic_env_key,
                allowed_tools=parse_csv_arg(getattr(args, "allowed_tools", None)),
                claude_runner=getattr(args, "claude_runner", "sdk"),
            )
            verification = verify_gold(item, agent_result)

        return build_trajectory(
            item=item,
            prompt=prompt,
            backend=args.backend,
            model=args.model,
            agent_result=agent_result,
            env_info=env_info,
            verification=verification,
            prompt_field=args.prompt_field,
        )
    finally:
        cleanup_environment(env_info)


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--nyuctf-root", type=Path, default=DEFAULT_NYUCTF_ROOT)
    parser.add_argument("--image-name", default=DEFAULT_IMAGE)
    parser.add_argument("--backend", choices=("codex", "claude"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--prompt-field", default="harmful", choices=("harmful", "query"))
    parser.add_argument("--ids", help="Comma-separated item ids/challenge names to run.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--allowed-tools", help="Comma-separated Claude Code tools/rules, e.g. Bash,Write,Read.")
    parser.add_argument("--claude-runner", choices=("sdk", "cli"), default="sdk")
    parser.add_argument("--bypass", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--codex-write-config", action="store_true")
    parser.add_argument("--responses-base-url")
    parser.add_argument("--responses-env-key", default="OPENAI_API_KEY")
    parser.add_argument("--anthropic-base-url")
    parser.add_argument("--anthropic-env-key", default="ANTHROPIC_API_KEY")
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument("--memory", default="4g")
    parser.add_argument("--container-name")
