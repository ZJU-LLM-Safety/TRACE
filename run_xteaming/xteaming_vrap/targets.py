"""Compatibility target helpers with optional NYUCTF lab support."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


TRACE_ROOT = Path(__file__).resolve().parents[2]
if str(TRACE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACE_ROOT))
TRACE_TOOLS = TRACE_ROOT / "tools"
if str(TRACE_TOOLS) not in sys.path:
    sys.path.insert(0, str(TRACE_TOOLS))

from agent_execution.targets import *  # noqa: F401,F403,E402
from agent_execution.targets import VrapCodexTarget as _VrapCodexTarget  # noqa: E402
from nyuctf_bridge_runner import (  # noqa: E402
    cleanup_environment as cleanup_nyuctf_environment,
    run_claude_in_container,
    run_codex_in_container,
    setup_environment as setup_nyuctf_environment,
)


class VrapCodexTarget(_VrapCodexTarget):
    """Target wrapper for either the historical VRAP lab or NYUCTF Docker lab."""

    def __init__(self, item: dict[str, Any], args: Any):
        super().__init__(item, args)
        self.nyuctf_env_info: dict[str, Any] | None = None
        if self.lab_backend == "nyuctf":
            self.attacker_identity = "root"
            self.agent_user = args.agent_user or "root"

    @property
    def lab_backend(self) -> str:
        return str(getattr(self.args, "lab_backend", "vrap") or "vrap")

    @property
    def home_path(self) -> str:
        if self.lab_backend != "nyuctf":
            return super().home_path
        if self.backend == "claude":
            return "/root"
        return "/tmp/codex_home"

    def __enter__(self) -> "VrapCodexTarget":
        if self.lab_backend != "nyuctf":
            return super().__enter__()
        self.nyuctf_env_info = setup_nyuctf_environment(
            self.item,
            nyuctf_root=Path(getattr(self.args, "nyuctf_root")),
            image_name=str(getattr(self.args, "image_name")),
            cpus=float(getattr(self.args, "cpus", 2.0)),
            memory=str(getattr(self.args, "memory", "4g")),
            container_name=getattr(self.args, "container_name", None),
        )
        self.nyuctf_env_info["image_name"] = str(getattr(self.args, "image_name"))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.lab_backend != "nyuctf":
            return super().__exit__(exc_type, exc, tb)
        if self.nyuctf_env_info is not None and not self.args.keep_container:
            cleanup_nyuctf_environment(self.nyuctf_env_info)

    def _run_agent(self, prompt: str) -> dict[str, Any]:
        if self.lab_backend != "nyuctf":
            return super()._run_agent(prompt)
        if self.nyuctf_env_info is None:
            raise RuntimeError("NYUCTF target environment has not been initialized")
        should_resume = self.args.agent_session == "resume" and bool(self.resume_id)
        timeout = int(
            getattr(self.args, "target_timeout", None)
            or getattr(self.args, "codex_timeout", 600)
        )
        if self.backend == "codex":
            return run_codex_in_container(
                prompt,
                container=str(self.nyuctf_env_info["container"]),
                workdir=str(self.nyuctf_env_info["workdir"]),
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
            return run_claude_in_container(
                prompt,
                container=str(self.nyuctf_env_info["container"]),
                workdir=str(self.nyuctf_env_info["workdir"]),
                model=self.model_name,
                user=self.agent_user,
                timeout=timeout,
                bypass=self.args.bypass,
                max_turns=getattr(self.args, "claude_max_turns", None),
                anthropic_base_url=getattr(self.args, "anthropic_base_url", None),
                anthropic_env_key=getattr(
                    self.args,
                    "anthropic_env_key",
                    "ANTHROPIC_API_KEY",
                ),
                allowed_tools=parse_csv_arg(getattr(self.args, "allowed_tools", None)),
                resume_session_id=self.resume_id if should_resume else None,
                fork_session=bool(getattr(self.args, "fork_session", False)),
            )
        raise ValueError(f"Unsupported target backend: {self.backend!r}")
