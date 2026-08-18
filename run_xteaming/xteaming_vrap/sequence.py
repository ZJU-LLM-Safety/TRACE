"""X-teaming item-level execution across all AdvCUA subprompts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .bridge import CLIENT_CONTAINER
from .runner import run_xteaming_subtask


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
            "container": CLIENT_CONTAINER,
            "link_mode": self.args.link_mode,
            "subprompt_count": len(self.subprompts),
            "target_execution_mode": self.args.target_execution_mode,
            "post_subtask_execution_turn": self.args.post_subtask_execution_turn,
            "bypass": self.args.bypass,
            "strategies_per_behavior": self.attacker_config.get("strategies_per_behavior"),
            "gpt_judge": self.args.use_gpt_judge,
            "textgrad": self.args.textgrad,
            "isolated_textgrad": self.args.isolated_textgrad if self.args.textgrad else None,
            "textgrad_isolation_mode": self.args.textgrad_isolation_mode if self.args.textgrad else None,
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
    return XTeamingSubtaskRunner(
        item=item,
        attacker_config=attacker_config,
        tg_config=tg_config,
        eval_config=eval_config,
        plans_by_behavior_id=plans_by_behavior_id,
        args=args,
    ).run()
