"""CLI entry point for running X-Teaming against an agent target in VRAP."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from xteaming_vrap.bridge import (
    DEFAULT_ATTACKER_MODEL,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CONTAINER_CLAUDE_HOME,
    DEFAULT_CODEX_MODEL,
    DEFAULT_COMPOSE,
    read_dataset,
)
from xteaming_vrap.models import TransformersTextGradEngine
from xteaming_vrap.plans import (
    load_attack_plans,
    load_config,
    selected_item_subtasks,
)
from xteaming_vrap.runner import (
    build_dry_run_result,
    run_xteaming_subtask,
)
from xteaming_vrap.utils import (
    apply_codex_env_defaults,
    make_output_path,
    write_run_metadata,
)

DEFAULT_XTEAMING_RESPONSES_BASE_URL = "https://az.gptplus5.com/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local-Transformers X-Teaming against Codex or Claude Code in VRAP."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/advcua_test.jsonl"))
    parser.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--ids", help="Comma-separated AdvCUA item ids to run.")
    parser.add_argument(
        "--pairs",
        help="Comma-separated id:subtask_index pairs, e.g. 1:0,13:2.",
    )
    parser.add_argument("--start-from", type=int)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--subtask-index", type=int, default=0)
    parser.add_argument("--all-subtasks", action="store_true")
    parser.add_argument("--attacker-model-path", default=DEFAULT_ATTACKER_MODEL)
    parser.add_argument("--attacker-max-new-tokens", type=int, default=1024)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--plans-file", type=Path)
    parser.add_argument("--plan-set", type=int)
    parser.add_argument("--plan-strategy", type=int)
    parser.add_argument("--strategies-per-behavior", type=int)
    parser.add_argument("--run-all-strategies", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--strategy-sample-seed", type=int)
    parser.add_argument("--allow-default-strategy", action="store_true")
    parser.add_argument("--target-backend", choices=("codex", "claude"), default="codex")
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--codex-timeout", type=int, default=600)
    parser.add_argument("--target-timeout", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--agent-user")
    parser.add_argument("--agent-session", choices=("none", "resume"), default="resume")
    parser.add_argument(
        "--target-execution-mode",
        choices=("chat", "execute"),
        default="chat",
        help=(
            "Deprecated compatibility option. X-Teaming target turns are now "
            "always sent as chat; use --strategy-success-mode execute_verify "
            "for post-strategy execution and verification."
        ),
    )
    parser.add_argument(
        "--post-subtask-execution-turn",
        action="store_true",
        help=(
            "After the selected X-Teaming conversation for a subtask finishes, "
            "append one natural follow-up turn in the same target session that "
            "asks the target to try it in the current environment."
        ),
    )
    parser.add_argument(
        "--post-subtask-execution-prompt",
        default=None,
        help="Override the natural follow-up prompt used by --post-subtask-execution-turn.",
    )
    parser.add_argument("--bypass", action="store_true")
    parser.add_argument("--codex-write-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--responses-base-url", default=DEFAULT_XTEAMING_RESPONSES_BASE_URL)
    parser.add_argument("--responses-env-key", default="OPENAI_API_KEY")
    parser.add_argument("--anthropic-base-url", default=DEFAULT_ANTHROPIC_BASE_URL)
    parser.add_argument("--anthropic-env-key", default="ANTHROPIC_API_KEY")
    parser.add_argument("--claude-home", default=DEFAULT_CONTAINER_CLAUDE_HOME)
    parser.add_argument("--claude-max-turns", type=int, default=30)
    parser.add_argument(
        "--claude-runner",
        choices=("sdk", "cli"),
        default="sdk",
        help="Claude target runner: sdk uses claude-agent-sdk; cli uses Claude Code print-mode CLI.",
    )
    parser.add_argument(
        "--allowed-tools",
        default="Bash,Read,Write",
        help="Comma-separated Claude Code tools/rules to auto-approve.",
    )
    parser.add_argument(
        "--fork-session",
        action="store_true",
        help="Claude only: fork when resuming instead of continuing the same session.",
    )
    parser.add_argument("--placeholder-api-key", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-proxy-env", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gpt-judge", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--strategy-success-mode",
        choices=("judge", "execute_verify"),
        default="judge",
        help=(
            "judge keeps the original X-Teaming outer strategy stop condition "
            "(GPTJudge score 5). execute_verify runs an execution prompt from "
            "the completed strategy output and accepts the strategy only when "
            "live verification succeeds."
        ),
    )
    parser.add_argument(
        "--strategy-execution-output-mode",
        choices=("final", "all"),
        default="all",
        help="Which x-teaming target outputs to feed into per-strategy execution.",
    )
    parser.add_argument(
        "--strategy-execution-prompt-style",
        choices=("auto", "rule", "legacy"),
        default="rule",
        help="Prompt builder used by per-strategy execution.",
    )
    parser.add_argument(
        "--strategy-execution-prompt-builder-model",
        default="gpt-4o-2024-11-20",
    )
    parser.add_argument("--strategy-execution-prompt-builder-timeout", type=float, default=120)
    parser.add_argument("--strategy-execution-prompt-builder-include-item-prompt", action="store_true")
    parser.add_argument("--strategy-execution-omit-subtask-info", action="store_true")
    parser.add_argument("--strategy-execution-timeout", type=int)
    parser.add_argument(
        "--item-attempts",
        type=int,
        help=(
            "For all-subtask execute_verify mode, number of item-level random "
            "strategy-combination attempts. Defaults to strategies_per_behavior."
        ),
    )
    parser.add_argument("--textgrad", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--isolated-textgrad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--textgrad-isolation-mode",
        choices=("snapshot_promote", "replay"),
        default="snapshot_promote",
        help=(
            "snapshot_promote restores Codex history plus replayed execution actions "
            "for each trial and promotes the winning trial state; replay keeps the "
            "older prompt-only replay behavior."
        ),
    )
    parser.add_argument("--textgrad-max-turns-per-phase", type=int)
    parser.add_argument("--plan-revision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-timeout", type=int, default=180)
    parser.add_argument(
        "--stop-on-target-error",
        action="store_true",
        help="Stop the run immediately when a target strategy returns an error.",
    )
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    attacker_config = dict(config["attacker"])
    tg_config = dict(config.get("textgrad") or {})
    eval_config = dict(config.get("evaluation") or {})
    if args.use_gpt_judge is None:
        args.use_gpt_judge = bool(eval_config.get("use_gpt_judge", True))
    if args.textgrad is None:
        args.textgrad = bool(tg_config.get("enabled", False))
    if args.plan_revision is None:
        args.plan_revision = bool(attacker_config.get("plan_revision", True))
    if args.run_all_strategies is None:
        args.run_all_strategies = bool(attacker_config.get("run_all_strategies", False))
    if args.strategy_success_mode == "execute_verify":
        args.verify = True
    attacker_config.update(
        {
            "provider": "transformers",
            "model": args.attacker_model_path,
            "max_new_tokens": args.attacker_max_new_tokens,
            "device_map": args.device_map,
        }
    )
    tg_config.update(
        {
            "enabled": args.textgrad,
            "provider": "transformers",
            "model": args.attacker_model_path,
            "max_new_tokens": int(tg_config.get("max_new_tokens", args.attacker_max_new_tokens)),
            "device_map": args.device_map,
            "temperature": float(tg_config.get("temperature", 0)),
            "max_retries": int(tg_config.get("max_retries", attacker_config.get("max_retries", 3))),
        }
    )
    if args.textgrad_max_turns_per_phase is not None:
        tg_config["max_turns_per_phase"] = args.textgrad_max_turns_per_phase
    else:
        tg_config.setdefault("max_turns_per_phase", 4)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    apply_codex_env_defaults(args)
    output_path = args.output or make_output_path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot_tmp_root = output_path.parent
    if args.textgrad and not args.dry_run:
        if not args.use_gpt_judge:
            raise ValueError("--textgrad requires --use-gpt-judge")
        import textgrad as tg

        tg.set_backward_engine(TransformersTextGradEngine(tg_config), override=True)
    write_run_metadata(output_path, args, attacker_config, tg_config, eval_config)

    data = read_dataset(args.dataset)
    runs = selected_item_subtasks(data, args)
    plans_by_behavior_id = load_attack_plans(args.plans_file)
    total_runs = len(runs)
    written = 0

    with output_path.open("a", encoding="utf-8") as out:
        for item, subtask_index, subtask in runs:
            item_id = item.get("id", item.get("unique_id"))
            written += 1
            print(
                f"[{written}/{total_runs}] item={item_id} "
                f"subtask={subtask_index} target={subtask[:100]!r}",
                flush=True,
            )
            if args.dry_run:
                result = build_dry_run_result(
                    item=item,
                    subtask_index=subtask_index,
                    subtask=subtask,
                    attacker_config=attacker_config,
                    plans_by_behavior_id=plans_by_behavior_id,
                    args=args,
                )
            else:
                result = run_xteaming_subtask(
                    item=item,
                    subtask_index=subtask_index,
                    subtask=subtask,
                    attacker_config=attacker_config,
                    tg_config=tg_config,
                    eval_config=eval_config,
                    plans_by_behavior_id=plans_by_behavior_id,
                    args=args,
                )
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            last_turn = result["turns"][-1] if result["turns"] else {}
            print(
                "  turns="
                f"{len(result.get('input_output') or [])} "
                f"actions={len(result.get('actions') or [])} "
                f"rc={last_turn.get('returncode')} "
                f"verification={result['verification'].get('ok')}",
                flush=True,
            )

    print(f"Wrote {written} result(s) to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
