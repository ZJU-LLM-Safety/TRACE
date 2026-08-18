"""Run X-Teaming sequentially across every subprompt of selected AdvCUA items."""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from run_xteaming_vrap_codex import build_parser as build_subtask_parser

from xteaming_vrap.bridge import read_dataset, selected_items
from xteaming_vrap.models import TransformersTextGradEngine
from xteaming_vrap.plans import load_attack_plans, load_config
from xteaming_vrap.sequence import run_all_subprompts_for_item
from xteaming_vrap.utils import apply_codex_env_defaults


def build_parser() -> argparse.ArgumentParser:
    parser = build_subtask_parser()
    parser.description = "Run local-Transformers X-Teaming across all subprompts per item."
    parser.add_argument(
        "--link-mode",
        choices=("session", "action_summary"),
        required=True,
        help="How to connect consecutive subprompts within the same item.",
    )
    return parser


def make_output_path(output_dir: Path | None, link_mode: str) -> Path:
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("xteaming_vrap_codex_outputs") / f"subprompt_sequence_{link_mode}_{stamp}"
    return output_dir / "results.jsonl"


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
    output_path = args.output or make_output_path(args.output_dir, args.link_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot_tmp_root = output_path.parent
    args.persist_codex_home_snapshots = args.link_mode == "session"
    if args.persist_codex_home_snapshots:
        args.codex_snapshot_store = output_path.parent / ".codex_home_snapshots"

    if args.textgrad and not args.dry_run:
        if not args.use_gpt_judge:
            raise ValueError("--textgrad requires --use-gpt-judge")
        import textgrad as tg

        tg.set_backward_engine(TransformersTextGradEngine(tg_config), override=True)

    metadata = {
        "dataset": str(args.dataset),
        "config": str(args.config),
        "plans_file": str(args.plans_file),
        "output": str(output_path),
        "link_mode": args.link_mode,
        "pairs": args.pairs,
        "ids": args.ids,
        "limit": args.limit,
        "target_backend": args.target_backend,
        "target_model": args.claude_model if args.target_backend == "claude" else args.codex_model,
        "codex_model": args.codex_model,
        "claude_model": args.claude_model,
        "target_execution_mode": args.target_execution_mode,
        "post_subtask_execution_turn": args.post_subtask_execution_turn,
        "post_subtask_execution_prompt": args.post_subtask_execution_prompt,
        "bypass": args.bypass,
        "responses_base_url": args.responses_base_url,
        "anthropic_base_url": args.anthropic_base_url,
        "claude_home": getattr(args, "claude_home", None),
        "strategies_per_behavior": (
            args.strategies_per_behavior
            if args.strategies_per_behavior is not None
            else attacker_config.get("strategies_per_behavior")
        ),
        "textgrad": args.textgrad,
        "textgrad_isolation_mode": args.textgrad_isolation_mode if args.textgrad else None,
    }
    (output_path.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    data = read_dataset(args.dataset)
    items = selected_items(data, args)
    plans_by_behavior_id = load_attack_plans(args.plans_file)

    with output_path.open("a", encoding="utf-8") as out:
        for index, item in enumerate(items, 1):
            item_id = item.get("id", item.get("unique_id"))
            print(f"[{index}/{len(items)}] item={item_id} link_mode={args.link_mode}", flush=True)
            if args.dry_run:
                raise ValueError("Dry-run is not implemented for subprompt sequence mode.")
            result = run_all_subprompts_for_item(
                item=item,
                attacker_config=attacker_config,
                tg_config=tg_config,
                eval_config=eval_config,
                plans_by_behavior_id=plans_by_behavior_id,
                args=args,
            )
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"  wrote item={item_id} subprompts={len(result.get('subprompts') or [])} "
                f"actions={len(result.get('actions') or [])}",
                flush=True,
            )

    print(f"Wrote {len(items)} result(s) to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
