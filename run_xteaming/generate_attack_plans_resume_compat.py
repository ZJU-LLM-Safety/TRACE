#!/usr/bin/env python3
"""Generate attack plans in CSV order while preserving partial progress."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from agents.base_agent import BaseAgent
from generate_attack_plans import load_and_format_prompts
from generate_attack_plans_compat import normalize_strategies, parse_json_object, strategy_sort_key


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def behavior_id(record: dict[str, Any]) -> str | None:
    details = record.get("behavior_details") or {}
    value = details.get("BehaviorID")
    return None if value is None else str(value)


def process_row(
    *,
    index: int,
    row: pd.Series,
    agent: BaseAgent,
    temperature: float,
    num_sets: int,
    strategies_per_set: int,
    raw_dir: Path,
    set_attempts: int,
) -> dict[str, Any]:
    behavior = row["Behavior"]
    all_responses: dict[str, Any] = {}
    behavior_details = {
        key: (None if row.isna()[key] else value)
        for key, value in row.to_dict().items()
    }
    behavior_data = {
        "behavior_number": index,
        "behavior_details": behavior_details,
        "attack_strategies": all_responses,
    }
    for set_num in range(1, num_sets + 1):
        system_prompt, formatted_user_prompt = load_and_format_prompts(
            behavior=behavior,
            set_number=set_num,
            previous_responses=all_responses,
        )
        formatted_user_prompt = formatted_user_prompt.replace(
            "Generate 10 distinct jailbreak strategies",
            f"Generate {strategies_per_set} distinct jailbreak strategy"
            if strategies_per_set == 1
            else f"Generate {strategies_per_set} distinct jailbreak strategies",
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt},
        ]
        all_responses[f"Set_{set_num}"] = generate_set_with_raw_retries(
            agent=agent,
            behavior_id=str(behavior_details["BehaviorID"]),
            set_num=set_num,
            messages=messages,
            temperature=temperature,
            strategies_per_set=strategies_per_set,
            raw_dir=raw_dir,
            attempts=set_attempts,
        )
    return behavior_data


def safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def generate_set_with_raw_retries(
    *,
    agent: BaseAgent,
    behavior_id: str,
    set_num: int,
    messages: list[dict[str, str]],
    temperature: float,
    strategies_per_set: int,
    raw_dir: Path,
    attempts: int = 8,
) -> dict[str, Any]:
    label = safe_label(f"{behavior_id}_set_{set_num}")
    last_error: Exception | None = None
    partial: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        response = ""
        try:
            response = agent.call_api(
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            if not response:
                response = agent.call_api(messages=messages, temperature=temperature)
            parsed = parse_json_object_lenient(response)
            parsed_partial = extract_partial_strategies(parsed)
            if len(parsed_partial) >= strategies_per_set:
                return renumber_strategies(parsed_partial, strategies_per_set)
            if parsed_partial:
                partial = merge_partial_strategies(partial, parsed_partial, strategies_per_set)
                print(
                    f"partial set {behavior_id} Set_{set_num} "
                    f"attempt {attempt}/{attempts}: collected {len(partial)}/{strategies_per_set}",
                    flush=True,
                )
                if len(partial) >= strategies_per_set:
                    return renumber_strategies(partial, strategies_per_set)
                continue
            return normalize_strategies(parsed, strategies_per_set)
        except Exception as exc:
            last_error = exc
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{label}_attempt_{attempt}.txt").write_text(
                str(response or ""),
                encoding="utf-8",
            )
            print(
                f"retry set {behavior_id} Set_{set_num} "
                f"attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}",
                flush=True,
            )
    if partial:
        print(
            f"using partial set {behavior_id} Set_{set_num}: "
            f"{len(partial)}/{strategies_per_set} strategies",
            flush=True,
        )
        return renumber_strategies(partial, len(partial))
    print(
        f"using empty set {behavior_id} Set_{set_num}: "
        f"no parseable strategies after {attempts} attempts",
        flush=True,
    )
    return {}


def parse_json_object_lenient(text: str) -> Any:
    try:
        return parse_json_object(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
                return parsed
            except json.JSONDecodeError:
                continue
        raise


def extract_partial_strategies(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, str):
        return extract_partial_strategies(parse_json_object_lenient(parsed))
    if isinstance(parsed, list):
        return {
            f"strategy_{index}": value
            for index, value in enumerate(parsed, 1)
            if isinstance(value, dict)
        }
    if not isinstance(parsed, dict):
        return {}

    strategy_items = {
        str(key): value
        for key, value in parsed.items()
        if str(key).lower().startswith("strategy") and isinstance(value, dict)
    }
    if strategy_items:
        return strategy_items

    for wrapper_key in ("strategies", "attack_strategies", "plans"):
        value = parsed.get(wrapper_key)
        nested = extract_partial_strategies(value)
        if nested:
            return nested

    strategy_fields = {"persona", "context", "approach", "turns_needed", "conversation_plan"}
    if strategy_fields.intersection(parsed):
        return {"strategy_1": parsed}
    return {}


def merge_partial_strategies(
    existing: dict[str, Any],
    new: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    merged = dict(existing)
    for _, value in sorted(new.items(), key=lambda item: strategy_sort_key(item[0])):
        if len(merged) >= limit:
            break
        if value in merged.values():
            continue
        merged[f"strategy_{len(merged) + 1}"] = value
    return merged


def renumber_strategies(strategies: dict[str, Any], count: int) -> dict[str, Any]:
    return {
        f"strategy_{index}": value
        for index, (_, value) in enumerate(
            sorted(strategies.items(), key=lambda item: strategy_sort_key(item[0])),
            1,
        )
        if index <= count
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-sets", type=int, default=5)
    parser.add_argument("--strategies-per-set", type=int, default=10)
    parser.add_argument("--behavior-retries", type=int, default=3)
    parser.add_argument("--set-attempts", type=int)
    parser.add_argument("--raw-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    generator_config = dict(config["attack_plan_generator"])
    df = pd.read_csv(generator_config["behavior_path"])
    df = df[df["FunctionalCategory"] == "standard"].reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = load_existing(args.output)
    done = {behavior_id(record) for record in records}
    agent = BaseAgent(generator_config)
    raw_dir = args.raw_dir or args.output.parent / "raw_plan_generation_failures"
    set_attempts = (
        args.set_attempts
        if args.set_attempts is not None
        else int(generator_config.get("max_retries", 3))
    )

    for index, row in df.iterrows():
        current_id = str(row["BehaviorID"])
        if current_id in done:
            print(f"skip existing {current_id}", flush=True)
            continue
        print(f"[{index + 1}/{len(df)}] generating {current_id}", flush=True)
        last_error: Exception | None = None
        for attempt in range(1, args.behavior_retries + 1):
            try:
                record = process_row(
                    index=index,
                    row=row,
                    agent=agent,
                    temperature=generator_config["temperature"],
                    num_sets=args.num_sets,
                    strategies_per_set=args.strategies_per_set,
                    raw_dir=raw_dir,
                    set_attempts=set_attempts,
                )
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"retry behavior {current_id} "
                    f"attempt {attempt}/{args.behavior_retries}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        else:
            raise RuntimeError(f"failed to generate {current_id}") from last_error
        records.append(record)
        done.add(current_id)
        args.output.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(records)} record(s) to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
