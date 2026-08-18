"""Generate X-Teaming attack plans with tolerant JSON normalization.

This keeps the upstream ``generate_attack_plans.py`` unchanged while handling
OpenAI-compatible endpoints that wrap the requested ``strategy_1`` ...
``strategy_10`` object in an extra key such as ``strategies``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from pprint import pprint
from typing import Any

import yaml
from tenacity import retry, stop_after_attempt, wait_fixed

from agents.base_agent import BaseAgent
from generate_attack_plans import (
    create_output_directory,
    load_and_format_prompts,
    load_datasets,
    setup_logging,
)


class TransformersPlanAgent:
    def __init__(self, model_path: str, max_new_tokens: int, device_map: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        if device_map.lower() not in {"", "none", "null"}:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

    def call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        del response_format
        tokenizer = self.tokenizer
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        do_sample = temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.9
        generated = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_strategies(parsed: Any, expected_count: int = 10) -> dict[str, Any]:
    if isinstance(parsed, str):
        return normalize_strategies(parse_json_object(parsed), expected_count)

    if isinstance(parsed, dict):
        strategy_keys = [
            key
            for key in parsed
            if str(key).lower().startswith("strategy")
        ]
        if len(strategy_keys) == expected_count:
            return {
                f"strategy_{index}": parsed[key]
                for index, key in enumerate(sorted(strategy_keys, key=strategy_sort_key), 1)
            }

        for wrapper_key in ("strategies", "attack_strategies", "plans"):
            if wrapper_key in parsed:
                return normalize_strategies(parsed[wrapper_key], expected_count)

        if len(parsed) == 1:
            return normalize_strategies(next(iter(parsed.values())), expected_count)

        dict_values = [value for value in parsed.values() if isinstance(value, dict)]
        if len(parsed) == expected_count and len(dict_values) == expected_count:
            return {
                f"strategy_{index}": parsed[key]
                for index, key in enumerate(sorted(parsed, key=strategy_sort_key), 1)
            }

    if isinstance(parsed, list) and len(parsed) == expected_count:
        return {
            f"strategy_{index}": strategy
            for index, strategy in enumerate(parsed, 1)
        }

    keys = list(parsed) if isinstance(parsed, dict) else None
    raise ValueError(
        f"Expected {expected_count} strategies, or a wrapper containing them; "
        f"got {type(parsed).__name__} with keys={keys}"
    )


def strategy_sort_key(key: str) -> int:
    try:
        return int(key.rsplit("_", 1)[-1])
    except ValueError:
        return 0


@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def generate_strategies(
    agent: BaseAgent,
    messages: list[dict[str, str]],
    set_num: int,
    temperature: float,
    strategies_per_set: int,
) -> dict[str, Any]:
    response = agent.call_api(
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    if not response:
        response = agent.call_api(
            messages=messages,
            temperature=temperature,
        )
    if not response:
        raise ValueError("Empty response from attack plan generator")
    parsed = parse_json_object(response)
    strategies = normalize_strategies(parsed, strategies_per_set)
    logging.info("\nSet %s Generated Strategies:", set_num)
    logging.info(json.dumps(strategies, ensure_ascii=False, indent=2))
    return strategies


def process_single_behavior(
    index: int,
    row: Any,
    agent: BaseAgent,
    temperature: float,
    num_sets: int,
    strategies_per_set: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    behavior = row["Behavior"]
    behavior_id = row["BehaviorID"]
    logging.info("\n%s", "=" * 50)
    logging.info("Processing Behavior %s (ID: %s):", index, behavior_id)
    logging.info("Behavior: %s", behavior)
    logging.info("%s", "=" * 50)

    all_messages = []
    all_responses = {}
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
        logging.info("\nGenerating Set %s:", set_num)
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
        logging.info("Messages for set %s", set_num)
        logging.info(pprint(messages, indent=2))
        all_messages.append(
            {
                "behavior_index": index,
                "behavior": behavior,
                "set_number": set_num,
                "messages": messages,
            }
        )
        all_responses[f"Set_{set_num}"] = generate_strategies(
            agent=agent,
            messages=messages,
            set_num=set_num,
            temperature=temperature,
            strategies_per_set=strategies_per_set,
        )

    return behavior_data, all_messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate X-Teaming attack plans with tolerant JSON parsing."
    )
    parser.add_argument("-c", "--config", default="./config/config.yaml")
    parser.add_argument("--num-sets", type=int, default=5)
    parser.add_argument("--strategies-per-set", type=int, default=10)
    parser.add_argument("--backend", choices=("config", "transformers"), default="config")
    parser.add_argument(
        "--transformers-model-path",
        default=os.environ.get("QWEN32B", "Qwen/Qwen2.5-32B-Instruct"),
    )
    parser.add_argument("--transformers-max-new-tokens", type=int, default=4096)
    parser.add_argument("--device-map", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = create_output_directory(
        config["attack_plan_generator"]["attack_plan_generation_dir"]
    )
    setup_logging(output_dir)
    if args.backend == "transformers":
        agent = TransformersPlanAgent(
            model_path=args.transformers_model_path,
            max_new_tokens=args.transformers_max_new_tokens,
            device_map=args.device_map,
        )
    else:
        agent = BaseAgent(config["attack_plan_generator"])
    df = load_datasets(
        config["attack_plan_generator"]["behavior_path"],
        config["attack_plan_generator"]["num_behaviors"],
    )

    all_behaviors_data = []
    all_messages = []
    for index, row in df.iterrows():
        behavior_result, messages_result = process_single_behavior(
            index=index,
            row=row,
            agent=agent,
            temperature=config["attack_plan_generator"]["temperature"],
            num_sets=args.num_sets,
            strategies_per_set=args.strategies_per_set,
        )
        all_behaviors_data.append(behavior_result)
        all_messages.extend(messages_result)
        Path(output_dir, "attack_plans.json").write_text(
            json.dumps(all_behaviors_data, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        Path(output_dir, "all_messages.json").write_text(
            json.dumps(all_messages, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    logging.info("Finished")
    print(Path(output_dir, "attack_plans.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
