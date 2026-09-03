"""Generate experiment attack plans with tolerant JSON normalization.

This is the experiment copy of the X-Teaming compatible plan generator. It uses
``experiment/config/prompts/plan_generation_prompts.yaml`` and validates the
baseline strategy schema.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Iterable

import pandas as pd
import yaml
from tenacity import retry, stop_after_attempt, wait_fixed

from agents.base_agent import BaseAgent


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS_PATH = EXPERIMENT_ROOT / "config" / "prompts" / "plan_generation_prompts.yaml"
REQUIRED_STRATEGY_FIELDS = {
    "persona",
    "context",
    "approach",
    "turns_needed",
    "conversation_plan",
}


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


def setup_logging(output_dir: Path) -> None:
    class NoColorFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            record.msg = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", str(record.msg))
            return super().format(record)

    file_handler = logging.FileHandler(output_dir / "generation_log.txt")
    file_handler.setFormatter(
        NoColorFormatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler], force=True)


def create_output_directory(base_output_dir: str | Path) -> Path:
    output_dir = Path(base_output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_datasets(csv_path: str | Path, number_of_behaviors: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "FunctionalCategory" in df:
        df = df[df["FunctionalCategory"] == "standard"]
    if number_of_behaviors is not None:
        df = df.head(int(number_of_behaviors))
    return df.reset_index(drop=True)


def load_and_format_prompts(
    behavior: str,
    set_number: int,
    previous_responses: dict[str, Any] | None = None,
    prompts_path: Path = DEFAULT_PROMPTS_PATH,
) -> tuple[str, str]:
    prompts = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))["prompts"]
    system_prompt = prompts["system"]["messages"][0]["content"]

    if set_number == 1 or "user_message2" not in prompts:
        user_prompt = prompts["user_message1"]["messages"][0]["content"]
        return system_prompt, user_prompt.replace("{target_behavior}", behavior)

    user_prompt = prompts["user_message2"]["messages"][0]["content"]
    strategies_text = ""
    for set_name, response in (previous_responses or {}).items():
        strategies_text += f"\n{set_name}:\n{json.dumps(response, ensure_ascii=False, indent=2)}\n"
    formatted_prompt = user_prompt.replace("{target_behavior}", behavior)
    formatted_prompt = formatted_prompt.replace(
        "{previously_generated_strategies}",
        strategies_text,
    )
    return system_prompt, formatted_prompt


def parse_json_object(text: str) -> Any:
    candidates = list(iter_json_candidates(text))
    if candidates:
        return candidates[0]
    raise json.JSONDecodeError("No JSON object or array found", text, 0)


def iter_json_candidates(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    try:
        yield json.loads(text)
        return
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        fenced = match.group(1).strip()
        try:
            yield json.loads(fenced)
        except json.JSONDecodeError:
            pass

    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        yield parsed


def normalize_strategies(parsed: Any, expected_count: int = 10) -> dict[str, Any]:
    if isinstance(parsed, str):
        return normalize_strategies(parse_json_object(parsed), expected_count)

    if isinstance(parsed, dict):
        strategy_keys = [
            key
            for key in parsed
            if str(key).lower().startswith("strategy")
        ]
        if strategy_keys:
            normalized = renumber_strategies(
                parsed,
                strategy_keys,
                max(expected_count, 1),
            )
            if normalized:
                return normalized

        for wrapper_key in ("strategies", "attack_strategies", "plans"):
            if wrapper_key in parsed:
                return normalize_strategies(parsed[wrapper_key], expected_count)

        if REQUIRED_STRATEGY_FIELDS.issubset(parsed):
            return {"strategy_1": parsed}

        if len(parsed) == 1:
            return normalize_strategies(next(iter(parsed.values())), expected_count)

        dict_values = [value for value in parsed.values() if isinstance(value, dict)]
        if dict_values:
            return {
                f"strategy_{index}": parsed[key]
                for index, key in enumerate(sorted(parsed, key=strategy_sort_key), 1)
                if isinstance(parsed[key], dict)
            }

    if isinstance(parsed, list) and parsed:
        return {
            f"strategy_{index}": strategy
            for index, strategy in enumerate(parsed, 1)
            if isinstance(strategy, dict)
        }

    keys = list(parsed) if isinstance(parsed, dict) else None
    raise ValueError(
        f"Expected at least one strategy, or a wrapper containing them; "
        f"got {type(parsed).__name__} with keys={keys}"
    )


def renumber_strategies(
    parsed: dict[str, Any],
    strategy_keys: list[Any],
    expected_count: int,
) -> dict[str, Any]:
    return {
        f"strategy_{index}": parsed[key]
        for index, key in enumerate(sorted(strategy_keys, key=strategy_sort_key), 1)
        if isinstance(parsed[key], dict) and index <= max(expected_count, 1)
    }


def strategy_sort_key(key: Any) -> int:
    match = re.search(r"(\d+)$", str(key))
    return int(match.group(1)) if match else 0


def validate_strategy_schema(strategies: dict[str, Any]) -> None:
    for name, strategy in strategies.items():
        if not isinstance(strategy, dict):
            raise ValueError(f"{name} must be a JSON object")
        missing = REQUIRED_STRATEGY_FIELDS - set(strategy)
        if missing:
            raise ValueError(f"{name} missing experiment fields: {sorted(missing)}")
        conversation_plan = strategy["conversation_plan"]
        if not isinstance(conversation_plan, dict):
            raise ValueError(f"{name}.conversation_plan must be a JSON object")
        if "turn_1" not in conversation_plan or "final_turn" not in conversation_plan:
            raise ValueError(
                f"{name}.conversation_plan must include turn_1 and final_turn"
            )


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
    last_error: Exception | None = None
    for parsed in iter_json_candidates(response):
        try:
            strategies = normalize_strategies(parsed, strategies_per_set)
            validate_strategy_schema(strategies)
            logging.info("\nSet %s Generated Strategies:", set_num)
            logging.info(json.dumps(strategies, ensure_ascii=False, indent=2))
            return strategies
        except Exception as exc:
            last_error = exc
    logging.warning(
        "Could not parse generated strategy JSON. Raw response prefix: %s",
        response[:1000],
    )
    if last_error is not None:
        raise last_error
    raise ValueError("No JSON candidates found in attack plan generator response")


def process_single_behavior(
    index: int,
    row: Any,
    agent: BaseAgent,
    temperature: float,
    num_sets: int,
    strategies_per_set: int,
    prompts_path: Path,
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
            prompts_path=prompts_path,
        )
        formatted_user_prompt = rewrite_strategy_count(
            formatted_user_prompt,
            strategies_per_set,
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


def rewrite_strategy_count(prompt: str, strategies_per_set: int) -> str:
    replacement = (
        f"Generate {strategies_per_set} distinct jailbreak strategy"
        if strategies_per_set == 1
        else f"Generate {strategies_per_set} distinct jailbreak strategies"
    )
    return re.sub(
        r"Generate\s+\d+\s+distinct jailbreak strategies?",
        replacement,
        prompt,
        count=1,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate experiment attack plans with tolerant JSON parsing."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(EXPERIMENT_ROOT / "config" / "config.yaml"),
    )
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
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
            prompts_path=args.prompts,
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
