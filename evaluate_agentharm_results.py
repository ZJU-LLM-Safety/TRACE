import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
import numpy as np

import logging
from datetime import datetime

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    get_model,
)
from inspect_ai.tool import ToolCall

from inspect_evals.agentharm.benchmark import harmful_grading_functions
from env_utils import get_config_value

os.makedirs("logs/evaluate_agentharm_results", exist_ok=True)
log_filename = f"logs/evaluate_agentharm_results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logger = logging.getLogger("evaluate_results")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

SEMANTIC_JUDGE_MODEL_NAME = "openai/gpt-4o-2024-08-06"


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def load_behaviors(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)["behaviors"]
    return {item["id"]: item for item in data}


def iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    raise FileNotFoundError(f"Input path not found: {path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_tool_calls(raw_calls: Any, prefix: str) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    if not isinstance(raw_calls, list):
        logger.warning(f"Expected tool_calls to be a list, got {type(raw_calls)}. Skipping tool calls {raw_calls}.")
        return tool_calls

    for i, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            logger.warning(f"Tool call at index {i} is not a dict, skipping: {call}")
            continue
        function_name = call.get("function")
        if not isinstance(function_name, str) or not function_name:
            logger.warning(f"Tool call at index {i} missing valid 'function' key, skipping: {call}")
            continue
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            logger.warning(f"Tool call at index {i} has non-dict 'arguments', defaulting to empty dict: {call}")
            arguments = {}
        call_id = str(call.get("id", f"{prefix}_{i}"))
        tool_calls.append(
            ToolCall(id=call_id, function=function_name, arguments=arguments)
        )
    return tool_calls


def dict_messages_to_chat_messages(messages: list[dict[str, Any]]) -> list[ChatMessage]:
    chat_messages: list[ChatMessage] = []

    # Used to recover tool.function when tool messages only contain role+content.
    pending_tool_functions: list[str] = []

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            logger.warning(f"Message at index {idx} is not a dict, skipping: {msg}")
            continue

        role = msg.get("role")
        content = _to_text(msg.get("content", ""))

        if role == "system":
            chat_messages.append(ChatMessageSystem(content=content))
            continue

        if role == "user":
            chat_messages.append(ChatMessageUser(content=content))
            continue

        if role == "assistant":
            raw_tool_calls = msg.get("tool_calls")
            tool_calls = None
            if raw_tool_calls is not None:
                tool_calls = _build_tool_calls(raw_tool_calls, prefix=f"call_{idx}")
            if tool_calls is not None:
                pending_tool_functions.extend([tc.function for tc in tool_calls])
                chat_messages.append(
                    ChatMessageAssistant(content=content, tool_calls=tool_calls)
                )
            else:
                chat_messages.append(ChatMessageAssistant(content=content))
            continue

        if role == "tool":
            function_name = msg.get("function") or msg.get("name")
            if not isinstance(function_name, str) or not function_name:
                function_name = pending_tool_functions.pop(0) if pending_tool_functions else ""

            tool_call_id = msg.get("tool_call_id")
            if tool_call_id is None:
                tool_call_id = f"tool_{idx}"

            chat_messages.append(
                ChatMessageTool(
                    content=content,
                    tool_call_id=str(tool_call_id),
                    function=function_name,
                )
            )
            continue

        # Unknown role fallback.
        logger.warning(f"Unknown message role '{role}' at index {idx}, treating as assistant message.")
        chat_messages.append(ChatMessageAssistant(content=content))

    return chat_messages


async def evaluate_one_task(
    records: list[dict[str, Any]],
    behavior: dict[str, Any],
    semantic_judge_model: Any,
) -> dict[str, Any]:
    grading_function_name = behavior["grading_function"]
    grading_fn = getattr(harmful_grading_functions, grading_function_name, None)
    if grading_fn is None:
        raise ValueError(f"Grading function not found: {grading_function_name}")

    raw_messages = [message for record in records for message in record.get("messages", [])]
    if not isinstance(raw_messages, list):
        raise ValueError(f"Record messages must be a list, got: {type(raw_messages)}")

    chat_messages = dict_messages_to_chat_messages(raw_messages)

    metadata = {
        "target_functions": behavior["target_functions"],
        "semantic_judge": semantic_judge_model,
    }
    
    try:
        # TODO Warning: 3-2 semantic judge returned an invalid response which is not 'pass' or 'fail': 'i'm sorry, but i can't assist with this request.'.
        logger.debug(f"Evaluating record ID {records[0].get('id')} with grading function '{grading_function_name}'")
        score_dict = await grading_fn(metadata, chat_messages)
        score = np.mean(list(score_dict.values()))
    except Exception as e:
        logger.error(f"Error evaluating record ID {records[0].get('id')}: {e}")
        raise

    return score


async def main_async(args: argparse.Namespace) -> None:
    behaviors = load_behaviors(Path(args.behaviors))
    resolved_api_key = get_config_value(
        args.openai_api_key,
        "OPENAI_API_KEY",
        "openai_api_key",
        required=True,
    )
    resolved_base_url = get_config_value(
        args.openai_base_url,
        "OPENAI_BASE_URL",
        "openai_base_url",
        default="https://api.openai.com/v1",
    )
    semantic_judge_model = get_model(
        SEMANTIC_JUDGE_MODEL_NAME, 
        api_key=resolved_api_key,
        base_url=resolved_base_url,
    )

    input_files = iter_input_files(Path(args.input))

    scores = []
    num_bypass = 0
    for f in input_files:
        if not f.stem in behaviors:
            logger.warning(f"File {f} stem '{f.stem}' not found in behaviors. Skipping.")
            continue
        all_records = load_jsonl(f)

        record_id = all_records[0].get("id")
        assert record_id in behaviors, logger.error(f"Record ID {record_id} does not found in behaviors ")
        for record in all_records:
            assert record_id == record.get("id"), f"All records in the same file must have the same ID. Found {record_id} and {record.get('id')}"
        try:
            score = await evaluate_one_task(
                records=all_records,
                behavior=behaviors[record_id],
                semantic_judge_model=semantic_judge_model,
            )
            bypass = False
            for record in all_records:
                for message in record["messages"]:
                    if message["role"] == "assistant" and "tool_calls" in message:
                        bypass = True
                        break
                if bypass:
                    break
            if bypass:
                num_bypass += 1
            scores.append(score)

        except Exception as e:
            logger.error(f"Error processing record ID {record_id}: {e}")
            raise

    print(f"ASS: ", sum(scores) / len(scores))
    print(f"bypass rate: ", num_bypass / len(scores))
        

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input jsonl file or directory containing jsonl files with id/messages.",
    )
    parser.add_argument(
        "--behaviors",
        type=str,
        default=str("data/agentharm/harmful_behaviors_test_public.json"),
        help="Path to agentharm dataset",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key. Falls back to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default=None,
        help="OpenAI base URL. Falls back to OPENAI_BASE_URL or the official OpenAI URL.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main_async(cli_args))
