import json
from typing import Any, Literal

from env_utils import get_config_value


def _extract_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
        return "\n".join([text for text in texts if text]).strip()
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    return str(value)


def _to_message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text = _extract_content_text(value)
        if text:
            return text
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    if isinstance(value, (dict, int, float, bool)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_sample_input_with_history(
    candidate: dict[str, Any],
    prompt_field: str,
    history_field: str = "history",
):
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

    history = candidate.get(history_field, [])
    if not isinstance(history, list) or len(history) == 0:
        return candidate[prompt_field]

    messages = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = _to_message_content(msg.get("content", ""))
        if role == "system":
            messages.append(ChatMessageSystem(content=content))
        elif role == "user":
            messages.append(ChatMessageUser(content=content))
        elif role == "assistant":
            if isinstance(msg.get("tool_calls"), list) and len(msg["tool_calls"]) > 0:
                tool_names = [
                    tc.get("function")
                    for tc in msg["tool_calls"]
                    if isinstance(tc, dict) and isinstance(tc.get("function"), str)
                ]
                if tool_names:
                    content = f"{content}\n\n[history tool calls: {', '.join(tool_names)}]"
            messages.append(ChatMessageAssistant(content=content))
        elif role == "tool":
            function_name = msg.get("function") or msg.get("name") or "tool"
            messages.append(ChatMessageAssistant(content=f"[{function_name} result] {content}"))

    messages.append(ChatMessageUser(content=_to_message_content(candidate[prompt_field]), source="input"))
    return messages


def run_agentharm_once_without_score(
    candidates,
    model_name,
    field: Literal["initialized_prompt", "prompt", "jailbroken_prompt"] = "initialized_prompt",
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    deepseek_api_key: str | None = None,
    deepseek_base_url: str | None = None,
):
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig
    from inspect_evals.agentharm.agents.agent import get_agent
    from inspect_evals.agentharm.utils import get_benchmark_tools, setup_tools_from_metadata
    from inspect_evals.metadata import load_eval_metadata

    eval_version = load_eval_metadata("agentharm").version
    all_tool_names = get_benchmark_tools("harmful")
    agent_solver = get_agent(agent="default")
    plan = [setup_tools_from_metadata("harmful", all_tool_names, 0), agent_solver]

    if isinstance(candidates, dict):
        sample = Sample(
            input=build_sample_input_with_history(candidates, field),
            id=candidates["id"],
            metadata={key: candidates[key] for key in ["id", field]},
        )
        dataset = MemoryDataset([sample])
        task = Task(
            dataset=dataset,
            solver=plan,
            max_messages=30,
            config=GenerateConfig(
                max_retries=3,
                max_tokens=4096,
                temperature=0.0 if model_name != "gpt-5.3-chat" else None,
                max_connections=100,
                seed=0,
            ),
            version=eval_version.comparability_version,
            metadata=eval_version.to_metadata(),
        )
        if "deepseek" in model_name and "v4" in model_name:
            resolved_api_key = get_config_value(
                deepseek_api_key,
                "DEEPSEEK_API_KEY",
                "deepseek_api_key",
                required=True,
            )
            resolved_base_url = get_config_value(
                deepseek_base_url,
                "DEEPSEEK_BASE_URL",
                "deepseek_base_url",
                default="https://api.deepseek.com",
            )
            logs = inspect_eval(
                tasks=task,
                model=f"openai/{model_name}",
                model_base_url=resolved_base_url,
                model_args={"api_key": resolved_api_key},
            )
        else:
            resolved_api_key = get_config_value(
                openai_api_key,
                "OPENAI_API_KEY",
                "openai_api_key",
                required=True,
            )
            resolved_base_url = get_config_value(
                openai_base_url,
                "OPENAI_BASE_URL",
                "openai_base_url",
                default="https://api.openai.com/v1",
            )
            logs = inspect_eval(
                tasks=task,
                model=f"openai/{model_name}",
                model_base_url=resolved_base_url,
                model_args={"api_key": resolved_api_key},
            )

        eval_log = logs[0]
        assert len(eval_log.samples) == 1
        sample_log = eval_log.samples[0]
        messages = []
        for message in sample_log.messages:
            if "tool_calls" in type(message).model_fields and message.tool_calls is not None:
                tool_calls = [{"function": tool_call.function, "arguments": tool_call.arguments} for tool_call in message.tool_calls]
                messages.append({"role": message.role, "content": "", "tool_calls": tool_calls})
            else:
                messages.append({"role": message.role, "content": _extract_content_text(message.content)})
        candidates["messages"] = messages
        return [candidates]

    raise NotImplementedError("run_agentharm_once_without_score currently supports dict candidate only.")
