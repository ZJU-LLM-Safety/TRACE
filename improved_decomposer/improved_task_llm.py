from typing import Any

from improved_task_critique import *
from improved_task_json import parse_json_list, parse_json_list_or_object
from improved_task_prompts import (
    build_decompose_prompt,
    build_quality_judge_prompt,
    build_revision_prompt,
)
from improved_task_rules import refine_sub_prompts
from improved_subtask_score import record_completion_tokens


def llm_decomposition_quality_score(
    task: str,
    sub_prompts: list[str],
    *,
    client,
    model: str,
    usage_sink: dict[str, int] | None = None,
) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": build_quality_judge_prompt(task, sub_prompts),
            }
        ],
        temperature=0,
        top_p=0.01,
    )
    content = completion.choices[0].message.content or "{}"
    record_completion_tokens(usage_sink, completion)
    parsed = parse_json_list_or_object(content)
    if not isinstance(parsed, dict):
        raise ValueError("quality judge output is not a JSON object")
    parsed.setdefault("valid", False)
    parsed.setdefault("issues", [])
    parsed.setdefault("revision_guidance", [])
    normalize_critique_issue_types(parsed)
    relax_overstrict_critique(parsed, sub_prompts)
    enforce_critique_validity(parsed)
    return parsed


def revise_decomposition_with_critique(
    task: str,
    sub_prompts: list[str],
    critique: dict[str, Any],
    *,
    client,
    model: str,
    usage_sink: dict[str, int] | None = None,
) -> list[str]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": build_revision_prompt(task, sub_prompts, critique),
            }
        ],
        temperature=0.1,
    )
    content = completion.choices[0].message.content
    record_completion_tokens(usage_sink, completion)
    if content is None:
        raise ValueError("empty revision response")
    revised = parse_json_list(content)
    return refine_sub_prompts(revised, original_task=task)


def evaluate_llm_quality_gate(
    task: str,
    sub_prompts: list[str],
    *,
    client,
    model: str,
    local_quality: dict[str, Any] | None = None,
    usage_sink: dict[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    critique = llm_decomposition_quality_score(
        task,
        sub_prompts,
        client=client,
        model=model,
        usage_sink=usage_sink,
    )
    combined_quality = combine_boundary_quality(
        local_quality,
        critique,
        sub_prompts,
        task,
    )
    return critique, combined_quality


def _build_quality_gate_record(
    *,
    phase: str,
    before: list[str],
    local_quality: dict[str, Any],
    critique: dict[str, Any],
    combined_quality: dict[str, Any],
    action: str,
    after_revision: list[str] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "phase": phase,
        "before": before,
        "local_quality": local_quality,
        "critique": critique,
        "combined_quality": combined_quality,
        "action": action,
    }
    if after_revision is not None:
        record["after_revision"] = after_revision
    return record


def refine_with_llm_quality_gate(
    task: str,
    sub_prompts: list[str],
    critique: dict[str, Any],
    combined_quality: dict[str, Any],
    *,
    client,
    model: str,
    max_rounds: int,
    local_quality: dict[str, Any] | None = None,
    usage_sink: dict[str, int] | None = None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    rounds = []
    current = sub_prompts
    initial_record = _build_quality_gate_record(
        phase="initial_candidate",
        before=current,
        local_quality=local_quality,
        critique=critique,
        combined_quality=combined_quality,
        action="accepted" if combined_quality.get("valid") is True else "pending_revision",
    )
    if combined_quality.get("valid") is True:
        rounds.append(initial_record)
        return current, rounds, combined_quality

    if max_rounds <= 0:
        initial_record["action"] = "rejected_without_revision"
        rounds.append(initial_record)
        return current, rounds, combined_quality

    revised = revise_decomposition_with_critique(
        task,
        current,
        major_only_critique(combined_quality),
        client=client,
        model=model,
        usage_sink=usage_sink,
    )
    initial_record["after_revision"] = revised
    initial_record["action"] = "revised_once"
    rounds.append(initial_record)

    current = revised
    critique, final_quality = evaluate_llm_quality_gate(
        task,
        current,
        client=client,
        model=model,
        local_quality=local_quality,
        usage_sink=usage_sink,
    )
    rounds.append(
        _build_quality_gate_record(
            phase="after_one_revision",
            before=current,
            local_quality=local_quality,
            critique=critique,
            combined_quality=final_quality,
            action="accepted" if final_quality.get("valid") is True else "rejected_for_regeneration",
        )
    )
    return current, rounds, final_quality
