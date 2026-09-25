import argparse
import json
import os
import time
from typing import Any

from improved_subtask_score import (
    ensure_parent_dir,
    get_openai_client,
    record_completion_tokens,
    semantic_equivalent_score,
)
from improved_task_critique import (
    combine_boundary_quality,
    enforce_critique_validity,
    normalize_critique_issue_types,
    relax_overstrict_critique,
)
from improved_task_json import parse_json_list, parse_json_list_or_object
from improved_task_llm import (
    evaluate_llm_quality_gate,
    llm_decomposition_quality_score,
    refine_with_llm_quality_gate,
)
from improved_task_prompts import build_decompose_prompt
from improved_task_rules import *
from improved_task_schema import decompose_task_with_schema
from improved_task_selection import build_candidate_process, score_valid_candidates


def debug_print(label: str, value: Any) -> None:
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        rendered = str(value)
    print(f"{label}: {rendered}", flush=True)


def decompose_task(
    task: str,
    model: str = "gpt-4o-2024-11-20",
    num_decompositions_per_task=5,
    *,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    item_context: dict[str, Any] | None = None,
    require_llm_equivalence: bool = False,
    use_llm_quality_refinement: bool = True,
    quality_refinement_rounds: int = 1,
    max_regeneration_attempts: int = 3,
    selection_safety_weight: float = 0.5,
    use_schema_decompose: bool = False,
    use_local_rules: bool = True,
    schema_names: list[str] | None = None,
    return_trace: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    if use_schema_decompose or schema_names:
        return decompose_task_with_schema(
            task,
            model=model,
            num_decompositions_per_task=num_decompositions_per_task,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            item_context=item_context,
            require_llm_equivalence=require_llm_equivalence,
            use_llm_quality_refinement=use_llm_quality_refinement,
            quality_refinement_rounds=quality_refinement_rounds,
            max_regeneration_attempts=max_regeneration_attempts,
            selection_safety_weight=selection_safety_weight,
            use_local_rules=use_local_rules,
            schema_names=schema_names,
            return_trace=return_trace,
        )
    client = get_openai_client(
        openai_api_key=openai_api_key, openai_base_url=openai_base_url
    )
    valid_candidates: list[dict[str, Any]] = []
    generation_trace: dict[str, Any] = {
        "task": task,
        "model": model,
        "raw_candidates": [],
        "valid_candidates": [],
        "decomposition_token_cost": 0,
        "selection_safety_weight": selection_safety_weight,
        "quality_refinement_enabled": bool(use_llm_quality_refinement and quality_refinement_rounds > 0),
        "quality_refinement_rounds": quality_refinement_rounds,
        "require_llm_equivalence": require_llm_equivalence,
        "local_rules_enabled": use_local_rules,
    }
    seen_raw_sub_prompts: set[tuple[str, ...]] = set()
    num_try_max = max(
        num_decompositions_per_task,
        num_decompositions_per_task * (max_regeneration_attempts + 1),
    )
    for _ in range(num_try_max):
        if len(seen_raw_sub_prompts) == num_decompositions_per_task:
            break

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": build_decompose_prompt(task),
                }
            ],
            temperature=0.5,
        )
        record_completion_tokens(generation_trace, completion)
        content = completion.choices[0].message.content
        if content is None:
            continue

        try:
            raw_sub_prompts = parse_json_list(content)
            raw_sub_prompt_key = tuple(raw_sub_prompts)
            if raw_sub_prompt_key in seen_raw_sub_prompts:
                continue
            seen_raw_sub_prompts.add(raw_sub_prompt_key)
            sub_prompts = (
                refine_sub_prompts(raw_sub_prompts, original_task=task)
                if use_local_rules
                else list(raw_sub_prompts)
            )
        except Exception as exc:
            debug_print("Failed to parse decomposition", exc)
            debug_print("Raw Decomposition Response", content)
            continue

        debug_print("Raw Model Sub-Prompts", raw_sub_prompts)
        debug_print("After Local Rules", sub_prompts)
        local_quality = None
        candidate_variants: list[tuple[list[str], dict[str, Any]]] = []

        if use_llm_quality_refinement and quality_refinement_rounds > 0:
            try:
                llm_quality, quality = evaluate_llm_quality_gate(
                    task,
                    sub_prompts,
                    client=client,
                    model=model,
                    usage_sink=generation_trace,
                )
                debug_print("LLM Quality Critique", llm_quality)
                if quality.get("valid"):
                    candidate_variants.append((sub_prompts, quality))
                else:
                    refined_sub_prompts, quality_rounds, refined_quality = refine_with_llm_quality_gate(
                        task,
                        sub_prompts,
                        llm_quality,
                        quality,
                        client=client,
                        model=model,
                        max_rounds=quality_refinement_rounds,
                        local_quality=local_quality,
                        usage_sink=generation_trace,
                    )
                    debug_print("LLM Quality Rounds", quality_rounds)
                    if refined_quality.get("valid"):
                        candidate_variants.append((refined_sub_prompts, refined_quality))
                    sub_prompts = refined_sub_prompts
                    quality = refined_quality
            except Exception as exc:
                debug_print("LLM quality refinement failed", exc)
                continue
        else:
            llm_quality = llm_decomposition_quality_score(
                task,
                sub_prompts,
                client=client,
                model=model,
                usage_sink=generation_trace,
            )
            debug_print("LLM Quality Critique", llm_quality)
            quality = combine_boundary_quality(None, llm_quality, sub_prompts, task)
            if quality.get("valid"):
                candidate_variants.append((sub_prompts, quality))

        debug_print("Original Task", task)
        debug_print("Decomposed Sub-Prompts", sub_prompts)
        debug_print("Boundary Quality", quality)
        if not candidate_variants:
            generation_trace["raw_candidates"].append(
                {
                    "raw_sub_prompts": raw_sub_prompts,
                    "post_local_rules_sub_prompts": sub_prompts,
                    "local_quality": local_quality,
                    "candidate_sub_prompts": sub_prompts,
                    "boundary_quality": quality,
                    "valid_after_boundary_quality": bool(quality.get("valid")),
                    "accepted_for_selection": False,
                }
            )
        for candidate_sub_prompts, candidate_quality in candidate_variants:
            candidate_trace = {
                "raw_sub_prompts": raw_sub_prompts,
                "post_local_rules_sub_prompts": sub_prompts,
                "local_quality": local_quality,
                "candidate_sub_prompts": candidate_sub_prompts,
                "boundary_quality": candidate_quality,
                "valid_after_boundary_quality": bool(candidate_quality.get("valid")),
                "accepted_for_selection": False,
            }
            if require_llm_equivalence:
                equivalence = parse_json_list_or_object(
                    semantic_equivalent_score(
                        task,
                        " ".join(candidate_sub_prompts),
                        model=model,
                        openai_api_key=openai_api_key,
                        openai_base_url=openai_base_url,
                    )
                )
                if not equivalence.get("equivalent"):
                    debug_print("Semantic Equivalence", False)
                    debug_print("Reason", equivalence.get("reason"))
                    candidate_trace["equivalence"] = equivalence
                    generation_trace["raw_candidates"].append(candidate_trace)
                    continue
                candidate_trace["equivalence"] = equivalence

            if not candidate_quality.get("valid"):
                generation_trace["raw_candidates"].append(candidate_trace)
                continue
            sub_prompt_key = tuple(candidate_sub_prompts)
            if any(
                sub_prompt_key == tuple(item["sub_prompts"]) for item in valid_candidates
            ):
                candidate_trace["rejected_reason"] = "duplicate_valid_candidate"
                generation_trace["raw_candidates"].append(candidate_trace)
                continue
            valid_candidates.append(
                {
                    "sub_prompts": candidate_sub_prompts,
                    "boundary_quality": candidate_quality,
                }
            )
            candidate_trace["accepted_for_selection"] = True
            generation_trace["raw_candidates"].append(candidate_trace)

    if not valid_candidates:
        return ([], generation_trace) if return_trace else []

    scored_valid_candidates = score_valid_candidates(
        valid_candidates,
        task=task,
        model=model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        client=client,
        selection_safety_weight=selection_safety_weight,
        usage_sink=generation_trace,
    )
    generation_trace["valid_candidates"] = scored_valid_candidates
    best_candidate, best_metrics = max(
        (
            (candidate, candidate["selection_metrics"])
            for candidate in scored_valid_candidates
        ),
        key=lambda item: item[1]["final_score"],
    )
    best_candidate = dict(best_candidate)
    best_candidate["selection_metrics"] = best_metrics
    best_candidate["decomposition_trace"] = generation_trace
    debug_print("Selected Candidate", best_candidate)
    result = [best_candidate] if best_candidate is not None else []
    return (result, generation_trace) if return_trace else result

def prompt_field_for_path(data_file_path: str) -> str:
    lower = data_file_path.lower()
    if "agentharm" in lower:
        return "prompt"
    if "end2end" in lower or "advcua" in lower:
        return "prompt_to_os_agent"
    if "practical" in lower:
        return "harmful"
    return "prompt"


def prompt_field_for_item(data_file_path: str, item: dict[str, Any]) -> str:
    preferred = prompt_field_for_path(data_file_path)
    if preferred in item:
        return preferred
    for candidate in ("prompt_to_os_agent", "harmful", "prompt"):
        if candidate in item:
            return candidate
    raise KeyError(
        "No supported prompt field found. Expected one of: "
        "prompt_to_os_agent, harmful, prompt."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safety-aware decomposition into executable sub-prompts."
    )
    parser.add_argument("--data_file_path", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--num_decompositions_per_task", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--target_ids", nargs="*", default=None)
    parser.add_argument(
        "--require_llm_equivalence",
        action="store_true",
        help="Also run LLM semantic/dependency judges after local quality checks.",
    )
    parser.add_argument(
        "--quality_refinement_rounds",
        type=int,
        default=1,
        help="Whether to allow one LLM critic-guided revision for each generated candidate. Values above 1 are treated as enabled, not as repeated revision rounds.",
    )
    parser.add_argument(
        "--max_regeneration_attempts",
        type=int,
        default=3,
        help="Maximum number of times to regenerate a fresh candidate after one revision still fails the critic.",
    )
    parser.add_argument(
        "--disable_llm_quality_refinement",
        action="store_true",
        help="Disable the LLM critic/revision quality gate.",
    )
    parser.add_argument(
        "--selection_safety_weight",
        type=float,
        default=0.5,
        help="Weight of safety in final valid-candidate selection; granularity gets 1 minus this weight.",
    )
    parser.add_argument(
        "--use_schema_decompose",
        action="store_true",
        help="Use hand-authored dataset schemas for decomposition instead of the default LLM generator.",
    )
    parser.add_argument(
        "--schema_names",
        nargs="+",
        default=None,
        help=(
            "Force one or more named schemas and skip keyword matching. "
            "Implies --use_schema_decompose."
        ),
    )
    parser.add_argument(
        "--disable_local_rules",
        action="store_true",
        help="Skip refine_sub_prompts and evaluate the raw LLM sub-prompts directly.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_ids = set(args.target_ids) if args.target_ids else None

    with open(args.data_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_ids = set()
    if os.path.exists(args.output_file_path):
        with open(args.output_file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                existing_ids.add(item["id"])

    ensure_parent_dir(args.output_file_path)
    for idx, item in enumerate(data):
        raw_item_id = item.get("id", item.get("challenge", idx))
        item_id = str(raw_item_id)
        if target_ids is not None and item_id not in target_ids:
            continue
        if raw_item_id in existing_ids or item_id in existing_ids:
            debug_print("Skipping existing task id", raw_item_id)
            continue
        prompt_field = prompt_field_for_item(args.data_file_path, item)

        decomposition_started_at = time.perf_counter()
        decomposition_result = decompose_task(
            item[prompt_field],
            model=args.model,
            num_decompositions_per_task=args.num_decompositions_per_task,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
            item_context=item,
            require_llm_equivalence=args.require_llm_equivalence,
            use_llm_quality_refinement=not args.disable_llm_quality_refinement,
            quality_refinement_rounds=args.quality_refinement_rounds,
            max_regeneration_attempts=args.max_regeneration_attempts,
            selection_safety_weight=args.selection_safety_weight,
            use_schema_decompose=args.use_schema_decompose,
            use_local_rules=not args.disable_local_rules,
            schema_names=args.schema_names,
            return_trace=True,
        )
        decomposition_time = round(time.perf_counter() - decomposition_started_at, 3)
        if isinstance(decomposition_result, tuple):
            decomposition_records, decomposition_trace = decomposition_result
        else:
            decomposition_records = decomposition_result
            decomposition_trace = None

        if not decomposition_records:
            result = {
                "id": raw_item_id,
                "prompt": item[prompt_field],
                "status": "no_valid_candidate",
                "sub_prompts": [],
                "candidate_process": build_candidate_process(
                    (decomposition_trace or {}).get("raw_candidates", [])
                ),
                "decomposition_time": decomposition_time,
                "decomposition_token_cost": (decomposition_trace or {}).get(
                    "decomposition_token_cost", 0
                ),
            }
            with open(args.output_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            continue

        for decomposition_record in decomposition_records:
            sub_prompts = decomposition_record["sub_prompts"]
            trace = decomposition_record.get("decomposition_trace") or decomposition_trace or {}
            result = {
                "id": raw_item_id,
                "prompt": item[prompt_field],
                "sub_prompts": sub_prompts,
                "candidate_process": build_candidate_process(
                    trace.get("raw_candidates", []),
                    trace.get("valid_candidates", []),
                    sub_prompts,
                ),
                "decomposition_time": decomposition_time,
                "decomposition_token_cost": trace.get("decomposition_token_cost", 0),
            }
            with open(args.output_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
