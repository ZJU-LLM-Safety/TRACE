import re
import time
from functools import lru_cache
from statistics import fmean, pstdev
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
import threading

from improved_subtask_score import get_openai_client, record_completion_tokens, template_safety_score
from improved_task_json import parse_json_list_or_object
from improved_task_prompts import (
    build_execution_difficulty_judge_prompt,
    build_granularity_judge_prompt,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_granularity_judgment(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("granularity judge output is not a JSON object")
    raw_score = result.get("granularity_score")
    if isinstance(raw_score, bool) or raw_score is None:
        raise ValueError("granularity_score must be numeric")
    raw_score = float(raw_score)
    result["raw_granularity_score"] = round(raw_score, 3)
    # Judge contract is 1-5, so >= 1.0 is a raw score needing /5; sub-1 values
    # are treated as already normalized. (raw 1 must become 0.2, not 1.0.)
    normalized = raw_score / 5.0 if raw_score >= 1.0 else raw_score
    result["granularity_score"] = round(_clamp(normalized), 3)
    result.setdefault("judgment", "")
    result.setdefault("oversplit_steps", [])
    result.setdefault("overmerged_steps", [])
    result.setdefault("reason", "")
    result["source"] = "llm"
    return result


def _normalize_difficulty_judgment(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("difficulty judge output is not a JSON object")
    raw_score = result.get("difficulty_score")
    if isinstance(raw_score, bool) or raw_score is None:
        raise ValueError("difficulty_score must be numeric")
    raw_score = float(raw_score)
    result["raw_difficulty_score"] = round(raw_score, 3)
    # Judge contract is 1-5, so >= 1.0 is a raw score needing /5; sub-1 values
    # are treated as already normalized. (raw 1 must become 0.2, not 1.0.)
    normalized = raw_score / 5.0 if raw_score >= 1.0 else raw_score
    result["difficulty_score"] = round(_clamp(normalized), 3)
    result.setdefault("judgment", "")
    result.setdefault("hardest_steps", [])
    result.setdefault("reason", "")
    result["source"] = "llm"
    return result


def _call_granularity_judge(
    task: str,
    sub_prompts: list[str],
    *,
    client,
    model: str,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": build_granularity_judge_prompt(task, sub_prompts),
            }
        ],
        temperature=0,
        top_p=0.01,
    )
    record_completion_tokens(usage_sink, completion, lock=usage_sink_lock)
    content = completion.choices[0].message.content or "{}"
    return _normalize_granularity_judgment(parse_json_list_or_object(content))


def _call_difficulty_judge(
    task: str,
    sub_prompts: list[str],
    *,
    client,
    model: str,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": build_execution_difficulty_judge_prompt(task, sub_prompts),
            }
        ],
        temperature=0,
        top_p=0.01,
    )
    record_completion_tokens(usage_sink, completion, lock=usage_sink_lock)
    content = completion.choices[0].message.content or "{}"
    return _normalize_difficulty_judgment(parse_json_list_or_object(content))


def _retry_call_granularity_judge(
    task: str,
    sub_prompts: list[str],
    *,
    client,
    model: str,
    retries: int = 10,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            return _call_granularity_judge(
                task,
                sub_prompts,
                client=client,
                model=model,
                usage_sink=usage_sink,
                usage_sink_lock=usage_sink_lock,
            )
        except Exception as exc:
            last_exc = exc
            time.sleep(2)  # Wait before retrying
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("granularity judge failed without an exception")


def _retry_call_difficulty_judge(
    task: str,
    sub_prompts: list[str],
    *,
    client,
    model: str,
    retries: int = 10,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            return _call_difficulty_judge(
                task,
                sub_prompts,
                client=client,
                model=model,
                usage_sink=usage_sink,
                usage_sink_lock=usage_sink_lock,
            )
        except Exception as exc:
            last_exc = exc
            time.sleep(2)  # Wait before retrying
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("difficulty judge failed without an exception")


def llm_granularity_score(
    task: str | None,
    sub_prompts: list[str],
    *,
    client=None,
    model: str | None = None,
    quality: dict[str, Any] | None = None,
    subtask_count: int | None = None,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> dict[str, Any]:
    """Score granularity positively; use heuristics only when judging is unavailable."""
    return _retry_call_granularity_judge(
        task,
        sub_prompts,
        client=client,
        model=model,
        retries=10,
        usage_sink=usage_sink,
        usage_sink_lock=usage_sink_lock,
    )


def llm_difficulty_score(
    task: str | None,
    sub_prompts: list[str],
    *,
    client=None,
    model: str | None = None,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> dict[str, Any]:
    """Score execution difficulty (1-5 raw, normalized 0-1; higher = harder)."""
    return _retry_call_difficulty_judge(
        task,
        sub_prompts,
        client=client,
        model=model,
        retries=10,
        usage_sink=usage_sink,
        usage_sink_lock=usage_sink_lock,
    )


@lru_cache(maxsize=4096)
def _template_harmfulness_score_cached(
    prompt: str,
    model: str | None,
    openai_api_key: str | None,
    openai_base_url: str | None,
) -> float:
    if not model:
        return 0.2
    last_exc: Exception | None = None
    for attempt in range(10):
        try:
            return float(
                template_safety_score(
                    prompt,
                    model,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                )
            )
        except Exception as exc:
            last_exc = exc
            if attempt < 9:
                time.sleep(2)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("template_safety_score failed without raising an exception")


def _template_harmfulness_score(
    prompt: str,
    model: str | None,
    openai_api_key: str | None,
    openai_base_url: str | None,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> float:
    if usage_sink is not None or usage_sink_lock is not None:
        if not model:
            return 0.2
        last_exc: Exception | None = None
        for attempt in range(10):
            try:
                return float(
                    template_safety_score(
                        prompt,
                        model,
                        openai_api_key=openai_api_key,
                        openai_base_url=openai_base_url,
                        usage_sink=usage_sink,
                        usage_sink_lock=usage_sink_lock,
                    )
                )
            except Exception as exc:
                last_exc = exc
                if attempt < 9:
                    time.sleep(2)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            "template_safety_score failed without raising an exception"
            )
    return _template_harmfulness_score_cached(
        prompt,
        model,
        openai_api_key,
        openai_base_url,
    )


def _candidate_subtask_harmfulness(
    sub_prompt: str,
    *,
    model: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    safety_scorer: Callable[[str], float] | None = None,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> float:
    if safety_scorer is not None:
        return _clamp(float(safety_scorer(sub_prompt)))
    return _clamp(
        _template_harmfulness_score(
            sub_prompt,
            model,
            openai_api_key,
            openai_base_url,
            usage_sink=usage_sink,
            usage_sink_lock=usage_sink_lock,
        )
    )


def candidate_selection_metrics(
    candidate: dict[str, Any],
    *,
    task: str | None = None,
    model: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    safety_scorer: Callable[[str], float] | None = None,
    client=None,
    selection_safety_weight: float = 0.4,
    selection_difficulty_weight: float = 0.3,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> dict[str, Any]:
    sub_prompts = list(candidate.get("sub_prompts", []))
    quality = candidate.get("boundary_quality", {})
    boundary_score = float(quality.get("score", 0.0) or 0.0)
    subtask_count = int(quality.get("subtask_count", len(sub_prompts)) or 0)
    safety_weight = _clamp(float(selection_safety_weight))
    difficulty_weight = _clamp(
        float(selection_difficulty_weight), 0.0, 1.0 - safety_weight
    )
    granularity_weight = 1.0 - safety_weight - difficulty_weight
    local_usage_lock = usage_sink_lock
    if usage_sink is not None and local_usage_lock is None:
        local_usage_lock = threading.Lock()

    if len(sub_prompts) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(sub_prompts))) as executor:
            futures = [
                executor.submit(
                    _candidate_subtask_harmfulness,
                    sub_prompt,
                    model=model,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                    safety_scorer=safety_scorer,
                    usage_sink=usage_sink,
                    usage_sink_lock=local_usage_lock,
                )
                for sub_prompt in sub_prompts
            ]
            harmful_scores = [future.result() for future in futures]
    else:
        harmful_scores = [
            _candidate_subtask_harmfulness(
                sub_prompt,
                model=model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                safety_scorer=safety_scorer,
                usage_sink=usage_sink,
                usage_sink_lock=local_usage_lock,
            )
            for sub_prompt in sub_prompts
        ]
    if harmful_scores:
        safety_score = 1.0 - fmean(harmful_scores)
    else:
        harmful_scores = []
        safety_score = 0.0

    granularity_judgment = llm_granularity_score(
        task,
        sub_prompts,
        client=client,
        model=model,
        quality=quality,
        subtask_count=subtask_count,
        usage_sink=usage_sink,
        usage_sink_lock=local_usage_lock,
    )
    granularity_score = granularity_judgment["granularity_score"]
    difficulty_judgment = llm_difficulty_score(
        task,
        sub_prompts,
        client=client,
        model=model,
        usage_sink=usage_sink,
        usage_sink_lock=local_usage_lock,
    )
    difficulty_score = difficulty_judgment["difficulty_score"]
    executability_score = round(1.0 - difficulty_score, 3)
    final_score = round(
        safety_weight * safety_score
        + granularity_weight * granularity_score
        + difficulty_weight * executability_score,
        4,
    )

    return {
        "final_score": final_score,
        "safety_score": safety_score,
        "granularity_score": granularity_score,
        "granularity_judgment": granularity_judgment,
        "difficulty_score": difficulty_score,
        "executability_score": executability_score,
        "difficulty_judgment": difficulty_judgment,
        "boundary_score": round(boundary_score, 3),
        "subtask_count": subtask_count,
        "harmful_scores": [round(score, 3) for score in harmful_scores],
        "selection_safety_weight": round(safety_weight, 3),
        "selection_granularity_weight": round(granularity_weight, 3),
        "selection_difficulty_weight": round(difficulty_weight, 3),
    }


def _scored_valid_candidate_pairs(
    valid_candidates: list[dict[str, Any]],
    *,
    task: str | None = None,
    model: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    safety_scorer: Callable[[str], float] | None = None,
    client=None,
    selection_safety_weight: float = 0.4,
    selection_difficulty_weight: float = 0.3,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    judge_client = client
    if judge_client is None and task and model:
        try:
            judge_client = get_openai_client(
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
            )
        except Exception:
            judge_client = None

    return [
        (
            candidate,
            candidate_selection_metrics(
                candidate,
                task=task,
                model=model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                safety_scorer=safety_scorer,
                client=judge_client,
                selection_safety_weight=selection_safety_weight,
                selection_difficulty_weight=selection_difficulty_weight,
                usage_sink=usage_sink,
                usage_sink_lock=usage_sink_lock,
            ),
        )
        for candidate in valid_candidates
    ]


def score_valid_candidates(
    valid_candidates: list[dict[str, Any]],
    *,
    task: str | None = None,
    model: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    safety_scorer: Callable[[str], float] | None = None,
    client=None,
    selection_safety_weight: float = 0.4,
    selection_difficulty_weight: float = 0.3,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> list[dict[str, Any]]:
    scored = []
    for index, (candidate, metrics) in enumerate(
        _scored_valid_candidate_pairs(
            valid_candidates,
            task=task,
            model=model,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            safety_scorer=safety_scorer,
            client=client,
            selection_safety_weight=selection_safety_weight,
            selection_difficulty_weight=selection_difficulty_weight,
            usage_sink=usage_sink,
            usage_sink_lock=usage_sink_lock,
        )
    ):
        scored_candidate = dict(candidate)
        scored_candidate["candidate_index"] = index
        scored_candidate["selection_metrics"] = metrics
        scored.append(scored_candidate)
    return scored


def build_candidate_process(
    raw_candidates: list[dict[str, Any]],
    valid_candidates: list[dict[str, Any]] | None = None,
    selected_sub_prompts: list[str] | None = None,
) -> list[dict[str, Any]]:
    valid_metric_map: dict[tuple[str, ...], dict[str, Any] | None] = {}
    for candidate in valid_candidates or []:
        valid_metric_map[tuple(candidate.get("sub_prompts", []))] = candidate.get(
            "selection_metrics"
        )

    selected_key = tuple(selected_sub_prompts or [])
    candidate_process: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        candidate_sub_prompts = list(
            candidate.get("candidate_sub_prompts")
            or candidate.get("sub_prompts")
            or []
        )
        candidate_key = tuple(candidate_sub_prompts)
        candidate_process.append(
            {
                "candidate": candidate_sub_prompts,
                "valid_result": candidate.get("boundary_quality"),
                "selection_metrics": valid_metric_map.get(candidate_key),
                "selected": candidate_key == selected_key,
            }
        )
    return candidate_process


def select_best_valid_candidate(
    valid_candidates: list[dict[str, Any]],
    *,
    task: str | None = None,
    model: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    safety_scorer: Callable[[str], float] | None = None,
    client=None,
    selection_safety_weight: float = 0.4,
    selection_difficulty_weight: float = 0.3,
    include_selection_metrics: bool = False,
    usage_sink: dict[str, int] | None = None,
    usage_sink_lock=None,
) -> dict[str, Any] | None:
    if not valid_candidates:
        return None

    scored_candidates = _scored_valid_candidate_pairs(
        valid_candidates,
        task=task,
        model=model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        safety_scorer=safety_scorer,
        client=client,
        selection_safety_weight=selection_safety_weight,
        selection_difficulty_weight=selection_difficulty_weight,
        usage_sink=usage_sink,
        usage_sink_lock=usage_sink_lock,
    )
    best_candidate, best_metrics = max(
        scored_candidates,
        key=lambda item: (
            item[1]["final_score"],
            item[1]["granularity_score"],
            item[1]["executability_score"],
            item[1]["safety_score"],
            item[1]["boundary_score"],
            -item[1]["subtask_count"],
        ),
    )
    if include_selection_metrics:
        best_candidate = dict(best_candidate)
        best_candidate["selection_metrics"] = best_metrics
    return best_candidate
