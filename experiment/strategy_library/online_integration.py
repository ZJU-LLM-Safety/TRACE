"""Flag-gated integration between the online runner and the strategy library.

Enabled by passing --strategy-library <path> to experiment/main.py.
All integration points are no-ops when the flag is absent.

Wiring:
- Pool enrichment: before item attempts, each subtask is classified and the
  best library entry is instantiated into a baseline strategy prepended to the
  generated pool (pool entries shift by one), so retries still fall back to
  generated strategies.
- Outcome recording: after each attempt's subtask judge, library-sourced
  strategies get success/failure recorded against their entry; failures are
  attributed a coarse failure mode ("conversation-no-compliance" when the
  conversation never achieved target compliance, else
  "execution-not-completed") so counts aggregate past the revision threshold.
- Success re-injection: newly successful subtasks whose strategy was generated
  (not library-sourced) go through filter + extract and are added as new
  entries, deduplicated by pattern name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from library_store import StrategyLibrary
from retriever import StrategyRetriever
from instantiate import instantiate_strategy
from extract_patterns_from_success import extract_record, filter_record


class StrategyLibraryRuntime:
    """Bundles the store and retriever plus per-run bookkeeping."""

    def __init__(self, library: StrategyLibrary, retriever: StrategyRetriever):
        self.library = library
        self.retriever = retriever
        self.last_instantiations: dict[int, dict] = {}
        self.reinjected_subtasks: set[tuple[str, int]] = set()


def load_strategy_library_runtime(
    library_path: Path,
    *,
    model: str = "gpt-4o-2024-11-20",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> StrategyLibraryRuntime:
    library = StrategyLibrary(Path(library_path))
    if not library.entries:
        raise ValueError(f"strategy library {library_path} has no entries")
    retriever = StrategyRetriever(
        library,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )
    return StrategyLibraryRuntime(library=library, retriever=retriever)


def compose_library_plans(
    *,
    handle: StrategyLibraryRuntime,
    selected: list[tuple[int, str]],
    item_id: str,
    pool_size: int,
    max_per_subtask: int,
    fit_threshold: float = 4.0,
) -> dict[int, dict]:
    """Decide pool composition per subtask before any generation happens.

    Returns {subtask_index: {library_strategies, library_meta, generate_count,
    task_type, notes}}. Library quota is taken only from candidates whose
    score clears fit_threshold (a known task_type fit); generic-only matches
    generate the full pool.
    """
    composition: dict[int, dict] = {}
    for subtask_index, subtask in selected:
        target = str(subtask)
        result = handle.retriever.select(
            item_id=item_id,
            subtask_index=subtask_index,
            target=target,
        )
        fitting = [item for item in result.selected if item.score >= fit_threshold]
        library_quota = min(max_per_subtask, len(fitting)) if fitting else 0
        library_strategies = []
        library_meta = []
        for candidate in fitting[:library_quota]:
            strategy = instantiate_strategy(
                entry=candidate.entry,
                target=target,
                key_elements=result.key_elements,
                model=handle.retriever.model,
                base_url=handle.retriever.base_url,
                api_key=handle.retriever.api_key,
                timeout=handle.retriever.timeout,
            )
            library_strategies.append(strategy)
            library_meta.append(
                {
                    "entry_id": candidate.entry["entry_id"],
                    "entry_name": candidate.entry["name"],
                    "score": candidate.score,
                    "reason": candidate.reason,
                }
            )
        generate_count = max(pool_size - len(library_strategies), 0)
        notes = list(result.notes)
        if result.selected and not fitting:
            notes.append("generic-only match: full generation, no library use")
        composition[subtask_index] = {
            "library_strategies": library_strategies,
            "library_meta": library_meta,
            "generate_count": generate_count,
            "task_type": result.task_type,
            "matched_known_label": result.matched_known_label,
            "need_generation": result.need_generation,
            "notes": notes,
        }
    return composition


def merge_library_into_pools(
    *,
    pools: dict[int, dict],
    composition: dict[int, dict],
) -> dict[int, dict]:
    """Place instantiated library strategies first in each pool; generated
    strategies shift behind them."""
    for subtask_index, pool in pools.items():
        comp = composition.get(subtask_index)
        if comp is None:
            continue
        library_strategies = comp["library_strategies"]
        generated = sorted(
            (pool.get("strategies") or {}).items(),
            key=lambda pair: _strategy_number(pair[0]),
        )
        merged = {
            f"strategy_{position}": strategy
            for position, strategy in enumerate(library_strategies, start=1)
        }
        for position, (_name, strategy) in enumerate(generated, start=len(library_strategies) + 1):
            merged[f"strategy_{position}"] = strategy
        pool["strategies"] = merged
        pool["library_composition"] = {
            "library_count": len(library_strategies),
            "library_meta": comp["library_meta"],
            "generate_count": comp["generate_count"],
            "task_type": comp["task_type"],
        }
    return pools


def _strategy_number(name: Any) -> int:
    try:
        return int(str(name).rsplit("_", 1)[-1])
    except ValueError:
        return 0


def attribute_failure_mode(subtask_result: dict) -> str:
    """Coarse, aggregatable failure attribution from the subtask result."""
    strategies = subtask_result.get("strategies") or []
    complied = any(strategy.get("jailbreak_achieved") for strategy in strategies)
    if not complied:
        return "conversation-no-compliance"
    return "execution-not-completed"


def apply_library_feedback(
    *,
    handle: StrategyLibraryRuntime,
    item_id: str,
    subtask_runs: list[tuple[int, str, dict]],
    subtask_judge: dict,
) -> dict:
    """Record outcomes for library-sourced strategies and reinject successes."""
    successful = set(subtask_judge.get("successful_subtask_indices") or [])
    summary = {"recorded": [], "reinjected": [], "skipped": []}
    for subtask_index, _subtask, subtask_result in subtask_runs:
        if subtask_result.get("reused_conversation_for_attempt"):
            summary["skipped"].append({"subtask_index": subtask_index, "why": "reused_conversation"})
            continue
        strategy = subtask_result.get("strategy") or {}
        entry_id = strategy.get("library_entry_id")
        outcome = "success" if subtask_index in successful else "failure"
        if entry_id:
            failure_mode = None if outcome == "success" else attribute_failure_mode(subtask_result)
            handle.library.record_outcome(
                entry_id=str(entry_id),
                item_id=item_id,
                subtask_index=subtask_index,
                outcome=outcome,
                failure_mode=failure_mode,
            )
            summary["recorded"].append(
                {"subtask_index": subtask_index, "entry_id": entry_id, "outcome": outcome, "failure_mode": failure_mode}
            )
        elif outcome == "success" and (item_id, subtask_index) not in handle.reinjected_subtasks:
            added = reinject_success(
                handle=handle,
                item_id=item_id,
                attempt=subtask_result.get("run", {}).get("item_attempts"),
                subtask_index=subtask_index,
                subtask_result=subtask_result,
            )
            summary["reinjected"].append({"subtask_index": subtask_index, "added": added})
            handle.reinjected_subtasks.add((item_id, subtask_index))
    return summary


def reinject_success(
    *,
    handle: StrategyLibraryRuntime,
    item_id: str,
    attempt: Any,
    subtask_index: int,
    subtask_result: dict,
) -> dict | None:
    """Filter + extract one successful conversation and add it to the library."""
    record = {
        "item_id": item_id,
        "attempt": attempt,
        "subtask_index": subtask_index,
        "target": (subtask_result.get("subtask") or {}).get("target"),
        "strategy": subtask_result.get("strategy") or {},
        "conversation": [
            {
                "turn": turn.get("turn"),
                "attacker": turn.get("attacker"),
                "target_truncated": turn.get("target_truncated"),
            }
            for turn in (subtask_result.get("strategies") or [{}])[0].get("conversation") or []
        ],
    }
    verdict = filter_record(
        record,
        model=handle.retriever.model,
        base_url=handle.retriever.base_url,
        api_key=handle.retriever.api_key,
        timeout=handle.retriever.timeout,
    )
    if not verdict["kept"]:
        return {"added": False, "why": "filtered", "reason": verdict.get("reason")}
    extracted = extract_record(
        record,
        model=handle.retriever.model,
        base_url=handle.retriever.base_url,
        api_key=handle.retriever.api_key,
        timeout=handle.retriever.timeout,
    )
    entry = entry_from_extracted(
        extracted,
        entry_id=_next_entry_id(handle.library),
    )
    if handle.library.entry(entry["name"]) is not None or any(
        existing["name"] == entry["name"] for existing in handle.library.entries
    ):
        return {"added": False, "why": "duplicate_name", "name": entry["name"]}
    handle.library.add_entry(entry)
    return {"added": True, "entry_id": entry["entry_id"], "name": entry["name"]}


def entry_from_extracted(extracted: dict, *, entry_id: str) -> dict:
    task_signature = extracted.get("task_signature") or {}
    return {
        "entry_id": entry_id,
        "name": (extracted.get("interaction_pattern") or {}).get("name") or entry_id,
        "interaction_pattern": extracted.get("interaction_pattern") or {},
        "progression_phases": extracted.get("progression_phases") or [],
        "experience_profile": {
            "fits_task_types": [task_signature["task_type"]] if task_signature.get("task_type") else [],
            "avoid_task_types": [],
            "known_failure_modes": [],
        },
        "provenance": {
            "merged_from": [(extracted.get("interaction_pattern") or {}).get("name")],
            "source": "online_run",
            "source_subtask": extracted.get("source"),
        },
        "usage": {"applied_count": 0, "success_count": 0, "used_on": []},
        "needs_revision": False,
    }


def _next_entry_id(library: StrategyLibrary) -> str:
    numbers = []
    for entry in library.entries:
        suffix = str(entry["entry_id"]).rsplit("-", 1)[-1]
        try:
            numbers.append(int(suffix))
        except ValueError:
            continue
    return f"sp-{max(numbers, default=0) + 1:03d}"
