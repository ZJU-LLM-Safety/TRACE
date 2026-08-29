"""Runtime strategy-library store.

Holds library entries produced by seeding/extraction plus the runtime state
that grows during online runs: per-entry usage history and failure-mode
counters. A failure mode that reaches REVISION_THRESHOLD occurrences marks the
entry as needing pattern revision (the revision itself is a separate step).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LIBRARY_VERSION = 1
REVISION_THRESHOLD = 3


def new_entry_from_seed(seed_entry: dict) -> dict:
    """Convert a reconcile-stage library entry into the runtime format."""
    profile = dict(seed_entry.get("experience_profile_seed") or {})
    return {
        "entry_id": seed_entry["entry_id"],
        "name": seed_entry["name"],
        "interaction_pattern": seed_entry.get("interaction_pattern") or {},
        "progression_phases": seed_entry.get("progression_phases") or [],
        "experience_profile": {
            "fits_task_types": list(profile.get("fits_task_types") or []),
            "avoid_task_types": list(profile.get("avoid_task_types") or []),
            "known_failure_modes": list(profile.get("known_failure_modes") or []),
        },
        "provenance": {
            "merged_from": list(seed_entry.get("merged_from") or []),
            "distinct_rationale": seed_entry.get("distinct_rationale"),
        },
        "usage": {
            "applied_count": 0,
            "success_count": 0,
            "used_on": [],
        },
        "needs_revision": False,
    }


class StrategyLibrary:
    """Persistent, mutable strategy library backed by one JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict = {
            "version": LIBRARY_VERSION,
            "updated_at": None,
            "entries": [],
        }
        if self.path.exists():
            self.reload()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def reload(self) -> None:
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_seed(cls, seed_path: Path, library_path: Path) -> "StrategyLibrary":
        seed = json.loads(Path(seed_path).read_text(encoding="utf-8"))
        library = cls(library_path)
        library.data = {
            "version": LIBRARY_VERSION,
            "updated_at": None,
            "seed_source": str(seed_path),
            "entries": [
                new_entry_from_seed(entry) for entry in seed.get("library_entries") or []
            ],
        }
        library.save()
        return library

    # -- queries -------------------------------------------------------

    @property
    def entries(self) -> list[dict]:
        return self.data.get("entries") or []

    def entry(self, entry_id: str) -> dict | None:
        return next(
            (entry for entry in self.entries if entry["entry_id"] == entry_id),
            None,
        )

    def used_on_subtask(self, item_id: str, subtask_index: int) -> set[str]:
        """Entry ids already applied to this exact subtask."""
        used = set()
        for entry in self.entries:
            for record in entry["usage"]["used_on"]:
                if (
                    record.get("item_id") == item_id
                    and record.get("subtask_index") == subtask_index
                ):
                    used.add(entry["entry_id"])
        return used

    def fits_vocabulary(self) -> list[str]:
        labels: list[str] = []
        for entry in self.entries:
            labels.extend(entry["experience_profile"]["fits_task_types"])
            labels.extend(entry["experience_profile"]["avoid_task_types"])
        return sorted(set(labels))

    # -- runtime updates -------------------------------------------------

    def record_outcome(
        self,
        *,
        entry_id: str,
        item_id: str,
        subtask_index: int,
        outcome: str,
        failure_mode: str | None = None,
    ) -> dict:
        """Record one application of an entry and update failure counters.

        outcome is "success" or "failure". For failures, failure_mode is a
        short normalized description; reaching REVISION_THRESHOLD occurrences
        of the same mode flips needs_revision on the entry.
        """
        entry = self.entry(entry_id)
        if entry is None:
            raise KeyError(f"unknown library entry {entry_id}")
        usage = entry["usage"]
        usage["applied_count"] += 1
        if outcome == "success":
            usage["success_count"] += 1
        usage["used_on"].append(
            {
                "item_id": item_id,
                "subtask_index": int(subtask_index),
                "outcome": outcome,
                "failure_mode": failure_mode,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if outcome == "failure" and failure_mode:
            modes = entry["experience_profile"]["known_failure_modes"]
            existing = next(
                (mode for mode in modes if mode["description"] == failure_mode),
                None,
            )
            if existing is None:
                existing = {"description": failure_mode, "count": 0, "last_seen": None}
                modes.append(existing)
            existing["count"] += 1
            existing["last_seen"] = datetime.now(timezone.utc).isoformat()
            if existing["count"] >= REVISION_THRESHOLD:
                entry["needs_revision"] = True
                existing["revision_due"] = True
        self.save()
        return entry

    def add_entry(self, entry: dict) -> dict:
        if self.entry(entry["entry_id"]) is not None:
            raise ValueError(f"entry {entry['entry_id']} already exists")
        self.data["entries"].append(entry)
        self.save()
        return entry
