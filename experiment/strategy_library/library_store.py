"""Runtime strategy-library store.

Holds library entries produced by seeding/extraction plus the runtime state
that grows during online runs: failure-lesson counters and per-task-type
success/failure ledgers.

Failure knowledge is a flat list of imperative lessons; the attribution LLM
matches a new failure onto an existing lesson (by index) or coins a new one,
so the lesson text itself is the aggregation key and no label layer or
global vocabulary exists. A lesson reaching REVISION_THRESHOLD occurrences
marks the entry for pattern revision (one revision call decides in-place
versus fork and rewrites the pattern).

Entry schema:
{
  "entry_id": "sp-001",
  "name": "kebab-case-pattern-name",
  "interaction_pattern": {...},
  "progression_phases": [...],
  "experience_profile": {
    "fits_task_types": [...], "avoid_task_types": [...],
    "known_failure_modes": [{"lesson": "imperative sentence", "count": 1}],
    "failure_task_type": {"<task_type>": 0},
    "success_task_type": {"<task_type>": 0}
  },
  "needs_revision": false
}

Task-type ledgers are symmetric: a success cancels one recorded failure for
that task_type (or counts as a success when no failure is recorded), a
failure cancels one recorded success (or counts as a failure). Net failures
reach TASK_TYPE_PROMOTION_THRESHOLD -> the task_type moves from fits into
avoid (absolute exclusion: fit matches and random fallback both skip it).
Net successes reach the same threshold -> the task_type enters fits (the
only growth path for classes earned through fallback selections; an avoided
class can never be selected, so it can never earn its way out - revision
clearing is the only avoid exit).
"""
from __future__ import annotations

import json
from pathlib import Path

REVISION_THRESHOLD = 3
TASK_TYPE_PROMOTION_THRESHOLD = 5
TOP_FAILURE_LESSONS = 3


def _normalize_failure_modes(modes) -> list[dict]:
    normalized = []
    for mode in modes or []:
        if not isinstance(mode, dict):
            continue
        lesson = str(mode.get("lesson") or mode.get("label") or "").strip()
        if lesson:
            normalized.append({"lesson": lesson, "count": int(mode.get("count") or 1)})
    return normalized


def normalize_entry(entry: dict) -> dict:
    """Convert any historical entry format into the current schema."""
    profile = dict(entry.get("experience_profile") or {})
    return {
        "entry_id": entry["entry_id"],
        "name": entry["name"],
        "interaction_pattern": entry.get("interaction_pattern") or {},
        "progression_phases": entry.get("progression_phases") or [],
        "experience_profile": {
            "fits_task_types": list(profile.get("fits_task_types") or []),
            "avoid_task_types": list(profile.get("avoid_task_types") or []),
            "proven_task_types": list(profile.get("proven_task_types") or []),
            "known_failure_modes": _normalize_failure_modes(profile.get("known_failure_modes")),
            "failure_task_type": {
                str(key): int(value)
                for key, value in (profile.get("failure_task_type") or {}).items()
            },
            "success_task_type": {
                str(key): int(value)
                for key, value in (profile.get("success_task_type") or {}).items()
            },
        },
        "needs_revision": bool(entry.get("needs_revision")),
        "probation": bool(entry.get("probation")),
    }


def new_entry_from_seed(seed_entry: dict) -> dict:
    """Convert a reconcile-stage library entry into the runtime format."""
    profile = dict(seed_entry.get("experience_profile_seed") or {})
    return normalize_entry(
        {
            "entry_id": seed_entry["entry_id"],
            "name": seed_entry["name"],
            "interaction_pattern": seed_entry.get("interaction_pattern") or {},
            "progression_phases": seed_entry.get("progression_phases") or [],
            "experience_profile": {
                "fits_task_types": list(profile.get("fits_task_types") or []),
                "avoid_task_types": list(profile.get("avoid_task_types") or []),
                "known_failure_modes": [],
            },
            "needs_revision": False,
        }
    )


class StrategyLibrary:
    """Persistent, mutable strategy library backed by one JSON file."""

    def __init__(
        self,
        path: Path,
        *,
        revision_threshold: int = REVISION_THRESHOLD,
        task_type_threshold: int = TASK_TYPE_PROMOTION_THRESHOLD,
        top_failure_lessons: int = TOP_FAILURE_LESSONS,
    ):
        self.path = Path(path)
        self.revision_threshold = revision_threshold
        self.task_type_threshold = task_type_threshold
        self.top_failure_lessons = top_failure_lessons
        self.data: dict = {"entries": []}
        if self.path.exists():
            self.reload()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def reload(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["entries"] = [normalize_entry(entry) for entry in raw.get("entries") or []]
        raw.pop("failure_mode_vocabulary", None)
        raw.pop("version", None)
        self.data = raw

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_seed(cls, seed_path: Path, library_path: Path) -> "StrategyLibrary":
        seed = json.loads(Path(seed_path).read_text(encoding="utf-8"))
        library = cls(library_path)
        library.data = {
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

    def fits_vocabulary(self) -> list[str]:
        labels: list[str] = []
        for entry in self.entries:
            labels.extend(entry["experience_profile"]["fits_task_types"])
            labels.extend(entry["experience_profile"]["avoid_task_types"])
        return sorted(set(labels))

    def failure_lessons(self, entry: dict, limit: int | None = None) -> list[dict]:
        """Highest-count failure lessons of one entry, for instantiation."""
        if limit is None:
            limit = self.top_failure_lessons
        lessons = sorted(
            entry["experience_profile"].get("known_failure_modes") or [],
            key=lambda mode: int(mode.get("count") or 0),
            reverse=True,
        )
        return lessons[:limit]

    # -- runtime updates -------------------------------------------------

    def record_outcome(
        self,
        *,
        entry_id: str,
        outcome: str,
        failure_mode: dict | None = None,
        task_type: str | None = None,
    ) -> dict:
        """Record one application outcome.

        Failures count the attributed lesson (the failure_mode dict carries
        the resolved lesson text); a lesson reaching REVISION_THRESHOLD
        flips needs_revision. Task-type ledgers update symmetrically as
        described in the module docstring.
        """
        entry = self.entry(entry_id)
        if entry is None:
            raise KeyError(f"unknown library entry {entry_id}")
        # First application consumes the probation slot granted by a revision.
        entry["probation"] = False
        if outcome == "failure" and failure_mode and str(failure_mode.get("lesson") or "").strip():
            lesson = str(failure_mode["lesson"]).strip()
            modes = entry["experience_profile"].setdefault("known_failure_modes", [])
            existing = next((mode for mode in modes if mode.get("lesson") == lesson), None)
            if existing is None:
                existing = {"lesson": lesson, "count": 0}
                modes.append(existing)
            existing["count"] += 1
            if existing["count"] >= self.revision_threshold:
                entry["needs_revision"] = True
        if task_type:
            profile = entry["experience_profile"]
            failure_counts = profile.setdefault("failure_task_type", {})
            success_counts = profile.setdefault("success_task_type", {})
            if outcome == "success":
                if task_type in failure_counts:
                    failure_counts[task_type] -= 1
                    if failure_counts[task_type] <= 0:
                        del failure_counts[task_type]
                else:
                    success_counts[task_type] = int(success_counts.get(task_type) or 0) + 1
                    proven = profile.setdefault("proven_task_types", [])
                    if task_type not in proven:
                        proven.append(task_type)
                    if success_counts[task_type] >= self.task_type_threshold:
                        fits = profile.get("fits_task_types") or []
                        if task_type not in fits:
                            profile["fits_task_types"] = fits + [task_type]
                        # The earned promotion consumes the success ledger.
                        del success_counts[task_type]
            else:
                if task_type in success_counts:
                    success_counts[task_type] -= 1
                    if success_counts[task_type] <= 0:
                        del success_counts[task_type]
                else:
                    failure_counts[task_type] = int(failure_counts.get(task_type) or 0) + 1
                    if failure_counts[task_type] >= self.task_type_threshold:
                        fits = profile.get("fits_task_types") or []
                        profile["proven_task_types"] = [
                            label for label in (profile.get("proven_task_types") or [])
                            if label != task_type
                        ]
                        if task_type in fits:
                            # Soft demotion: leave fits and reset the ledger;
                            # the entry stays eligible via random fallback.
                            profile["fits_task_types"] = [
                                label for label in fits if label != task_type
                            ]
                        elif task_type not in (profile.get("avoid_task_types") or []):
                            # Hard demotion: already outside fits (random
                            # fallback kept failing) -> absolute exclusion.
                            profile.setdefault("avoid_task_types", []).append(task_type)
                        del failure_counts[task_type]
        self.save()
        return entry

    def apply_revision(
        self,
        *,
        entry_id: str,
        name: str,
        interaction_pattern: dict,
        progression_phases: list,
        triggering_lesson: str | None = None,
        fork: bool = False,
        next_entry_id: str | None = None,
    ) -> dict:
        """Apply a revision, in place or as a fork.

        The triggering lesson record is removed from the entry that carries
        it (the revised pattern bakes the lesson in, so the record is
        addressed). An in-place revision also resets class affinity
        entirely: fits and avoid are cleared and the task-type ledgers
        restart empty, so the revised pattern re-earns its classes through
        future outcomes; a fork leaves the source entry's class affinity
        untouched while its new entry starts clean.
        """

        def _drop_triggering(entry: dict) -> None:
            if triggering_lesson:
                modes = entry["experience_profile"].get("known_failure_modes") or []
                entry["experience_profile"]["known_failure_modes"] = [
                    mode for mode in modes if mode.get("lesson") != triggering_lesson
                ]

        if fork:
            if not next_entry_id:
                raise ValueError("fork requires next_entry_id")
            entry = {
                "entry_id": next_entry_id,
                "name": name,
                "interaction_pattern": interaction_pattern,
                "progression_phases": progression_phases,
                "experience_profile": {
                    "fits_task_types": [],
                    "avoid_task_types": [],
                    "proven_task_types": [],
                    "known_failure_modes": [],
                    "failure_task_type": {},
                    "success_task_type": {},
                },
                "needs_revision": False,
                "probation": True,
            }
            self.data["entries"].append(entry)
            source_entry = self.entry(entry_id)
            if source_entry is not None:
                _drop_triggering(source_entry)
                source_entry["needs_revision"] = False
            self.save()
            return entry
        entry = self.entry(entry_id)
        if entry is None:
            raise KeyError(f"unknown library entry {entry_id}")
        entry["name"] = name or entry["name"]
        entry["interaction_pattern"] = interaction_pattern
        entry["progression_phases"] = progression_phases
        _drop_triggering(entry)
        profile = entry["experience_profile"]
        profile["fits_task_types"] = []
        profile["avoid_task_types"] = []
        profile["failure_task_type"] = {}
        profile["success_task_type"] = {}
        entry["needs_revision"] = False
        entry["probation"] = True
        self.save()
        return entry

    def rename_entry(self, entry_id: str, name: str) -> dict:
        entry = self.entry(entry_id)
        if entry is None:
            raise KeyError(f"unknown library entry {entry_id}")
        entry["name"] = name
        self.save()
        return entry

    def entry_names(self) -> list[str]:
        return [entry["name"] for entry in self.entries]

    def next_entry_id(self) -> str:
        numbers = []
        for entry in self.entries:
            suffix = str(entry["entry_id"]).rsplit("-", 1)[-1]
            try:
                numbers.append(int(suffix))
            except ValueError:
                continue
        return f"sp-{max(numbers, default=0) + 1:03d}"

    def add_entry(self, entry: dict) -> dict:
        entry = normalize_entry(entry)
        if self.entry(entry["entry_id"]) is not None:
            raise ValueError(f"entry {entry['entry_id']} already exists")
        self.data["entries"].append(entry)
        self.save()
        return entry
