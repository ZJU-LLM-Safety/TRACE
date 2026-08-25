"""Shared live-verification helpers for agent execution integrations."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


TRACE_ROOT = Path(__file__).resolve().parents[1]
TRACE_TOOLS = TRACE_ROOT / "tools"
if str(TRACE_TOOLS) not in sys.path:
    sys.path.insert(0, str(TRACE_TOOLS))

from nyuctf_bridge_runner import verify_gold  # noqa: E402
from vrap_bridge_runner import verify_item  # noqa: E402


def verify_history_for_lab(
    item: dict[str, Any],
    history: Any,
    *,
    lab_backend: str = "vrap",
    attacker_identity: str = "root",
    timeout: int = 180,
) -> dict[str, Any]:
    """Dispatch a prepared verification history to the active lab verifier."""

    if lab_backend == "nyuctf":
        verification = verify_gold(item, history)
        verification["status"] = "completed"
        return verification
    return verify_item(item, str(history), attacker_identity, timeout=timeout)


def verify_target_state(item: dict[str, Any], target: Any, args: Any) -> dict[str, Any]:
    """Verify a target wrapper's observed outputs/actions for the active lab."""

    history = json.dumps(
        {
            "input_output": getattr(target, "input_output", []),
            "actions": getattr(target, "actions", []),
        },
        ensure_ascii=False,
    )
    return verify_history_for_lab(
        item,
        history,
        lab_backend=getattr(args, "lab_backend", "vrap"),
        attacker_identity=getattr(target, "attacker_identity", "root"),
        timeout=int(getattr(args, "verify_timeout", 180)),
    )
