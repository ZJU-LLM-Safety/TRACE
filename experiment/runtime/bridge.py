"""Compatibility imports for reusable trace agent-execution bridge code."""
from __future__ import annotations

import sys
from pathlib import Path


TRACE_ROOT = Path(__file__).resolve().parents[2]
if str(TRACE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACE_ROOT))

from agent_execution.bridge import *  # noqa: F401,F403,E402

DEFAULT_OUTPUT_ROOT = Path("experiment_outputs")
