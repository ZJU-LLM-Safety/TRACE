"""Compatibility imports for reusable target prompting helpers."""
from __future__ import annotations

import sys
from pathlib import Path


TRACE_ROOT = Path(__file__).resolve().parents[2]
if str(TRACE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACE_ROOT))

from agent_execution.target_prompting import *  # noqa: F401,F403,E402
