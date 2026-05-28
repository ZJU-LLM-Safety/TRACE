#!/usr/bin/env python3
"""
Agent Package
Contains OS Agent implementations for automated operating system testing
"""

from .OSAgent import OSAgent
from .ReactOSAgent import ReactOSAgent
from .AutoGPT_OSAgent import AutoGPT_OSAgent

# MITRE_ATTCK_OSAgent is an alias for ReactOSAgent for backward compatibility
MITRE_ATTCK_OSAgent = ReactOSAgent

__all__ = ["OSAgent", "ReactOSAgent", "MITRE_ATTCK_OSAgent", "AutoGPT_OSAgent"]