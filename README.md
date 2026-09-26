# TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking

This repository provides the official implementation of **TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking**, submitted to ICLR 2027. TRACE is a framework for studying the safety of LLM agents through end-to-end task execution in controlled interactive environments.

## Overview

TRACE combines three components:

- **Scheme-based task decomposition.** Reusable procedural schemes generate candidate subtask sequences while preserving execution dependencies. Candidates are evaluated based on subtask harmfulness and execution difficulty.
- **Execution-oriented multi-turn interaction.** A strategy library supports interactions tailored to the operations, environment conditions, and tool requirements of individual subtasks.
- **Adaptive recovery and strategy evolution.** Execution feedback enables recovery from intermediate failures and updates the strategy library using historical trajectories.

The implementation includes task decomposition, target-agent interfaces, experiment orchestration, and strategy-library management.

## Demos

Two recorded demonstrations with Claude Code and Codex are provided in [`assets/`](assets/):

| Target agent | Demo |
| --- | --- |
| Claude Code | [Claude Code demo](assets/claude_demo.mp4) |
| Codex | [Codex demo](assets/codex_demo.mp4) |
