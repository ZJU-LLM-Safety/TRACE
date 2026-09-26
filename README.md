# TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking

## Overview

TRACE consists of three main components. In scheme-guided task decomposition, we develop around 20 procedural schemes to instantiate the harmful task as subtask sequences, preserving harmful intent and reducing execution difficulty. TRACE then evaluates the valid sequences along two dimensions and selects the optimal one.

Most of the subtasks in the selected sequence are benign-looking and can be executed by the target agent smoothly. For the remaining refused subtasks, TRACE constructs structured subtask profiles and retrieves profile-matched strategies that progressively guide the agent to complete these subtasks via multi-turn interactions.

Furthermore, TRACE leverages execution feedback to continuously evolve the execution-oriented strategy library: successful trajectories provide reusable strategies, and failure patterns guide revisions and evidence accumulation to optimize future strategy application.

## Demos

Two recorded demonstrations with Claude Code and Codex are provided in `assets/`:

| Target agent | Demo |
| --- | --- |
| Claude Code | [Claude Code demo](assets/claude_demo.mp4) |
| Codex | [Codex demo](assets/codex_demo.mp4) |

## Repository Structure

| Path | Contents |
| --- | --- |
| `improved_decomposer/` | Procedural schemes, candidate generation, validation, and selection |
| [`experiment/main.py`](experiment/main.py) | Experiment orchestration and execution feedback |
| `experiment/runtime/` | Runtime components for models, strategies, and subtask sequences |
| `experiment/strategy_library/` | Strategy storage, retrieval, revision, and feedback integration |
| `experiment/config/` | Experiment configurations and prompt templates |
| `agent_execution/` | Target-agent interfaces, session handling, trajectory collection, and verification |
| `tools/` | Benchmark bridges, API adapters, and diagnostics |
| `assets/` | Claude Code and Codex demonstration videos |
| [`environment.yml`](environment.yml) | Conda environment specification |

## Environment Setup

The provided environment specification targets Linux and Python 3.11.

```bash
conda env create -f environment.yml
conda activate trace
```

Execution experiments additionally require benchmark data and environment resources, Docker, the relevant target-agent CLI, and access to the models used by the selected configuration. Local model configurations also require the corresponding model weights and suitable hardware. API credentials should be supplied through environment variables.

