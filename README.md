# TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking

Official implementation for **[TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking](https://arxiv.org/abs/2605.30883)**.

## Overview

**TRACE** is an agentic jailbreaking framework that reduces attack difficulty, maintains execution dependencies, induces agents to execute end-to-end attack workflows, and recovers from intermediate failures without restarting from scratch.

TRACE consists of three main stages:

- **Scheme-guided task decomposition.** TRACE develops over 20 procedural decomposition schemes that specify key operations and their execution dependencies for different task types. These schemes instantiate complex harmful tasks into multiple more executable and benign-looking subtask sequences. TRACE evaluates valid candidates based on subtask harmfulness and execution difficulty and selects an appropriate sequence for attack execution.

- **Execution-oriented multi-turn interaction.** For subtasks that trigger execution refusals, TRACE constructs structured subtask profiles and retrieves profile-matched execution-oriented strategies. Each strategy specifies an interaction pattern and progression phases associated with intermediate goals, environment states, required tools, and success signals, progressively guiding the target agent to complete the refused subtask.

- **Self-evolving strategy library.** TRACE uses execution feedback to continuously evolve its strategy library. It retains strategies from successful trajectories, learns from recurring failure patterns to revise existing strategies or construct new ones, and prioritizes historically effective strategies for similar subtask profiles. When an intermediate step fails, TRACE localizes the failed subtask, preserves prior progress, adapts the corresponding strategy, and resumes execution from the failed state.

The evolved strategy library contains more than 30 reusable execution-oriented strategies.

For more details, please refer to our **[paper](https://arxiv.org/abs/2605.30883)**.

## Main Results

We evaluate TRACE on **AdvCUA** and **AgentHazard**, two agent safety benchmarks that require end-to-end execution of multi-step tasks in interactive environments. Experiments cover two agent frameworks, **Codex** and **Claude Code**, with multiple underlying models.

| Agent Framework | Underlying Model | AdvCUA | AgentHazard |
| --- | --- | ---: | ---: |
| Codex | GPT-5.5 | **0.62** | **0.97** |
| Codex | Gemini-3.5-Flash | **0.65** | **0.99** |
| Codex | DeepSeek-V4-Pro | **0.81** | **1.00** |
| Claude Code | Claude-Opus-4.8 | **0.54** | **0.97** |
| Claude Code | Gemini-3.5-Flash | **0.69** | **0.98** |
| Claude Code | DeepSeek-V4-Pro | **0.88** | **0.99** |

TRACE consistently achieves strong attack performance across different agent frameworks and underlying models, substantially outperforming prior baselines in most settings. On **AdvCUA**, TRACE achieves ASRs ranging from **0.54 to 0.88**. On **AgentHazard**, TRACE achieves **near-perfect ASRs across all evaluated configurations**, ranging from **0.97 to 1.00**.

Please refer to the [paper](https://arxiv.org/abs/2605.30883) for complete baseline comparisons, category-level results, efficiency analysis, ablation studies, and defense experiments.

## Demos

Two recorded demonstrations with **Claude Code** and **Codex** are provided in `assets/`:

| Target Agent | Demo |
| --- | --- |
| Claude Code | [Claude Code Demo](assets/claude_demo.mp4) |
| Codex | [Codex Demo](assets/codex_demo.mp4) |

## Installation

The provided environment specification targets Linux and Python 3.11.

```bash
git clone https://github.com/ZJU-LLM-Safety/TRACE.git
cd TRACE

conda env create -f environment.yml
conda activate trace
```

## Usage

### 1. Configure API Access

Set the credentials and base URL for an OpenAI-compatible Chat Completions endpoint:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="<your-api-base-url>"
```

### 2. Task Decomposition

Generate candidate subtask sequences using scheme-guided task decomposition:

```bash
python improved_decomposer/improved_task_decomposer.py \
  --data_file_path /path/to/tasks.json \
  --output_file_path outputs/decompositions.jsonl \
  --model "<decomposition-model>" \
  --num_decompositions_per_task 5 \
  --use_schema_decompose
```

`--use_schema_decompose` enables scheme-based candidate generation.

### 3. Run Experiments

The following example uses the AdvCUA benchmark with Codex:

```bash
python experiment/main.py \
  --dataset /path/to/benchmark_with_subtasks.jsonl \
  --config experiment/config/config.yaml \
  --lab-backend vrap \
  --compose-file /path/to/benchmark/docker-compose.yml \
  --attacker-model-path /path/to/attacker-model \
  --target-backend codex \
  --codex-model "<target-model>" \
  --responses-base-url "<target-responses-api-base-url>" \
  --responses-env-key OPENAI_API_KEY \
  --strategy-library outputs/experiment/strategy_library.json \
  --output outputs/experiment/results.jsonl
```

To inspect the available command-line options:

```bash
python improved_decomposer/improved_task_decomposer.py --help
python experiment/main.py --help
```

## Repository Structure

| Path | Description |
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

## Citation

If you find TRACE useful in your research, please consider citing:

```bibtex
@article{zeng2026trace,
  title={TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking},
  author={Zeng, Churui and Xiu, Kedong and Qi, Weiwei and Hao, Yuqi and Lu, Chaochao and He, Liang and Qin, Zhan and Zheng, Tianhang},
  journal={arXiv preprint arXiv:2605.30883},
  year={2026}
}
```

## Disclaimer

This work is intended solely to reveal and mitigate the safety risks of LLM agents. All experiments are conducted in controlled environments. The code and resources in this repository should only be used for research and authorized safety evaluation.
