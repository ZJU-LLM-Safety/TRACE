# TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking

<div align="center">

**Task-Aware Adaptive Self-Evolving Agentic Jailbreaking for LLM Agents**

[Paper](https://arxiv.org/abs/2605.30883) | [Code](https://github.com/ZJU-LLM-Safety/TRACE)

</div>

## 🔥 News

- **[Sep. 2026]** We released an updated version of TRACE with scheme-based task decomposition, execution-oriented multi-turn interaction, and adaptive recovery.
- **[May 2026]** TRACE was released on arXiv.

## 📖 Overview

Large language model (LLM) agents can autonomously plan, invoke tools, interact with environments, and execute long-horizon workflows. This introduces safety risks beyond conventional text-only jailbreaks: even when harmful instructions can be elicited, aligned agents may still refuse to execute them, while intermediate execution failures can interrupt an otherwise successful attack workflow.

**TRACE** is an agentic jailbreaking framework designed to address these challenges through task decomposition, execution-oriented multi-turn interaction, and adaptive recovery.

TRACE consists of three key components:

- **Scheme-based Task Decomposition.** TRACE develops around 20 procedural decomposition schemes that specify key operations and their dependencies for different task types. A complex harmful task is instantiated under different schemes to generate multiple candidate subtask sequences, which are evaluated based on harmfulness and execution difficulty.

- **Execution-oriented Multi-turn Interaction.** For subtasks that trigger execution refusals, TRACE constructs structured subtask profiles and retrieves corresponding execution-oriented strategies from an evolving strategy library. Each strategy specifies intermediate goals, relevant environment states, and required tool calls, guiding the agent toward completing the subtask through multi-turn interaction.

- **Adaptive Recovery and Strategy Evolution.** When an intermediate step fails, TRACE localizes the failed subtask, preserves previously completed execution states, adapts the corresponding strategy, and resumes execution without restarting the entire workflow. Execution feedback is further used to continuously evolve the strategy library.

## 📊 Main Results

We evaluate TRACE across different agent frameworks and underlying LLMs on **AdvCUA**, a benchmark containing realistic OS-level attack tasks that require end-to-end execution in interactive environments.

TRACE consistently achieves strong attack performance across different agent configurations and substantially outperforms existing jailbreak baselines.

| Agent Framework | Model | TRACE ASR |
| :--- | :--- | :---: |
| Codex | GPT-5.5 | **0.62** |
| Claude Code | Claude Opus 4.8 | **0.54** |
| Claude Code | Gemini-3.5-Flash | **0.69** |

For example, on **Codex + GPT-5.5**, TRACE achieves an ASR of **0.62**, compared with **0.27** for the strongest baseline, MD-Attack.

These results demonstrate the importance of preserving execution dependencies during task decomposition and adapting interaction strategies according to intermediate environment states and execution feedback.

For complete results across benchmarks, models, baselines, efficiency analysis, ablations, and defenses, please refer to our [paper](https://arxiv.org/abs/2605.30883).

## 🎬 Demos

We provide two recorded demonstrations of TRACE with **Claude Code** and **Codex**.

| Target Agent | Demo |
| :--- | :--- |
| Claude Code | [Claude Code Demo](assets/claude_demo.mp4) |
| Codex | [Codex Demo](assets/codex_demo.mp4) |

## 🛠️ Installation

The provided environment specification targets Linux and Python 3.11.

```bash
git clone https://github.com/ZJU-LLM-Safety/TRACE.git
cd TRACE

conda env create -f environment.yml
conda activate trace
```

## 🚀 Usage

### 1. Configure API Access

Set the credentials and base URL for an OpenAI-compatible Chat Completions endpoint:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="<your-api-base-url>"
```

### 2. Task Decomposition

Generate candidate subtask sequences using the procedural decomposition schemes:

```bash
python improved_decomposer/improved_task_decomposer.py \
  --data_file_path /path/to/tasks.json \
  --output_file_path outputs/decompositions.jsonl \
  --model "<decomposition-model>" \
  --num_decompositions_per_task 5 \
  --use_schema_decompose
```

`--use_schema_decompose` enables scheme-based candidate generation.

### 3. Run TRACE

The following example runs TRACE on **AdvCUA** with **Codex**:

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

To inspect all available command-line options:

```bash
python improved_decomposer/improved_task_decomposer.py --help
python experiment/main.py --help
```

## 📁 Repository Structure

| Path | Description |
| :--- | :--- |
| `improved_decomposer/` | Procedural decomposition schemes, candidate generation, validation, and selection |
| `experiment/main.py` | Main experiment orchestration and execution-feedback loop |
| `experiment/runtime/` | Runtime components for models, strategies, and subtask sequences |
| `experiment/strategy_library/` | Strategy storage, retrieval, revision, and feedback integration |
| `experiment/config/` | Experiment configurations and prompt templates |
| `agent_execution/` | Target-agent interfaces, session handling, trajectory collection, and verification |
| `tools/` | Benchmark bridges, API adapters, and diagnostics |
| `assets/` | Demonstration videos |
| `environment.yml` | Conda environment specification |

## 📝 Citation

If you find TRACE useful for your research, please cite our paper:

```bibtex
@misc{zeng2026trace,
  title        = {TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking},
  author       = {Churui Zeng and Weiwei Qi and Kedong Xiu and Tianhang Zheng and Chaochao Lu and Liang He and Zhan Qin and Kui Ren},
  year         = {2026},
  eprint       = {2605.30883},
  archivePrefix= {arXiv},
  primaryClass = {cs.CR}
}
```

## ⚠️ Disclaimer

This repository is intended solely for **academic research and controlled AI safety evaluation**. Experiments should only be conducted in authorized and isolated environments. Do not use this framework against real-world systems, third-party services, or infrastructure without explicit authorization.
