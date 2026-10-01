# TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking

Official implementation for **[TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking](https://arxiv.org/abs/2605.30883)**.

## 📖 Overview

Large language model (LLM) agents can autonomously plan, invoke tools, interact with environments, and execute long-horizon workflows. This introduces safety risks beyond conventional text-only jailbreaks: even when harmful instructions can be elicited, aligned agents may still refuse to execute them, while intermediate execution failures can interrupt an otherwise successful attack workflow.

**TRACE** is an agentic jailbreaking framework designed to address these challenges through task decomposition, execution-oriented multi-turn interaction, and adaptive recovery.

TRACE consists of three key components:

- **Scheme-based Task Decomposition.** TRACE develops around 20 procedural decomposition schemes that specify key operations and their dependencies for different task types. A complex harmful task is instantiated under different schemes to generate multiple candidate subtask sequences, which are evaluated based on harmfulness and execution difficulty.
- **Execution-oriented Multi-turn Interaction.** For subtasks that trigger execution refusals, TRACE constructs structured subtask profiles and retrieves corresponding execution-oriented strategies from an evolving strategy library. Each strategy specifies intermediate goals, relevant environment states, and required tool calls, guiding the agent toward completing the subtask through multi-turn interaction.
- **Adaptive Recovery and Strategy Evolution.** When an intermediate step fails, TRACE localizes the failed subtask, preserves previously completed execution states, adapts the corresponding strategy, and resumes execution without restarting the entire workflow. Execution feedback is further used to continuously evolve the strategy library.


## 📊 Main Results

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


## 🎬 Demos

Two recorded demonstrations with **Claude Code** and **Codex** are provided in `assets/`:

| Target Agent | Demo |
| --- | --- |
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

## 📁 Repository Structure

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

## 📝 Citation

If you find TRACE useful in your research, please consider citing:

```bibtex
@misc{zeng2026tracetaskawareadaptiveselfevolving,
      title={TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking}, 
      author={Churui Zeng and Kedong Xiu and Weiwei Qi and Yuqi Hao and Chaochao Lu and Liang He and Zhan Qin and Tianhang Zheng},
      year={2026},
      eprint={2605.30883},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2605.30883}, 
}
```
