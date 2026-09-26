# TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking

## Overview

- **Scheme-based task decomposition.** To reduce attack difficulty, we develop around 20 procedural decomposition schemes to resolve a complex harmful task into relatively benign and simple subtask sequences. We measure these candidate sequences through two dimensions, i.e., harmfulness and difficulty, and select the optimal one for instantiating the attack.

- **Execution-oriented multi-turn interaction.** For the subtasks that trigger execution refusals, TRACE constructs subtask profiles and retrieves appropriate execution-oriented multi-turn interaction strategies from a continuously evolving strategy library. TRACE leverages the selected strategy to instantiate the subtask and gradually constructs a plausible execution context through multi-turn interactions, inducing the target agent to complete the subtask.

- **Adaptive recovery and strategy evolution.** When an intermediate step fails, TRACE localizes the failed subtask, preserves prior progress, adapts the corresponding strategy, and resumes from the failed subtask without starting over. Furthermore, TRACE leverages execution feedback to continuously evolve the execution-oriented strategy library.

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

## Usage

Run the following commands from the repository root. Replace placeholder paths, model names, and endpoint settings with your local configuration.

### Task Decomposition

Set the credentials and base URL for an OpenAI-compatible Chat Completions endpoint:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="<your-api-base-url>"
```

The input must be a JSON array of task records. Each record should have a unique `id` and a task-text field named `prompt_to_os_agent`, `harmful`, or `prompt`. If the source dataset uses `query`, copy it to `prompt` before running the decomposer.

```bash
python improved_decomposer/improved_task_decomposer.py \
  --data_file_path /path/to/tasks.json \
  --output_file_path outputs/decompositions.jsonl \
  --model "<decomposition-model>" \
  --num_decompositions_per_task 5 \
  --use_schema_decompose
```

`--use_schema_decompose` enables scheme-based candidate generation. The JSONL output contains task IDs, selected `sub_prompts`, candidate evaluation records, and decomposition costs. Records with no valid candidate have `status: "no_valid_candidate"` and empty `sub_prompts`. Existing task IDs in the output file are skipped when the command is rerun. Use `--target_ids ID1 ID2` to process selected tasks.

### Main Experiment

Prepare a JSON or JSONL dataset by matching decomposition results to the original benchmark records by ID and adding the selected `sub_prompts`. Preserve the benchmark-specific environment and verification fields; the decomposer output alone does not retain all of these fields. Exclude records with no valid candidate when evaluating decomposed workflows.

Execution requires the benchmark's Docker environment and resources, the target-agent CLI inside that environment, a local Hugging Face-compatible attacker model with sufficient hardware, and access to the configured API models. Review `experiment/config/config.yaml` for the plan-generation and judge model settings. The target Responses endpoint must be reachable from the benchmark containers.

The following example uses the AdvCUA/VRAP backend with Codex. The strategy library is copied to the output directory because the experiment updates it during execution.

```bash
mkdir -p outputs/experiment
cp experiment/strategy_library/strategy_library.json outputs/experiment/strategy_library.json

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

If the target endpoint uses a different credential from the helper models, store it in a separate environment variable and pass its name through `--responses-env-key`. Use `--ids ID1,ID2` to select tasks. The main experiment processes all provided subtasks by default and writes results, metadata, and intermediate records beside the output file.

To inspect the available command-line options:

```bash
python improved_decomposer/improved_task_decomposer.py --help
python experiment/main.py --help
```
