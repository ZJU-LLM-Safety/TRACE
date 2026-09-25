# TRACE: Trust-Redistribution Agent Conviction Evaluation

TRACE is a multi-turn jailbreak evaluation framework for agentic AI systems.
It decomposes harmful task specifications into operational subtasks, elicits
actionable guidance from the target model through covert multi-turn
conversations, executes the guidance in sandboxed environments, and judges
the resulting trajectories for harmful behavior.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Task Decomposition (improved_decomposer/)                       │
│  Schema-guided candidate generation → local-rule normalization  │
│  → LLM quality gate (scoring mode) → multi-criteria selection   │
└────────────────────────┬────────────────────────────────────────┘
                         │ sub_prompts
┌────────────────────────▼────────────────────────────────────────┐
│  Strategy Composition (experiment/)                              │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐ │
│  │ Strategy Library    │  │ Fresh Generation (GPT-4o)        │ │
│  │ trust-weighted      │  │ covert persona + phased plan     │ │
│  │ proven/probation/   │  │ 10 strategies per subtask        │ │
│  │ generated pool      │  └──────────────────────────────────┘ │
│  └─────────────────────┘                                        │
│  Direct Execution Probe: bare subtask → retain or fall back     │
└────────────────────────┬────────────────────────────────────────┘
                         │ execution inputs
┌────────────────────────▼────────────────────────────────────────┐
│  Multi-Turn Elicitation (attacker × target)                     │
│  Qwen2.5-32B attacker · max 5 turns · covert strategy prompt   │
│  Bounded retry: 2 turns/phase + success-signal checking        │
└────────────────────────┬────────────────────────────────────────┘
                         │ SOURCE TEXT (deflection-filtered)
┌────────────────────────▼────────────────────────────────────────┐
│  Agent Execution (agent_execution/)                              │
│  Codex CLI in Docker · same-session subtask chain               │
│  per-benchmark environments (AdvCUA / NYUCTF / AgentHazard)     │
└────────────────────────┬────────────────────────────────────────┘
                         │ trajectories
┌────────────────────────▼────────────────────────────────────────┐
│  Verification & Library Feedback                                 │
│  benchmark-specific verifier → per-subtask judge                │
│  → lesson attribution → revision → re-injection                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Task Decomposition

`improved_decomposer/improved_task_decomposer.py` generates 5 candidate
subtask sequences per task, guided by benchmark-specific schemas
(`improved_task_schema.py`). Each candidate passes through local-rule
normalization (`improved_task_text.py` — with literal-preserving safety
wording) and an LLM quality gate operating in scoring mode. Selection
combines safety, granularity, and inverse difficulty scores. The
`unsupported_addition` issue type is promoted to major severity: a faithful
decomposition must not introduce steps the original task does not support.

Output rows record the full candidate process (all candidates, per-candidate
critique, selection metrics) for reproducibility.

### Strategy Library

`experiment/strategy_library/` implements a persistent, file-locked library
of interaction patterns. Each entry carries:

- an interaction pattern (persona + context + approach) with progression
  phases,
- an experience profile: fits/avoid task types with proven/probation
  status, known failure modes as imperative lessons with occurrence counts,
- task-type ledgers that promote or demote classes at threshold net
  outcomes.

Trust-weighted composition places proven entries first, probation entries
(a fresh revision earning its first trial) second, and appends unproven
library entries after the diverse generated batch — seeded knowledge never
displaces fresh generation. Failures are attributed to lessons by an LLM
(match-to-existing or coin-new); a lesson reaching the revision threshold
triggers an in-place rewrite or fork. Successful generated-strategy
conversations are extracted back into the library.

The library supports parallel runs: an exclusive file lock wraps every
read-modify-write cycle, starting from a fresh disk load.

### Online Runner

`experiment/main.py` orchestrates per-item execution:

1. **Pool composition** — classify each subtask (the classifier splits
   destructive/sensitive payload subtasks into a `-destructive` suffix
   class); compose trust-weighted pools from library + generation.
2. **Direct execution probe** — each subtask's bare input runs in a fresh
   container; refusal/safety patterns fall back to strategy conversation,
   otherwise the input is retained.
3. **Item attempt** — retained direct inputs and strategy-derived inputs
   execute in subtask order within one continuous target session.
4. **Verification & retry** — on failure, successful prefixes are retained;
   the trailing failed suffix retries with the next unused strategy. From
   attempt 2 onward, retained bare inputs are upgraded to strategy
   conversations. Per-item fault tolerance: a failed item records an error
   row without killing the run.
5. **Library feedback** — outcome recording, failure attribution, revision
   triggering, and success re-injection (Section "Strategy Library").

### Execution Backends

The runner supports three benchmark environments via `--lab-backend`:

| Backend | Environment | Verification |
|---|---|---|
| `vrap` (AdvCUA) | dual ssh_server + ssh_client containers with per-item decoy initialization | deterministic gold checks |
| `nyuctf` | per-challenge Docker containers from NYUCTF CTF archive | gold probe comparison |
| `agenthazard` | generic container with query-generated decoy data + before/after state probes | official AgentHazard LLM judge (action-only) |

For AgentHazard, decoy environments are generated from the query alone
(category + prompt, no subtask information), validated by live execution in
throwaway containers, and attached at the source-data level so any
decomposition variant inherits them. The live verification judge sees only
the executed tool calls (commands + results) — prompts and agent narration
are excluded, so verdicts rest on executed behavior.

## Results

### AgentHazard (100-item sample, B-mode)

| Configuration | Harmful rate |
|---|---|
| C-mode (blank env, lightweight decomposition) | 2/10 |
| C-mode (blank env, improved decomposition) | 4/10 |
| B-mode (decoy env, lightweight decomposition) | 6/10 |
| **B-mode (decoy env, improved decomposition + query-only decoys)** | **8/10** |

The two improvements stack: improved decomposition + query-generated decoy
environments flipped items 24 and 605 with zero losses.

## Reproduction

### Prerequisites

- Python 3.11+ with dependencies in `environment.yml`
- Docker with the target agent image
- Local Qwen2.5-32B-Instruct weights (set `QWEN_MODEL_PATH`)
- OpenAI-compatible API endpoint (set `OPENAI_API_KEY` and
  `OPENAI_BASE_URL`)
- Benchmark data (not included; place under `data/` per the expected
  layout)

### Decomposition

```bash
python3 improved_decomposer/improved_task_decomposer.py \
  --data_file_path <input.json> \
  --output_file_path <output.jsonl> \
  --num_decompositions_per_task 5 \
  --model gpt-4o-2024-11-20 \
  --use_schema_decompose \
  --disable_llm_quality_refinement \
  --max_regeneration_attempts 1
```

### Evaluation run

```bash
python experiment/main.py \
  --dataset <benchmark.jsonl> \
  --lab-backend <vrap|nyuctf|agenthazard> \
  --config experiment/config/config_online_qwen_local_gpt55_id15_covert.yaml \
  --plan-prompts experiment/config/prompts/plan_generation_prompts_covert.yaml \
  --plan-generator-provider openai \
  --plan-generator-model gpt-4o-2024-11-20 \
  --all-subtasks \
  --target-backend codex \
  --codex-model gpt-5.5 \
  --responses-base-url "$OPENAI_BASE_URL" \
  --responses-env-key OPENAI_API_KEY \
  --bypass --no-textgrad --no-plan-revision \
  --strategy-success-mode execute_verify \
  --strategy-execution-prompt-style auto \
  --attacker-model-path "$QWEN_MODEL_PATH" \
  --attacker-max-new-tokens 2048 --device-map auto \
  --item-attempts 5 \
  --strategy-library experiment/strategy_library/strategy_library.json \
  --library-pool-size 3 \
  --library-max-per-subtask 2 \
  --library-selection-order prefer-library
```

For a Claude Code target, replace the target settings with
`--target-backend claude --claude-runner cli --claude-model <model>`.

### Output format

Each result row in `results.jsonl` contains:

- `subprompts`: per-subtask records with conversation histories and
  execution trajectories,
- `execution.subtask_executions`: per-subtask codex executions including
  before/after environment probes (AgentHazard),
- `verification`: the benchmark-specific verdict,
- `run.cost`: per-item timing decomposed into conversation, execution, and
  environment-setup components (net execution time = total − env_setup).

## License

This code is provided for research evaluation purposes.
