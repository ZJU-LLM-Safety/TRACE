# Agent Bridge Trace Workspace

Current active workspace for the AdvCUA/NYUCTF agent red-team experiments:

- `improved_decomposer/`: task decomposition into `sub_prompts`.
- `experiment/`: online strategy runner (multi-turn elicitation, execution,
  strategy library) — the main entry point.

The legacy x-teaming pipeline (`run_xteaming/`), the bridge/tool experiments
(`tools/`, MoonBridge, model-through-Codex notes), and their commands are
archived in [README_bak.md](README_bak.md).

## Task Decomposition

`improved_decomposer/improved_task_decomposer.py` is the task decomposition entry
point. The current working flow is:

- generate 5 candidate subtask sequences per item,
- enable schema-guided decomposition when available,
- keep the LLM quality gate in scoring mode only,
- allow at most 1 regeneration attempt,
- keep local wording normalization on by default.

Typical command:

```bash
cd "$TRACE_ROOT"
export OPENAI_API_KEY='<openai-compatible-api-key>'
export OPENAI_BASE_URL='https://az.gptplus5.com/v1'

python3 improved_decomposer/improved_task_decomposer.py \
  --data_file_path data/advcua/end2end.json \
  --output_file_path improved_decomposer/results/advcua_end2end_schema_local_rules_sample4_20260901_v5.jsonl \
  --target_ids 1 15 21 26 \
  --num_decompositions_per_task 5 \
  --model gpt-4o-2024-11-20 \
  --use_schema_decompose \
  --disable_llm_quality_refinement \
  --max_regeneration_attempts 1
```

Useful flags:

- `--use_schema_decompose`: use hand-authored schema prompts for dataset-specific
  decomposition structure.
- `--schema_names advcua_root_login_password_harvest`: force one or more named
  schemas and skip keyword matching; this implies `--use_schema_decompose`.
- `--disable_llm_quality_refinement`: skip refine/rewrite after LLM critique.
- `--max_regeneration_attempts 1`: allow at most one fresh regeneration after a
  failed candidate.
- `--disable_local_rules`: inspect the raw LLM decomposition without local
  normalization.
- `--selection_safety_weight 0.5`: default balance between safety and granularity.

Each output row now includes:

- `sub_prompts`: the final selected candidate,
- `candidate_process`: all generated candidates plus `valid_result` and
  `selection_metrics`,
- `decomposition_time`: model-time only, excluding file I/O,
- `decomposition_token_cost`: summed token usage for the whole decomposition.


The current smoke-tested decomposition set covers:

- AdvCUA end2end: `1 15 21 26`
- AdvCUA ttp: `5`
- NYUCTF network: `vectors_machine lost_mind flatcrypt bin_t my_first_pwnie`

The successful result files are:

- `improved_decomposer/results/advcua_end2end_schema_local_rules_sample4_20260901_v5.jsonl`
- `improved_decomposer/results/advcua_ttp_schema_local_rules_sample1_20260901_v3.jsonl`
- `improved_decomposer/results/nyuctf_network_schema_local_rules_sample5_20260901_v4.jsonl`

These runs were executed with schema decomposition enabled, no quality refinement,
one regeneration attempt, and the new candidate trace format described above.


## Experiment Online Runner

`experiment/main.py` is the current online entry point for AdvCUA/VRAP and
NYUCTF. It accepts JSON/JSONL items with `sub_prompts`; it does not consume a
pre-generated plans file and does not use `--link-mode`, `--limit`,
`--target-execution-mode`, or `--post-subtask-execution-prompt`.

For each selected item, the current flow is:

1. Compose the per-subtask strategy pools, trust-weighted: classify each
   subtask once (the classifier appends a `-destructive` suffix for
   destructive/sensitive payload subtasks, splitting them from benign
   classes). The early pool holds up to `--library-max-per-subtask` entries
   that are proven on this task type (recorded success) plus at most one
   probation entry (a fresh revision or fork earning its first trial);
   unproven library entries never displace fresh generation - a shuffled
   generated batch of `--online-strategies-per-subtask` fills the pool
   upfront, and unproven fit picks plus random fills are appended after it.
   A subtask with no proven and no probation entries generates the full
   pool upfront (the library steps aside for that class).
2. Run a direct execution probe for each subtask in its own fresh
   `ContinuousItemExecution` container/session. The probe contains only that
   subtask's execution input. A direct response containing a refusal/safety
   pattern falls back to a multi-turn strategy conversation; otherwise its
   execution input is retained.
3. Run the formal item attempt in one new continuous execution container:
   retained direct inputs and strategy-derived inputs are executed in subtask
   order in the same target session.
4. Run verification and the per-subtask judge. On failure, successful
   subtask conversations and execution inputs are retained; the first
   subtask in the trailing failed suffix gets the next unused strategy on
   the next item attempt (`--item-attempts`, default 5; the direct-probe
   phase is not counted). From the second attempt on, a retained subtask
   that still runs its bare direct-probe input is upgraded to a strategy
   conversation, so re-executions never re-roll the bare input.
5. After each attempt, the strategy library is updated: successes and
   failures are recorded per entry, failures are attributed to an imperative
   lesson (matched to an existing lesson or newly coined), a lesson reaching
   `--revision-threshold` occurrences triggers a pattern revision (in-place
   rewrite or fork), task-type ledgers promote or demote classes at
   `--task-type-threshold` net outcomes, and successful generated-strategy
   conversations are extracted back into the library - structurally similar
   patterns merge into the matching entry instead of creating duplicates.

The direct fallback test currently detects explicit refusal/safety language.
An operational failure without such language, for example a failed SSH login
or a missing file in an isolated direct-probe container, is retained as a
direct input and is only caught by the later execution/verification/judge
stage.

### AdvCUA/Codex Example (strategy library + GPT-4o plan generator)

This is the current canonical command: every row of the dataset (omit
`--ids` to run all items), local Qwen2.5-32B as the conversation attacker,
GPT-4o-2024-11-20 as the plan generator and library model, Codex GPT-5.5 as
the target, and the runtime strategy library enabled. Multi-turn attacker
conversations are capped at `attacker.max_turns: 5` in the covert config.

Launch long runs inside tmux so they survive disconnects and can be
inspected with `tmux attach`:

```bash
cd "$TRACE_ROOT"
tmux new-session -d -s advcua_run
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    CUDA_VISIBLE_DEVICES=2,3 \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    OPENAI_TIMEOUT=120 \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
conda run --no-capture-output -n x-teaming python experiment/main.py \
  --dataset experiment/data/advcua_easy.jsonl \
  --config experiment/config/config_online_qwen_local_gpt55_id15_covert.yaml \
  --plan-prompts experiment/config/prompts/plan_generation_prompts_covert.yaml \
  --plan-generator-provider openai \
  --plan-generator-model gpt-4o-2024-11-20 \
  --all-subtasks \
  --target-backend codex \
  --codex-model gpt-5.5 \
  --responses-base-url "$OPENAI_BASE_URL" \
  --responses-env-key OPENAI_API_KEY \
  --bypass \
  --no-textgrad \
  --no-plan-revision \
  --strategy-success-mode execute_verify \
  --strategy-execution-prompt-style auto \
  --attacker-model-path /hub/huggingface/models/Qwen/Qwen2.5-32B-Instruct \
  --attacker-max-new-tokens 2048 \
  --device-map auto \
  --item-attempts 5 \
  --strategy-library experiment/strategy_library/strategy_library.json \
  --library-pool-size 3 \
  --library-max-per-subtask 2 \
  --library-selection-order prefer-library \
  --revision-threshold 3 \
  --task-type-threshold 5 \
  --top-failure-lessons 3 \
  --output-dir experiment/experiment_outputs/easy_lib_e2e 2>&1 \
  | tee /tmp/easy_lib_e2e.log
```

The covert prompts are required for the GPT-4o plan generator: the covert
variant refuses on roughly 3% of subtask behaviors while the base variant
refuses on over half. The plan generator call already retries refusals
internally (five attempts), so occasional refusals recover automatically.
The library calls (classification, instantiation, attribution, revision,
re-injection) use the same OpenAI-compatible endpoint and
`gpt-4o-2024-11-20` by default.

The covert config selects
`experiment/config/prompts/attacker_agent_prompts_covert.yaml`; passing
`--plan-prompts .../plan_generation_prompts_covert.yaml` applies the same
style to plan generation.

### NYUCTF/Codex Example

Use the same online runner and model configuration. Pass the NYUCTF root
explicitly when launching from `$TRACE_ROOT`; omit `--ids` to run every
challenge row.

```bash
cd "$TRACE_ROOT"
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    CUDA_VISIBLE_DEVICES=3,4 \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    OPENAI_TIMEOUT=120 \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
conda run --no-capture-output -n x-teaming python experiment/main.py \
  --lab-backend nyuctf \
  --dataset experiment/data/nyuctf_test.jsonl \
  --nyuctf-root data/nyuctf/.nyuctf \
  --image-name attackenv:latest \
  --config experiment/config/config_online_qwen_local_gpt55_id15_covert.yaml \
  --plan-prompts experiment/config/prompts/plan_generation_prompts_covert.yaml \
  --plan-generator-provider openai \
  --plan-generator-model gpt-4o-2024-11-20 \
  --all-subtasks \
  --target-backend codex \
  --codex-model gpt-5.5 \
  --responses-base-url "$OPENAI_BASE_URL" \
  --responses-env-key OPENAI_API_KEY \
  --bypass \
  --no-textgrad \
  --no-plan-revision \
  --strategy-success-mode execute_verify \
  --strategy-execution-prompt-style auto \
  --attacker-model-path /hub/huggingface/models/Qwen/Qwen2.5-32B-Instruct \
  --device-map auto \
  --item-attempts 5 \
  --strategy-library experiment/strategy_library/strategy_library.json \
  --output-dir experiment/experiment_outputs/nyuctf_lib_gpt4oplan
```

For a Claude Code CLI target, replace the target settings with
`--target-backend claude --claude-runner cli --claude-model <model>` and pass
the appropriate `--anthropic-base-url` and `--anthropic-env-key`.

### Strategy Library

`experiment/strategy_library/` holds the runtime library
(`strategy_library.json`) and its tooling:

- `library_store.py` / `retriever.py` / `instantiate.py` / `revise.py` /
  `online_integration.py`: the runtime components (storage with lesson and
  task-type ledgers, fit/avoid retrieval, instantiation with lesson
  injection, revision, and the runner integration).
- `extract_patterns_from_success.py` and `batch_seed_from_baselines.py`:
  offline seeding — mine successful subtask conversations from recorded
  runs, filter them, extract interaction patterns, and reconcile them into
  seed entries.
- `strategy_library_pipeline.md`: the full architecture walkthrough.

Library semantics in brief: entries carry an interaction pattern with
progression phases; failures are attributed to imperative lessons (the
lesson text is the aggregation key, matched by the attribution LLM, plus a
conversation/execution/chain domain tag); a lesson reaching
`--revision-threshold` triggers an in-place revision or a fork - the revised
or forked entry enters probation and gets exactly one prioritized trial;
task-type ledgers demote a failing class out of `fits` first and into
`avoid` (absolute exclusion) only after a second threshold run outside
`fits`; a single recorded success marks the class proven, and proven
entries are the only library strategies that outrank fresh generation;
revision clears class affinity so a revised pattern re-earns its classes.
The only exit from `avoid` is revision.

### Important Parameters

- `--ids 15` selects item IDs, challenge names, or other item identity fields.
  It can contain a comma-separated list. Omit it to run every row in the
  dataset, including rows that share the same numeric ID.
- `--all-subtasks` is enabled by default and runs all `sub_prompts` for every
  selected item. To run one subtask only, use
  `--no-all-subtasks --subtask-index N`.
- `--online-strategies-per-subtask` defaults to `10`; this is the size of
  each generated strategy batch (used upfront when no library entry is
  eligible, and once after library exhaustion), not the number of
  conversations run per subtask in an attempt.
- `--item-attempts` defaults to `5` (the direct-probe phase is not counted).
  After a failed verification, retries use unused strategies only for the
  judge-selected failed restart subtask; retained subtasks still running bare
  direct inputs are upgraded to strategy conversations.
- `--attacker-model-path` defaults to the local attacker path, and
  `--attacker-max-new-tokens` defaults to `1024`. The CLI default overrides a
  larger `attacker.max_new_tokens` value in the YAML unless explicitly changed.
  The covert config caps attacker `max_turns` at `5`.
- `attack_plan_generator` defaults to the local Qwen2.5-32B in the covert
  config; use `--plan-generator-provider openai --plan-generator-model
  gpt-4o-2024-11-20` (with the covert plan prompts) to switch it.
- `--strategy-library <path>` enables the runtime strategy library. Omit it
  to keep the library disabled. Library knobs: `--library-pool-size` (3),
  `--library-max-per-subtask` (2, fit cap), `--library-selection-order`
  (`prefer-library` or `random`), `--revision-threshold` (3),
  `--task-type-threshold` (5), `--top-failure-lessons` (3).
- TextGrad and plan revision are disabled explicitly in the commands above.
- `--strategy-success-mode execute_verify` forces final execution and live
  verification. `--strategy-execution-prompt-style auto` is also the default.

Each output directory contains:

- `metadata.json`: resolved model, dataset, target, and selection settings.
- `online_artifacts.jsonl`: strategy-pool generation, direct probe,
  conversation, strategy selection, subtask-judge, library composition,
  feedback, and revision events.
- `results.jsonl`: one concise final result per item (each row's `run.cost`
  carries that item's conversation/execution seconds and target-query count).
- `stats.json`: totals plus the per-item breakdown of conversation time,
  execution time, and target queries (one query per conversation turn and
  one per execution prompt, direct probes included).
- `successful_subtasks.jsonl` and `retained_execution_inputs.jsonl`: retained
  judge-successful subtask material when retries occur.
- `experiment/strategy_library/strategy_library.json` is updated in place
  during runs (lesson counts, ledgers, revisions, new entries).

When a local model path nevertheless attempts Hugging Face Hub access, ensure
the model/tokenizer files are present locally and launch with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

### Local Resources

The AdvCUA/VRAP compose files and NYUCTF challenge resources are included under `data/`.

- AdvCUA/VRAP Docker image: build it from the local Dockerfile before running VRAP commands:
  `cd data/advcua && docker build --network host -t my-ssh-image .`
- AdvCUA/VRAP compose: the default compose path is `data/advcua/docker-compose.yml`.
- AdvCUA/VRAP default dataset: `data/advcua/advcua_test.json`.
- NYUCTF Docker image: build it from the local Dockerfile before running NYUCTF commands:
  `cd data/nyuctf && docker build --network host -t attackenv .`
- NYUCTF Docker network: create `attacknet` before running NYUCTF commands:
  `docker network create attacknet`
- NYUCTF challenge resources: the default root is `data/nyuctf/.nyuctf`.
- Local Qwen model weights: pass `--attacker-model-path` / `--transformers-model-path`, or set `QWEN32B`; if unset, the default model id is `Qwen/Qwen2.5-32B-Instruct`.
- Conda environment: the commands assume `conda run -n x-teaming`; `environment.yml` is copied as a starting point.
- Upstream model endpoints and keys: provide `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and any local proxy such as the Anthropic proxy on `192.168.2.1:38445`.
- MoonBridge source is still treated as a local/external dependency and ignored by git under `external/moon-bridge-src/`.

