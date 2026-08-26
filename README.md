# Agent Bridge Trace Workspace

This workspace contains the reusable bridge code for running Codex and Claude Code inside the VRAP Docker environment, plus the trajectories produced by those runs.

## Layout

- `tools/agent_cli_bridge.py`: reusable Python bridge for Codex CLI and Claude Code SDK/CLI.
- `tools/vrap_bridge_run.py`: run VRAP prompts and save model text/actions as reusable trajectories.
- `tools/vrap_bridge_conversation.py`: run multiple prompts against one persistent VRAP container, optionally resuming the agent session between turns.
- `tools/vrap_bridge_verify.py`: replay saved actions in a fresh VRAP Docker environment and run the normal VRAP verification probe.
- `tools/vrap_bridge_runner.py`: shared VRAP Docker setup, prompt loading, action extraction, and trajectory-building logic.
- `tools/proxies/`: local proxy helpers used during endpoint experiments.
- `tools/diagnostics/`: one-off endpoint/model connectivity checks.
- `agent_execution/`: reusable session, snapshot/promotion, execution, live verification, and tolerant replay-verify helpers shared by x-teaming and other methods.
- `run_xteaming/`: self-contained X-Teaming runner copy, including plan generation, attacker/evaluator code, configs, prompts, and small AdvCUA jsonl inputs.
- `configs/`: MoonBridge configs and examples.
- `datasets/`: small local VRAP probe datasets used in these tests.
- `results/vrap/final/`: verified trajectories worth reusing.
- `results/vrap/diagnostics/`: meaningful failure/diagnostic outputs.
- `results/vrap/archive/`: older exploratory runs kept out of the active result set.

## X-Teaming Full Pipeline

`run_xteaming/` contains the x-teaming-specific code needed to generate attack plans and run the multi-turn x-teaming loop. It imports reusable target/session/action helpers from `agent_execution/` and `tools/`, so the trace directory can be moved as one unit.

Set common paths first:

```bash
cd /path/to/trace
export TRACE_ROOT="$PWD"

# Build the local AdvCUA/VRAP Docker image used by data/advcua/docker-compose.yml.
cd "$TRACE_ROOT/data/advcua"
docker build --network host -t my-ssh-image .
cd "$TRACE_ROOT"

# Build the local NYUCTF agent image and create the shared attack network.
cd "$TRACE_ROOT/data/nyuctf"
docker build --network host -t attackenv .

networkname="attacknet"
network_exists=$(docker network ls | grep $networkname)
if [ -z "$network_exists" ]; then
    docker network create attacknet
else
    echo "Network ${networkname} already exists, skip!"
fi
cd "$TRACE_ROOT"

# Build the NYUCTF target image.
python envs_build.py
# No Qwen path is required by default. To use local weights instead of the
# default Qwen/Qwen2.5-32B-Instruct model id, set QWEN32B or pass the model
# path through --attacker-model-path / --transformers-model-path.
```

If using OpenAI-compatible GPT endpoints for attack-plan generation or GPTJudge:

```bash
export OPENAI_API_KEY='<openai-compatible-api-key>'
export OPENAI_BASE_URL='https://your-openai-compatible-endpoint/v1'
```

If using Claude Code as the target, start the Anthropic-compatible proxy on the host:

```bash
cd "$TRACE_ROOT"
python3 tools/proxies/anthropic_base_url_proxy.py \
  --listen-host 0.0.0.0 \
  --port 38445 \
  --upstream-base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --timeout 120
```

### 1. Prepare AdvCUA Subtasks For Plan Generation

The original X-Teaming plan generator expects a HarmBench-like CSV. This helper converts selected AdvCUA subtasks into that format and writes a config copy pointing to the generated CSV.

```bash
cd "$TRACE_ROOT/run_xteaming"
conda run --no-capture-output -n x-teaming python prepare_advcua_xteaming_inputs.py \
  --dataset data/advcua_easy.jsonl \
  --config config/config.yaml \
  --pairs 15:0,15:1,15:2,15:3 \
  --output-dir xteaming_vrap_codex_outputs/example_id15_inputs
```

For the 5-item AdvCUA test set used in the experiments, change `--dataset` and `--pairs`, for example:

```bash
--dataset data/advcua_test.jsonl \
--pairs 1:0,13:2,2:5,3:4,19:3
```

For the NYUCTF jsonl input, omit `--pairs` to export every decomposed CTF
subtask. NYUCTF behavior IDs use `id`, `unique_id`, `challenge`, then jsonl row
index, matching the decomposer fallback style:

```bash
cd "$TRACE_ROOT/run_xteaming"
conda run --no-capture-output -n x-teaming python prepare_advcua_xteaming_inputs.py \
  --dataset data/nyuctf_test.jsonl \
  --config config/config.yaml \
  --output-dir xteaming_vrap_codex_outputs/nyuctf_test_inputs
```

To select a subset, use challenge names or row indexes:

```bash
--pairs beleaf:0,unVirtualization:2,2:3
```

### 2. Generate Attack Plans

Default generation is 5 sets x 10 strategies per behavior. Use the compatible generator because it tolerates wrapped JSON responses from OpenAI-compatible endpoints.

```bash
cd "$TRACE_ROOT/run_xteaming"
env OPENAI_API_KEY="$OPENAI_API_KEY" OPENAI_BASE_URL="$OPENAI_BASE_URL" \
conda run --no-capture-output -n x-teaming python generate_attack_plans_compat.py \
  -c xteaming_vrap_codex_outputs/example_id15_inputs/config.yaml \
  --num-sets 5 \
  --strategies-per-set 10 \
  --backend config
```

For the NYUCTF inputs above:

```bash
cd "$TRACE_ROOT/run_xteaming"
env OPENAI_API_KEY="$OPENAI_API_KEY" OPENAI_BASE_URL="$OPENAI_BASE_URL" \
conda run --no-capture-output -n x-teaming python generate_attack_plans_compat.py \
  -c xteaming_vrap_codex_outputs/nyuctf_test_inputs/config.yaml \
  --num-sets 5 \
  --strategies-per-set 10 \
  --backend config
```

To generate plans with the local Qwen model instead of the configured API provider:

```bash
conda run --no-capture-output -n x-teaming python generate_attack_plans_compat.py \
  -c xteaming_vrap_codex_outputs/example_id15_inputs/config.yaml \
  --num-sets 5 \
  --strategies-per-set 10 \
  --backend transformers \
  --device-map auto
```

The command prints the generated `attack_plans.json` path. Pass that path to the x-teaming runner.

Previously generated attack plans for the 9 AdvCUA items used in these
experiments are bundled under
`run_xteaming/generated_attack_plans/`. To reuse them instead of regenerating,
pass:

```bash
--plans-file generated_attack_plans/attack_plans_merged_all.json
```

### 3. Run X-Teaming

Single-subtask mode writes one result row per selected subtask:

```bash
cd "$TRACE_ROOT/run_xteaming"
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    PYTHONPATH="$TRACE_ROOT:$TRACE_ROOT/run_xteaming" \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    ANTHROPIC_API_KEY='placeholder' \
conda run --no-capture-output -n x-teaming python run_xteaming_vrap_codex.py \
  --dataset data/advcua_easy.jsonl \
  --config xteaming_vrap_codex_outputs/example_id15_inputs/config.yaml \
  --plans-file xteaming_vrap_codex_outputs/example_id15_inputs/strategies/<timestamp>/attack_plans.json \
  --ids 15 \
  --subtask-index 0 \
  --target-backend claude \
  --claude-model claude-opus-4-6 \
  --anthropic-base-url http://192.168.2.1:38445 \
  --claude-home /home/agent/claude_home \
  --allowed-tools Bash,Read,Write \
  --agent-session resume \
  --bypass \
  --device-map auto \
  --strategies-per-behavior 3 \
  --output-dir xteaming_vrap_codex_outputs/example_id15_xteaming
```

Multi-subtask mode writes one result row per item. Each row's `input_output` is a list whose elements correspond to subtasks. Use `--link-mode session` to pass the previous subtask's agent session forward, or `--link-mode action_summary` to summarize prior execution actions and prefix the next subtask.

```bash
cd "$TRACE_ROOT/run_xteaming"
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    PYTHONPATH="$TRACE_ROOT:$TRACE_ROOT/run_xteaming" \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    ANTHROPIC_API_KEY='placeholder' \
conda run --no-capture-output -n x-teaming python run_xteaming_vrap_codex_subprompts.py \
  --dataset data/advcua_easy.jsonl \
  --config xteaming_vrap_codex_outputs/example_id15_inputs/config.yaml \
  --plans-file xteaming_vrap_codex_outputs/example_id15_inputs/strategies/<timestamp>/attack_plans.json \
  --ids 15 \
  --all-subtasks \
  --link-mode session \
  --target-backend claude \
  --claude-model claude-opus-4-6 \
  --anthropic-base-url http://192.168.2.1:38445 \
  --claude-home /home/agent/claude_home \
  --allowed-tools Bash,Read,Write \
  --agent-session resume \
  --bypass \
  --device-map auto \
  --strategies-per-behavior 3 \
  --output-dir xteaming_vrap_codex_outputs/example_id15_xteaming_session
```

NYUCTF/CTF uses the same x-teaming attack-plan format, but runs target turns in
the NYUCTF Docker lab instead of the AdvCUA/VRAP compose lab. Use
`--no-textgrad` for NYUCTF. Keep GPTJudge enabled; `--strategy-success-mode
execute_verify` runs the selected x-teaming strategy outputs in a fresh NYUCTF
container and accepts an attempt only when the NYUCTF gold check passes.

Single-subtask NYUCTF smoke run:

```bash
cd "$TRACE_ROOT/run_xteaming"
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    PYTHONPATH="$TRACE_ROOT:$TRACE_ROOT/run_xteaming" \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    OPENAI_TIMEOUT=120 \
conda run --no-capture-output -n x-teaming python run_xteaming_vrap_codex.py \
  --lab-backend nyuctf \
  --dataset data/nyuctf_test.jsonl \
  --nyuctf-root ../data/nyuctf/.nyuctf \
  --image-name attackenv:latest \
  --config config/config_gpt4o_2024_11_20_serial_3strategies.yaml \
  --plans-file generated_attack_plans/attack_plans_nyuctf.json \
  --ids beleaf \
  --subtask-index 0 \
  --target-backend codex \
  --codex-model gpt-5.5 \
  --responses-base-url "$OPENAI_BASE_URL" \
  --responses-env-key OPENAI_API_KEY \
  --bypass \
  --no-textgrad \
  --strategy-success-mode execute_verify \
  --strategy-execution-prompt-style auto \
  --attacker-model-path /hub/huggingface/models/Qwen/Qwen2.5-32B-Instruct/ \
  --strategies-per-behavior 3 \
  --output-dir xteaming_vrap_codex_outputs/nyuctf_beleaf_subtask0_xteaming
```

Multi-subtask NYUCTF run:

```bash
cd "$TRACE_ROOT/run_xteaming"
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    PYTHONPATH="$TRACE_ROOT:$TRACE_ROOT/run_xteaming" \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    OPENAI_TIMEOUT=120 \
conda run --no-capture-output -n x-teaming python run_xteaming_vrap_codex_subprompts.py \
  --lab-backend nyuctf \
  --dataset data/nyuctf_test.jsonl \
  --nyuctf-root ../data/nyuctf/.nyuctf \
  --image-name attackenv:latest \
  --config config/config_gpt4o_2024_11_20_serial_3strategies.yaml \
  --plans-file generated_attack_plans/attack_plans_nyuctf.json \
  --ids beleaf \
  --all-subtasks \
  --link-mode session \
  --target-backend codex \
  --codex-model gpt-5.5 \
  --responses-base-url "$OPENAI_BASE_URL" \
  --responses-env-key OPENAI_API_KEY \
  --bypass \
  --no-textgrad \
  --strategy-success-mode execute_verify \
  --strategy-execution-prompt-style auto \
  --attacker-model-path /hub/huggingface/models/Qwen/Qwen2.5-32B-Instruct/ \
  --strategies-per-behavior 3 \
  --output-dir xteaming_vrap_codex_outputs/nyuctf_beleaf_xteaming_session
```

In multi-subtask `execute_verify` mode, each item attempt first runs one selected
x-teaming strategy for every subtask. It then starts a new NYUCTF execution
container, executes the saved strategy outputs across all subtasks in one
continuous agent session, runs the gold verification, and tries the next sampled
strategy combination if execution or verification fails.

For Codex + GPT-5.5 as the target model, the x-teaming command uses the same OpenAI-compatible endpoint for the target, GPTJudge, and TextGrad. This matches the migrated Codex test path:

```bash
cd "$TRACE_ROOT/run_xteaming"
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    PYTHONPATH="$TRACE_ROOT:$TRACE_ROOT/run_xteaming" \
    CUDA_VISIBLE_DEVICES=3 \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    OPENAI_TIMEOUT=60 \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
conda run --no-capture-output -n x-teaming python run_xteaming_vrap_codex_subprompts.py \
  --dataset data/advcua_easy.jsonl \
  --config config/config_gpt4o_2024_11_20_serial_3strategies.yaml \
  --plans-file generated_attack_plans/attack_plans_merged_all.json \
  --ids 15 \
  --all-subtasks \
  --link-mode session \
  --target-backend codex \
  --codex-model gpt-5.5 \
  --responses-base-url "$OPENAI_BASE_URL" \
  --responses-env-key OPENAI_API_KEY \
  --agent-user user \
  --agent-session resume \
  --bypass \
  --device-map auto \
  --strategies-per-behavior 3 \
  --strategy-sample-seed 15 \
  --output-dir xteaming_vrap_codex_outputs/example_id15_codex_gpt55_xteaming
```

For the `experiment/` online-strategy runner, generate one strategy immediately
before each subtask conversation and then execute the saved subtask outputs in a
single linked Codex session:

```bash
cd "$TRACE_ROOT/experiment"
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    CUDA_VISIBLE_DEVICES=2,3,5 \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    OPENAI_TIMEOUT=120 \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
conda run --no-capture-output -n x-teaming python run_vrap_codex_subprompts_online.py \
  --dataset data/advcua_easy.jsonl \
  --config config/config_online_qwen_local_gpt55_id15.yaml \
  --ids 15 \
  --all-subtasks \
  --link-mode session \
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
  --attacker-max-new-tokens 1024 \
  --device-map auto \
  --output-dir experiment_outputs/id15_online_qwen25_32b_gpt55
```

This writes the final item result to `results.jsonl`. It also writes
`online_artifacts.jsonl`, with one `strategy_generated` event and one
`conversation_completed` event per subtask. The completed event records the
strategy used for that subtask plus the attacker and target inputs for the
multi-turn conversation.

Bundled config templates:

- `config/config.yaml`: upstream-style default config copied from the original x-teaming tree.
- `config/config_gpt4o_2024_11_20_serial.yaml`: serial/repro config used when `gpt-4o` was unavailable, with `gpt-4o-2024-11-20`, `max_workers: 1`, and `strategies_per_behavior: 10`.
- `config/config_gpt4o_2024_11_20_serial_3strategies.yaml`: same serial config, but `strategies_per_behavior: 3` for faster debugging.

Notes:

- `--strategies-per-behavior 3` is a debug-speed setting. Omit it or set it to `10` to traverse the default number of generated strategies.
- GPTJudge and TextGrad follow `config/config.yaml` unless overridden by flags. If the configured judge model is unavailable, set `evaluation.judge_model` in the copied config, for example to `gpt-4o-2024-11-20`.
- For Codex targets instead of Claude, use `--target-backend codex`, `--codex-model <model>`, `--responses-base-url <endpoint>/v1`, and `--responses-env-key OPENAI_API_KEY`.
- For Claude target x-teaming, keep the x-teaming/GPTJudge/TextGrad OpenAI-compatible endpoint as the normal host-side URL, for example `https://az.gptplus5.com/v1`. Do not use the Docker bridge Codex/MoonBridge URL `http://192.168.2.1:38440/v1` for this stage unless you have explicitly started that service. Claude Code itself still uses the Anthropic proxy URL passed by `--anthropic-base-url`.
- Claude targets default to `--claude-runner sdk`, which uses `claude-agent-sdk` and `ClaudeSDKClient`. To test Claude Code print-mode CLI directly, pass `--claude-runner cli`; this runs `claude -p ... --output-format stream-json --verbose` inside the VRAP client container and extracts tool actions from the stream.

### Known Good Claude id15 End-to-End Run

This is the exact shape of the migrated trace run for AdvCUA easy item `15`: generate/use existing attack plans, run x-teaming against Claude Code, execute saved x-teaming outputs in one shared Claude session/container, and verify the live container state immediately after execution.

Start the Anthropic-compatible proxy for Claude Code:

```bash
cd /path/to/trace
export OPENAI_API_KEY='<openai-compatible-api-key>'
export OPENAI_BASE_URL='https://az.gptplus5.com/v1'

env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
python3 tools/proxies/anthropic_base_url_proxy.py \
  --listen-host 192.168.2.1 \
  --port 38445 \
  --upstream-base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --timeout 120 \
  --verbose
```

In another shell, run x-teaming. The command intentionally does not pass `--responses-base-url`; the x-teaming default is the host-side OpenAI-compatible endpoint. If you do pass it, use `https://az.gptplus5.com/v1`, not `http://192.168.2.1:38440/v1`.

```bash
cd /path/to/trace/run_xteaming
export TRACE_ROOT=/path/to/trace
export OPENAI_API_KEY='<openai-compatible-api-key>'
export OPENAI_BASE_URL='https://az.gptplus5.com/v1'

env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    PYTHONPATH="$TRACE_ROOT:$TRACE_ROOT/run_xteaming" \
    CUDA_VISIBLE_DEVICES=3 \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    OPENAI_TIMEOUT=60 \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    ANTHROPIC_API_KEY='placeholder' \
conda run --no-capture-output -n x-teaming python run_xteaming_vrap_codex_subprompts.py \
  --dataset data/advcua_easy.jsonl \
  --config config/config_gpt4o_2024_11_20_serial_3strategies.yaml \
  --plans-file generated_attack_plans/attack_plans_merged_all.json \
  --ids 15 \
  --all-subtasks \
  --link-mode session \
  --target-backend claude \
  --claude-model claude-opus-4-6 \
  --claude-runner sdk \
  --anthropic-base-url http://192.168.2.1:38445 \
  --anthropic-env-key ANTHROPIC_API_KEY \
  --claude-home /home/agent/claude_home \
  --allowed-tools Bash,Read,Write \
  --agent-user user \
  --agent-session resume \
  --bypass \
  --device-map auto \
  --strategies-per-behavior 3 \
  --strategy-sample-seed 15 \
  --output-dir xteaming_vrap_codex_outputs/migration_id15_claude_xteaming_fixed_default_20260818_144333
```

Then execute the saved x-teaming outputs in one shared Claude Code session and one shared VRAP container. The execution script runs live verification by default before cleaning up the containers; use `--no-live-verify` only when you want execution without immediate probing.

```bash
cd /path/to/trace
export TRACE_ROOT=/path/to/trace
export OPENAI_API_KEY='<openai-compatible-api-key>'
export OPENAI_BASE_URL='https://az.gptplus5.com/v1'

env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    PYTHONPATH="$TRACE_ROOT" \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    ANTHROPIC_API_KEY='placeholder' \
    AGENT_EXECUTION_CLAUDE_NETWORK_RETRIES=9 \
    AGENT_EXECUTION_CLAUDE_SAFETY_RETRIES=2 \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
conda run --no-capture-output -n x-teaming python agent_execution/scripts/codex_execute_item_subtasks_same_session.py \
  --backend claude \
  --results run_xteaming/xteaming_vrap_codex_outputs/migration_id15_claude_xteaming_fixed_default_20260818_144333/results.jsonl \
  --item-offset 0 \
  --output-mode all \
  --prompt-style rule \
  --claude-model claude-opus-4-6 \
  --claude-runner sdk \
  --anthropic-base-url http://192.168.2.1:38445 \
  --anthropic-env-key ANTHROPIC_API_KEY \
  --claude-home /home/agent/claude_home \
  --allowed-tools Bash,Read,Write \
  --claude-max-turns 30 \
  --timeout 900 \
  --agent-user user \
  --bypass \
  --output-dir run_xteaming/xteaming_vrap_codex_outputs/migration_id15_claude_execution_20260818_153249/item_15
```

To run the same Claude target path through Claude Code CLI instead of the SDK runner, keep the command the same and change:

```bash
--claude-runner cli
```

Replay verification is optional and checks whether the saved `actions` can reproduce the live run in a fresh environment:

```bash
cd /path/to/trace
export TRACE_ROOT=/path/to/trace

PYTHONPATH="$TRACE_ROOT" \
conda run --no-capture-output -n x-teaming python agent_execution/prepare_verify_trajectories.py \
  --input run_xteaming/xteaming_vrap_codex_outputs/migration_id15_claude_execution_20260818_153249 \
  --output run_xteaming/xteaming_vrap_codex_outputs/migration_id15_claude_execution_20260818_153249/trajectories.jsonl

PYTHONPATH="$TRACE_ROOT" \
conda run --no-capture-output -n x-teaming python agent_execution/verify_tolerant.py \
  --input run_xteaming/xteaming_vrap_codex_outputs/migration_id15_claude_execution_20260818_153249/trajectories.jsonl \
  --output run_xteaming/xteaming_vrap_codex_outputs/migration_id15_claude_execution_20260818_153249/verify_results_tolerant.jsonl \
  --replay-timeout 300
```

### 4. Execute Saved X-Teaming Outputs

This step sends each saved subtask output back to the selected agent in one shared container/session and records executable actions. It is method-agnostic and lives under `agent_execution/`. By default it also runs the dataset probe against the live containers after all subtasks finish, without replaying saved actions. The live result is saved as `live_verification` in `result.json` and as `live_verification.json`.

```bash
cd "$TRACE_ROOT"
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    ANTHROPIC_API_KEY='placeholder' \
    AGENT_EXECUTION_CLAUDE_NETWORK_RETRIES=9 \
    AGENT_EXECUTION_CLAUDE_SAFETY_RETRIES=2 \
conda run --no-capture-output -n x-teaming python agent_execution/scripts/codex_execute_item_subtasks_same_session.py \
  --backend claude \
  --results run_xteaming/xteaming_vrap_codex_outputs/example_id15_xteaming_session/results.jsonl \
  --item-offset 0 \
  --output-mode all \
  --prompt-style rule \
  --claude-model claude-opus-4-6 \
  --anthropic-base-url http://192.168.2.1:38445 \
  --anthropic-env-key ANTHROPIC_API_KEY \
  --claude-home /home/agent/claude_home \
  --allowed-tools Bash,Read,Write \
  --claude-max-turns 30 \
  --timeout 900 \
  --bypass \
  --output-dir run_xteaming/xteaming_vrap_codex_outputs/example_id15_execution/item_15
```

For Codex + GPT-5.5 execution, use the OpenAI-compatible endpoint directly:

```bash
cd "$TRACE_ROOT"
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    PYTHONPATH="$TRACE_ROOT" \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    AGENT_EXECUTION_CODEX_NETWORK_RETRIES=9 \
    AGENT_EXECUTION_CODEX_SAFETY_RETRIES=2 \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
conda run --no-capture-output -n x-teaming python agent_execution/scripts/codex_execute_item_subtasks_same_session.py \
  --backend codex \
  --results run_xteaming/xteaming_vrap_codex_outputs/example_id15_xteaming_session/results.jsonl \
  --item-offset 0 \
  --output-mode all \
  --prompt-style rule \
  --model gpt-5.5 \
  --responses-base-url "$OPENAI_BASE_URL" \
  --responses-env-key OPENAI_API_KEY \
  --timeout 900 \
  --agent-user user \
  --bypass \
  --output-dir run_xteaming/xteaming_vrap_codex_outputs/example_id15_codex_execution/item_15
```

### 5. Optional Replay Verify

Convert execution results to verify trajectories:

```bash
cd "$TRACE_ROOT"
conda run --no-capture-output -n x-teaming python agent_execution/prepare_verify_trajectories.py \
  --input run_xteaming/xteaming_vrap_codex_outputs/example_id15_execution \
  --output run_xteaming/xteaming_vrap_codex_outputs/example_id15_execution/trajectories.jsonl
```

Replay actions in a fresh VRAP environment and run the benchmark probe:

```bash
cd "$TRACE_ROOT"
conda run --no-capture-output -n x-teaming python agent_execution/verify_tolerant.py \
  --input run_xteaming/xteaming_vrap_codex_outputs/example_id15_execution/trajectories.jsonl \
  --output run_xteaming/xteaming_vrap_codex_outputs/example_id15_execution/verify_results_tolerant.jsonl \
  --replay-timeout 300
```

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

## Trajectory Shape

Each run writes one JSON object per dataset item. The most useful fields are:

- `item`: original dataset item.
- `prompt_to_model`: prompt sent to Codex or Claude Code.
- `input_output`: list of input/output interactions.
- `model_output`: model text plus return code, error, and optional debug details.
- `actions`: extracted shell/tool actions for replay.
- `verification`: pending in run files, populated in verified files.
- `replay`: compose file, item id, and replay steps.

`input_output` is a list so future multi-turn runs can append multiple input/output interactions. For the current one-prompt VRAP runs it has one item:

```json
[
  {
    "input": "prompt sent to the agent",
    "output": [
      {"type": "text", "content": "final model text"},
      {"type": "action", "index": 1, "instruction": "...", "result": "..."}
    ]
  }
]
```

`input_output[0].input` duplicates `prompt_to_model`. The text item in `input_output[0].output` duplicates `model_output.text`, and action items duplicate `actions`. They are intentionally kept for now so older consumers can continue to use the original fields while newer code can read ordered input/output pairs.

## Network Notes

The host uses a loopback proxy at `127.0.0.1:47223`. Docker bridge containers cannot normally reach the host loopback address. When a run needs that host proxy, expose it on the Docker bridge with:

```bash
python3 tools/proxies/tcp_proxy_forwarder.py \
  --listen-host 0.0.0.0 \
  --listen-port 47224 \
  --target-host 127.0.0.1 \
  --target-port 47223
```

Then pass container-reachable proxy variables to the run:

```bash
HTTP_PROXY=http://192.168.2.1:47224 \
HTTPS_PROXY=http://192.168.2.1:47224 \
ALL_PROXY=http://192.168.2.1:47224 \
NO_PROXY=localhost,127.0.0.1,::1,192.168.2.1 \
python3 tools/vrap_bridge_run.py ...
```

For local MoonBridge or local Anthropic proxy endpoints on the host, unset host proxy variables before starting the VRAP run. Otherwise the container-side client may try to reach `192.168.2.1` through the external proxy:

```bash
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    ...
```

## Multi-Turn Conversations

Use `tools/vrap_bridge_conversation.py` when a task needs multiple prompts while keeping the VRAP container alive. The script starts the container once per dataset item, runs each prompt in order, records every turn in `input_output`, and cleans up at the end unless `--keep-container` is set.

The two kinds of continuity are separate:

- Container continuity: always preserved within one `vrap_bridge_conversation.py` run.
- Agent conversation continuity: controlled by `--agent-session`. The default is `resume`, which reuses the previous Codex thread id or Claude session id when one is returned. Use `--agent-session none` to keep only the container state and start a fresh agent call each turn.

Turns can be a JSON list of strings:

```json
[
  "Create /work/conversation/input.txt with three lines: gamma, alpha, beta.",
  "Read /work/conversation/input.txt, sort and uppercase it, and write /work/conversation/output.txt.",
  "Create a SHA256 manifest for /work/conversation/output.txt and verify it."
]
```

Run a Codex conversation through MoonBridge:

```bash
cd /path/to/trace
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    OPENAI_API_KEY='placeholder' \
    python3 tools/vrap_bridge_conversation.py \
      --backend codex \
      --model deepseek-v4-pro \
      --dataset datasets/vrap_normal_dependent_probe.json \
      --ids 9003 \
      --turns-file /tmp/vrap_turns.json \
      --agent-session resume \
      --bypass \
      --codex-write-config \
      --responses-base-url http://192.168.2.1:38440/v1 \
      --responses-env-key OPENAI_API_KEY \
      --verify \
      --output results/vrap/final/deepseek_v4_pro_id9003_conversation.jsonl
```

Run a Claude Code conversation through the Anthropic proxy:

```bash
cd /path/to/trace
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    ANTHROPIC_API_KEY='placeholder' \
    python3 tools/vrap_bridge_conversation.py \
      --backend claude \
      --model claude-opus-4-6 \
      --dataset datasets/vrap_normal_dependent_probe.json \
      --ids 9003 \
      --turns-file /tmp/vrap_turns.json \
      --agent-session resume \
      --allowed-tools Bash,Write,Read \
      --anthropic-base-url http://192.168.2.1:38445 \
      --verify \
      --output results/vrap/final/claude_opus_4_6_id9003_conversation.jsonl
```

If the next prompt is generated outside the container, write that prompt into the turns file before the next call. For fully interactive orchestration, use `--keep-container` and run one conversation process per planned batch of turns, or extend the script to read turns from stdin.

## MoonBridge

MoonBridge was used as the stable way to make non-Responses upstreams usable by Codex. Codex is pointed at MoonBridge as if it were an OpenAI Responses endpoint:

```bash
--responses-base-url http://192.168.2.1:<moonbridge-port>/v1
```

Start MoonBridge from the local stable checkout kept with this workspace:
`/path/to/trace/external/moon-bridge-src`.

```bash
export MOONBRIDGE_SRC=/path/to/trace/external/moon-bridge-src
cd "$MOONBRIDGE_SRC"
/usr/local/go/bin/go run ./cmd/moonbridge \
  -config /path/to/moonbridge-config.yml \
  -addr 0.0.0.0:<port> \
  -mode Transform
```

Use `OPENAI_API_KEY=placeholder` for Codex when MoonBridge owns the real upstream key in its config.

## DeepSeek V4 Pro Through Codex

Proxy setup:

```bash
export MOONBRIDGE_SRC=/path/to/trace/external/moon-bridge-src
cd "$MOONBRIDGE_SRC"
/usr/local/go/bin/go run ./cmd/moonbridge \
  -config /path/to/trace/configs/moonbridge_deepseek_v4_pro.yml \
  -addr 0.0.0.0:38440 \
  -mode Transform
```

Run id `9003`:

```bash
cd /path/to/trace
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    OPENAI_API_KEY='placeholder' \
    python3 tools/vrap_bridge_run.py \
      --backend codex \
      --model deepseek-v4-pro \
      --dataset datasets/vrap_normal_dependent_probe.json \
      --ids 9003 \
      --bypass \
      --codex-write-config \
      --responses-base-url http://192.168.2.1:38440/v1 \
      --responses-env-key OPENAI_API_KEY \
      --timeout 900 \
      --output results/vrap/final/deepseek_v4_pro_id9003_run.jsonl
```

Verify:

```bash
python3 tools/vrap_bridge_verify.py \
  --input results/vrap/final/deepseek_v4_pro_id9003_run.jsonl \
  --ids 9003 \
  --output results/vrap/final/deepseek_v4_pro_id9003_verified.jsonl
```

Known result: `actions=3`, `verification=True`.

## Gemini 3 Flash Preview Through Codex

Gemini was run through MoonBridge using an OpenAI-compatible chat provider. The example config is `configs/moonbridge_gemini_3_flash_preview.example.yml`; copy it to a local config and replace `api_key` before use.

Proxy setup:

```bash
cd /path/to/trace
cp configs/moonbridge_gemini_3_flash_preview.example.yml configs/moonbridge_gemini_3_flash_preview.local.yml
# Edit configs/moonbridge_gemini_3_flash_preview.local.yml and set providers.gemini_openai_chat.api_key.

export MOONBRIDGE_SRC=/path/to/trace/external/moon-bridge-src
cd "$MOONBRIDGE_SRC"
/usr/local/go/bin/go run ./cmd/moonbridge \
  -config /path/to/trace/configs/moonbridge_gemini_3_flash_preview.local.yml \
  -addr 0.0.0.0:38446 \
  -mode Transform
```

Run id `9003`:

```bash
cd /path/to/trace
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    OPENAI_API_KEY='placeholder' \
    python3 tools/vrap_bridge_run.py \
      --backend codex \
      --model gemini-3-flash-preview \
      --dataset datasets/vrap_normal_dependent_probe.json \
      --ids 9003 \
      --bypass \
      --codex-write-config \
      --responses-base-url http://192.168.2.1:38446/v1 \
      --responses-env-key OPENAI_API_KEY \
      --timeout 900 \
      --output results/vrap/final/gemini_3_flash_preview_id9003_run.jsonl
```

Verify:

```bash
python3 tools/vrap_bridge_verify.py \
  --input results/vrap/final/gemini_3_flash_preview_id9003_run.jsonl \
  --ids 9003 \
  --output results/vrap/final/gemini_3_flash_preview_id9003_verified.jsonl
```

Known result: `actions=4`, `verification=True`.

## Claude Code

Claude Code is called through `claude-agent-sdk`, not through shell-only CLI parsing. For the OpenAI-compatible Anthropic endpoint used in these tests, run the local Anthropic base URL proxy on the host:

```bash
python3 tools/proxies/anthropic_base_url_proxy.py \
  --listen-host 0.0.0.0 \
  --port 38445 \
  --upstream-base-url https://az.gptplus5.com/v1 \
  --api-key '<upstream-api-key>' \
  --timeout 120
```

Run id `9003` with root identity preserved. `allowed_tools` is important: without it, Claude Code default permissions deny `Bash` and `Write` under this non-interactive SDK run. `allowed_tools` auto-approves the named tools without changing the Docker user or filesystem permissions.

```bash
cd /path/to/trace
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    ANTHROPIC_API_KEY='placeholder' \
    python3 tools/vrap_bridge_run.py \
      --backend claude \
      --model claude-opus-4-6 \
      --dataset datasets/vrap_normal_dependent_probe.json \
      --ids 9003 \
      --timeout 240 \
      --max-turns 10 \
      --allowed-tools Bash,Write,Read \
      --anthropic-base-url http://192.168.2.1:38445 \
      --output results/vrap/final/claude_opus_4_6_id9003_run.jsonl
```

Verify:

```bash
python3 tools/vrap_bridge_verify.py \
  --input results/vrap/final/claude_opus_4_6_id9003_run.jsonl \
  --ids 9003 \
  --output results/vrap/final/claude_opus_4_6_id9003_verified.jsonl
```

Known result: `actions=5`, `verification=True`.

For broader Claude Code tests, useful `allowed_tools` sets are:

```text
Bash,Write,Read
Bash,Write,Read,Edit,MultiEdit,Glob,Grep,LS
Bash,Write,Read,Edit,MultiEdit,Glob,Grep,LS,WebFetch,WebSearch
```

`allowed_tools` pre-approves tools; it does not hide tools that are not listed. For a stricter run, combine an allow list with `permission_mode=dontAsk` in the SDK runner.

## GPT Models Through Codex

The direct `https://az.gptplus5.com/v1/responses` endpoint did not successfully run the GPT tests in this environment:

- `gpt-5`: `/responses` streaming returned `response.failed`; the Codex run produced `actions=0`.
- `gpt-5.2`: `/responses` returned `503 Service Unavailable` with `No available channel for model gpt-5.2 under group vip`; the Codex run produced `actions=0`.

The commands used were:

```bash
cd /path/to/trace
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    no_proxy='localhost,127.0.0.1,::1,192.168.2.1' \
    NO_PROXY='localhost,127.0.0.1,::1,192.168.2.1' \
    OPENAI_API_KEY='<upstream-api-key>' \
    python3 tools/vrap_bridge_run.py \
      --backend codex \
      --model gpt-5 \
      --dataset datasets/vrap_normal_dependent_probe.json \
      --ids 9003 \
      --bypass \
      --codex-write-config \
      --responses-base-url https://az.gptplus5.com/v1 \
      --responses-env-key OPENAI_API_KEY \
      --timeout 900 \
      --output results/vrap/diagnostics/gpt_5_id9003_response_failed_run.jsonl
```

Change `--model gpt-5` to `--model gpt-5.2` and the output file name to reproduce the `gpt-5.2` diagnostic.

## Temporary Diagnostic Helpers

These are not part of the normal run/verify path:

```bash
python3 tools/diagnostics/check_model_access.py ...
python3 tools/diagnostics/check_responses_stream.py ...
python3 tools/proxies/codex_responses_chat_proxy.py ...
python3 tools/proxies/tcp_proxy_forwarder.py ...
```

The lightweight `codex_responses_chat_proxy.py` can bridge text-only Responses-to-chat cases, but MoonBridge is the preferred path for Codex tool/action trajectories.
