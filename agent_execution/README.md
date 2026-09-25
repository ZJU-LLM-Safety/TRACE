# agent_execution

Reusable execution interfaces for trace experiments.

This package contains method-agnostic building blocks for running an agent target in the prepared execution environment. It is intended to be reused by decomposition-based methods, and other attack/evaluation pipelines.

## Modules

- `bridge.py`: shared Codex/VRAP bridge defaults, retry handling, dataset selection imports, and container helpers from `trace/tools`.
- `targets.py`: Codex target wrapper, session resume support, Codex home snapshots, execution-action replay, post-subtask execution turn, and generic candidate promotion targets.
- `sequence.py`: link-mode helpers and `MultiSubtaskRunner` for multi-subtask runs, including session-state extraction and action-summary prefix construction.
- `target_prompting.py`: optional execution-oriented target prompt wrapping.
- `prepare_verify_trajectories.py`: converts saved same-session execution results into VRAP verify trajectories.
- `verify_tolerant.py`: timeout-tolerant verify CLI. It delegates action parsing/replay to `trace/tools/vrap_action_replay.py` and only owns tolerant verify orchestration.
- `scripts/codex_execute_item_subtasks_same_session.py`: CLI for feeding saved method outputs back to Codex or Claude in one session/container per item. It runs live verification by default after all subtasks finish; pass `--no-live-verify` to skip that probe.

## Multi-Subtask Runner

`MultiSubtaskRunner` implements the method-agnostic subtask loop. It handles:

- `session` link mode by carrying formal session state into the next subtask;
- `action_summary` link mode by summarizing prior execution actions and passing a prefix to the next subtask;
- per-subtask result collection;
- action aggregation;
- temporary Codex-home snapshot cleanup.

A concrete method should subclass it and implement:

```python
class MyMethodRunner(MultiSubtaskRunner):
    def run_subtask(self, *, subtask_index, subtask, sub_args, sub_config):
        ...

    def build_run_metadata(self):
        ...
```

The baseline adapter does exactly this in `xteaming_vrap.sequence.XTeamingSubtaskRunner`, where `run_subtask()` calls `run_xteaming_subtask(...)`.

## Promotion vs. TextGrad

State promotion is independent of TextGrad. The reusable concept is:

1. restore the current formal session/container state;
2. run a candidate input in an isolated trial;
3. snapshot the resulting Codex home/session state and collect actions;
4. let the caller decide which candidate wins;
5. promote the winning candidate state as the new formal state.

TextGrad is only one possible way to generate candidate inputs. Future optimizers can reuse `ReplayTrialTarget` or `SnapshotPromotedTrialTarget` without depending on TextGrad. Backward-compatible aliases `IsolatedTextGradTrialTarget` and `SnapshotPromotedTextGradTrialTarget` remain for the existing baseline code.

## Live Verify And Replay Verify

The same-session execution script verifies the live container state by default. This avoids depending on action replay when an agent used an interactive shell and the saved action only contains the outer command.

Replay verify is still available for checking whether saved `actions` are independently reproducible in a fresh environment. Use `prepare_verify_trajectories.py` followed by `verify_tolerant.py` for that path.

## Action Replay De-Duplication

Action parsing and replay are centralized in:

```text
tools/vrap_action_replay.py
```

Callers now share that implementation:

- `trace/tools/vrap_bridge_verify.py` uses `replay_trajectory_actions(..., tolerant_timeouts=False)`;
- `trace/agent_execution/verify_tolerant.py` uses `replay_trajectory_actions(..., tolerant_timeouts=True)`;
- `trace/agent_execution/targets.py` uses `replay_state_action(s)` for session/snapshot promotion restore.

This keeps Docker command execution and action shape handling in `trace/tools`, while `agent_execution` keeps method-agnostic orchestration such as session state, promotion, and tolerant verify output shaping.

## Compatibility

The self-contained baseline copy under `run_xteaming/` imports this package through the trace root. New methods can import directly from this package with:

```bash
PYTHONPATH=/path/to/trace python ...
```

Retry environment variables use the generic names:

- `AGENT_EXECUTION_CODEX_NETWORK_RETRIES`
- `AGENT_EXECUTION_CODEX_SAFETY_RETRIES`

The previous `XTEAMING_CODEX_NETWORK_RETRIES` and `XTEAMING_CODEX_SAFETY_RETRIES` names are still honored for compatibility.
