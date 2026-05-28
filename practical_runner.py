from __future__ import annotations

import datetime
import json
import subprocess
import tempfile
from dataclasses import dataclass
from getpass import getuser
from pathlib import Path
from typing import Any

from unidiff import PatchSet

from practical_deps.sweagent import CONFIG_DIR
from practical_deps.sweagent.agent.agents import Agent, AgentArguments
from practical_deps.sweagent.agent.models import ModelArguments
from practical_deps.sweagent.environment.swe_env import EnvironmentArguments, SWEEnv
from practical_deps.sweagent.utils.log import add_file_handler, get_logger

import logging

logger = logging.getLogger("execute_step")

@dataclass(frozen=True)
class ActionsArguments:
    open_pr: bool = False
    apply_patch_locally: bool = False
    skip_if_commits_reference_issue: bool = True


class SaveApplyPatchHook:
    def on_init(self, *, args, agent, env, traj_dir: Path):
        self._traj_dir = traj_dir
        self._apply_patch_locally = args.actions.apply_patch_locally
        self._instance = None

    def on_instance_start(self, *, index: int, instance: dict[str, Any]):
        self._instance = instance

    def on_instance_completed(self, *, info, trajectory):
        assert self._instance is not None
        instance_id = self._instance["instance_id"]
        patch_path = self._save_patch(instance_id, info)
        if not patch_path or not self._apply_patch_locally:
            return
        if info.get("exit_status") != "submitted" or info.get("submission") is None:
            return
        if self._instance["repo_type"] != "local":
            return
        self._apply_patch(patch_path, Path(self._instance["repo"]))

    def _save_patch(self, instance_id: str, info) -> Path | None:
        patch_output_dir = self._traj_dir / "patches"
        patch_output_dir.mkdir(exist_ok=True, parents=True)
        patch_output_file = patch_output_dir / f"{instance_id}.patch"
        if info.get("submission") is None:
            logger.info("No patch to save.")
            return None
        patch_output_file.write_text(info["submission"])
        return patch_output_file

    def _apply_patch(self, patch_file: Path, local_dir: Path) -> None:
        cmd = ["git", "apply", str(patch_file.resolve())]
        try:
            subprocess.run(cmd, cwd=local_dir, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply patch {patch_file} to {local_dir}: {e}")


@dataclass(frozen=True)
class ScriptArguments:
    environment: EnvironmentArguments
    agent: AgentArguments
    actions: ActionsArguments
    suffix: str = ""
    skip_existing: bool = False
    subtask_input: dict[str, Any] | None = None
    replay_subtask_history: bool = True
    print_config: bool = False


def run_ctf_once(datum, field, target_model, harmful=True):
    env = None
    if field not in datum:
        raise ValueError(f"Missing field '{field}' in datum")
    subtask_prompt = datum[field]
    raw_history = datum.get("history", []) or []
    if not isinstance(raw_history, list):
        raise ValueError("subtask history must be a list of trajectory steps")

    required_step_keys = {"action", "observation"}
    for idx, step in enumerate(raw_history):
        if not isinstance(step, dict):
            raise ValueError(f"subtask history step {idx} must be a dict")
        missing = required_step_keys - set(step.keys())
        if missing:
            raise ValueError(f"subtask history step {idx} missing keys: {sorted(missing)}")
    subtask_history = raw_history

    def build_subtask_setup_args(base_setup_args: dict[str, Any]) -> dict[str, Any]:
        setup_args = dict(base_setup_args)
        setup_args["subtask_prompt"] = subtask_prompt
        setup_args["subtask_history"] = subtask_history
        return setup_args

    config_file = CONFIG_DIR / "test.yaml"
    if not config_file.exists():
        config_file = CONFIG_DIR / "default_ctf_test.yaml"
    image_name = datum.get("image_name", "attackenv:latest")
    install_env = datum.get("install_environment", False)
    cache_task_images = datum.get("cache_task_images", False)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
        datum["harmful"] = datum[field]
        json.dump([datum], fp)
        temp_data_path = fp.name

    try:
        env_args = EnvironmentArguments(
            image_name=image_name,
            data_path=temp_data_path,
            split=datum.get("split", "dev"),
            verbose=True,
            install_environment=install_env,
            cache_task_images=cache_task_images,
            harmful=harmful,
        )
        agent_args = AgentArguments(
            model=ModelArguments(
                model_name=target_model,
                total_cost_limit=0.0,
                per_instance_cost_limit=3.0,
                temperature=0.0,
                top_p=0.95,
            ),
            config_file=config_file,
        )
        args = ScriptArguments(
            suffix="",
            environment=env_args,
            agent=agent_args,
            actions=ActionsArguments(open_pr=False, skip_if_commits_reference_issue=True),
            skip_existing=False,
            subtask_input={"subtask_prompt": subtask_prompt, "history": subtask_history},
            replay_subtask_history=True,
            print_config=False,
        )

        traj_dir = Path("trajectories") / Path(getuser()) / "agent_attack"
        traj_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        add_file_handler(traj_dir / f"run-{timestamp}.log")

        env = SWEEnv(args.environment)
        agent = Agent("primary", args.agent)
        hooks = [SaveApplyPatchHook()]
        for hook in hooks:
            hook.on_init(args=args, agent=agent, env=env, traj_dir=traj_dir)
            hook.on_instance_start(index=0, instance=env.data[0])

        observation, info = env.reset(0)
        if info is None:
            raise RuntimeError("Failed to reset environment")

        files = []
        assert env.record is not None
        if "patch" in env.record:
            files = "\n".join([f"- {x.path}" for x in PatchSet(env.record["patch"]).modified_files])
        test_files = []
        if "test_patch" in env.record:
            test_patch_obj = PatchSet(env.record["test_patch"])
            test_files = "\n".join([f"- {x.path}" for x in test_patch_obj.modified_files + test_patch_obj.added_files])
        tests = ""
        if "FAIL_endTO_PASS" in env.record:
            tests = "\n".join([f"- {x}" for x in env.record["FAIL_TO_PASS"]])

        setup_args = {"files": files, "test_files": test_files, "tests": tests}
        challenge = env.challenge
        if challenge is not None:
            setup_args["name"] = challenge["challenge"]
            setup_args["description"] = challenge["harmful"] if harmful else challenge["query"]
            setup_args["category_friendly"] = challenge["category"]
            setup_args["files"] = challenge["files"] or "No files included in this challenge."
            setup_args["box"] = challenge.get("box")
            setup_args["port"] = challenge.get("port")
            setup_args["server_description"] = challenge.get("server_description")
            if "flag_format" in challenge:
                setup_args["flag_format"] = challenge["flag_format"]

        if args.subtask_input:
            if args.replay_subtask_history and subtask_history:
                agent.init_environment_vars(env)
                agent.setup(setup_args, None)
                agent.config.summarizer_config.function.setup(setup_args, agent.config)
                replay_actions = [step["action"] for step in subtask_history]
                observation, info = agent.replay_actions(env, replay_actions)
            setup_args = build_subtask_setup_args(setup_args)

        trajectory = None
        if info.get("exit_status") != "submitted" or info.get("submission") is None:
            info, trajectory = agent.run(
                setup_args=setup_args,
                env=env,
                observation=observation,
                traj_dir=traj_dir,
                return_type="info_trajectory",
            )
        for hook in hooks:
            hook.on_instance_completed(info=info, trajectory=trajectory)
        env.reset_container()
        return info, trajectory
    finally:
        if env is not None:
            env.close()
        Path(temp_data_path).unlink(missing_ok=True)
