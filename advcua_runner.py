import random
import sys

def run_advcua_once_without_score(
    datum,
    field,
    model_name,
    seed=44,
    max_attempts=1,
    provider="openai",
    last_subtask=False,
    debug=True,
    reserve_all_messages=False,
):
    import numpy as np
    import torch

    try:
        from VRAP.Agent import AutoGPT_OSAgent
        from VRAP.utils import log_full_terminal_output, log_terminal_interaction
    except ImportError as exc:
        raise ImportError(
            "Cannot import AdvCUA runtime modules from local repo. "
            "Expected: VRAP/Agent and VRAP/utils.py under repository root."
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = AutoGPT_OSAgent(
        model_name=model_name,
        provider_name=provider,
        debug=debug,
        reserve_all_messages=reserve_all_messages,
    )

    log_terminal_interaction("Running 1 attempt. Mark as successful upon a single successful attempt.")
    attempts_results = []

    for attempt in range(1, max_attempts + 1):
        current_seed = seed + attempt * 1000
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)

        log_terminal_interaction(f"Attempt {attempt}/{max_attempts} (seed: {current_seed})")

        try:
            if hasattr(agent, "reset_memory"):
                agent.reset_memory()
                log_terminal_interaction("Agent memory resetted")

            attacker_identity = datum["attacker_identity"]
            environment_init_code = datum.get("environment_init_code")
            conversation_history = datum["history"]
            selected_prompt = datum[field]

            log_terminal_interaction(f"Attack goal: {selected_prompt}")
            container, server, victim = agent.setup_docker_environment(datum, attacker_identity)
            log_terminal_interaction("Docker environment setup completed")
            log_terminal_interaction(f"Run {agent.__class__.__name__}, provider: {provider}, model: {agent.model_name}")

            import io

            class TeeOutput:
                def __init__(self, original_stdout):
                    self.original_stdout = original_stdout
                    self.buffer = io.StringIO()

                def write(self, text):
                    self.original_stdout.write(text)
                    self.buffer.write(text)
                    return len(text)

                def flush(self):
                    self.original_stdout.flush()
                    self.buffer.flush()

                def getvalue(self):
                    return self.buffer.getvalue()

            original_stdout = sys.stdout
            tee_output = TeeOutput(original_stdout)
            sys.stdout = tee_output

            try:
                history, messages = agent.run(
                    container=container,
                    attacker_identity=attacker_identity,
                    environment_init_code=environment_init_code,
                    item=datum,
                    prompt_to_os_agent=selected_prompt,
                    conversation_history=conversation_history,
                )
                datum["messages"] = messages
                agent_interaction_log = tee_output.getvalue()
                log_full_terminal_output(agent_interaction_log)
            finally:
                sys.stdout = original_stdout

            token_usage = (
                agent.get_attempt_token_usage()
                if hasattr(agent, "get_attempt_token_usage")
                else {"input_token": 0, "output_token": 0, "total_tokens": 0}
            )
            datum["usage"] = token_usage

            if history == "TIMEOUT":
                log_terminal_interaction(f"Attempt {attempt} timed out")
                attempt_result = {
                    "attempt": attempt,
                    "result": False,
                    "reason": "timeout",
                    "agent_interaction_log": agent_interaction_log,
                }
                _ = attempt_result
                datum["messages"] = []
                return

        except Exception as e:
            log_terminal_interaction(f"Attempt {attempt} error: {e}")
            token_usage = (
                agent.get_attempt_token_usage()
                if hasattr(agent, "get_attempt_token_usage")
                else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            attempt_result = {
                "attempt": attempt,
                "result": False,
                "reason": f"error: {e}",
                "agent_interaction_log": "",
                "token_usage": token_usage,
            }
            attempts_results.append(attempt_result)
            datum["messages"] = messages if "messages" in locals() else []

        finally:
            try:
                agent.cleanup_docker_environment()
                log_terminal_interaction("Environment cleanup completed")
            except Exception as e:
                log_terminal_interaction(f"Cleanup warning: {e}")
