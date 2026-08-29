import json
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from tenacity import retry, stop_after_attempt, wait_fixed

from agents.base_agent import BaseAgent
from agents.gpt_evaluator import GPTJudge
from agents.target_model import TargetModel, truncate_response


class LLMRefusalError(Exception):
    pass


class AttackerAgent(BaseAgent):
    """Agent that turns a baseline-schema strategy into multi-turn attacker messages."""

    REQUIRED_STRATEGY_FIELDS = {
        "persona",
        "context",
        "approach",
        "turns_needed",
        "conversation_plan",
    }

    def __init__(
        self,
        config: Dict[str, Any],
        behavior: Dict[str, Any],
        strategy: Dict[str, Any],
        prompts_path: str | Path | None = None,
    ):
        super().__init__(config)
        self.messages: list[dict[str, str]] = []
        self.prompts_path = Path(prompts_path) if prompts_path else default_prompts_path()
        self._initialize_agent(behavior, strategy)

    def _initialize_agent(self, behavior: Dict[str, Any], strategy: Dict[str, Any]) -> None:
        missing = self.REQUIRED_STRATEGY_FIELDS - set(strategy)
        if missing:
            raise ValueError(
                "Experiment strategies must use the baseline schema; missing fields: "
                + ", ".join(sorted(missing))
            )

        self.behavior = str(behavior["Behavior"])
        self.strategy_dict = strategy
        self.strategy = self._format_strategy(strategy)
        self.phase_keys = ordered_conversation_plan_keys(strategy["conversation_plan"])
        self.plan_phase = 1
        self.num_phases = len(self.phase_keys)
        self.plan_score_history: list[int] = []
        self.next_message_candidates: list[dict[str, Any]] = []
        self.generated_records: list[dict[str, Any]] = []
        self._load_prompt_templates()

    def _format_strategy(self, strategy: Dict[str, Any]) -> str:
        return "\n".join(
            [
                f"Persona: {strategy['persona']}",
                f"Context: {strategy['context']}",
                f"Approach: {strategy['approach']}",
                f"Turns Needed: {strategy['turns_needed']}",
                "Conversation Plan: "
                + json.dumps(
                    strategy["conversation_plan"],
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )

    def _load_prompt_templates(self) -> dict[str, Any]:
        prompts = yaml.safe_load(self.prompts_path.read_text(encoding="utf-8"))["prompts"]
        self.system_prompt = prompts["system"]["messages"][0]["content"]
        self.first_turn_template = prompts["first_turn_messages"]["messages"][0]["content"]
        self.nth_turn_template = prompts["nth_turn_messages"]["messages"][0]["content"]
        self.final_turn_template = prompts["final_turn_messages"]["messages"][0]["content"]
        return prompts

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def generate_response(
        self,
        conversation_history: str | None = None,
        turn_number: int | None = None,
    ) -> str:
        actual_turn_number = turn_number or self.plan_phase
        if self.plan_phase == 1:
            response = self._generate_first_turn(actual_turn_number)
        elif self.plan_phase == self.num_phases:
            response = self._generate_final_turn(
                conversation_history or "",
                actual_turn_number,
            )
        else:
            response = self._generate_nth_turn(
                conversation_history or "",
                actual_turn_number,
            )

        if "can't assist with that" in response.lower():
            raise LLMRefusalError(response)
        return response

    def commit(self) -> None:
        if not self.next_message_candidates:
            raise ValueError("A response has not been generated yet.")

        best_candidate = self.next_message_candidates[0]
        for candidate in self.next_message_candidates[1:]:
            if candidate["score"] > best_candidate["score"]:
                best_candidate = candidate

        self.messages.extend(best_candidate["messages"])
        self.plan_score_history.append(best_candidate["score"])
        self.next_message_candidates.clear()
        if self.plan_phase < self.num_phases:
            self.plan_phase += 1

    def done(self) -> bool:
        return self.plan_phase >= self.num_phases and bool(self.generated_records)

    def generate_and_evaluate_candidate(
        self,
        target: TargetModel,
        evaluator: GPTJudge,
        conversation_history: str | None = None,
        turn_number: int | None = None,
    ) -> dict[str, Any]:
        actual_turn_number = turn_number or self.plan_phase
        attack_generation = self.generate_response(
            conversation_history,
            turn_number=actual_turn_number,
        )
        target_response = target.generate_response(attack_generation)
        truncated_response = truncate_response(target_response)
        score, reason = evaluator.infer_single(self.behavior, truncated_response)
        self.next_message_candidates[-1]["score"] = score
        self.generated_records[-1]["target_response"] = target_response
        self.generated_records[-1]["target_truncated"] = truncated_response
        self.generated_records[-1]["evaluation_score"] = score
        self.generated_records[-1]["evaluation_reason"] = reason

        return {
            "turn_number": actual_turn_number,
            "phase": self.plan_phase,
            "phase_key": self.phase_keys[self.plan_phase - 1],
            "attacker": attack_generation,
            "target": target_response,
            "target_truncated": truncated_response,
            "evaluation_score": score,
            "evaluation_reason": reason,
        }

    def _current_flow(self) -> str:
        policy = self.strategy_dict["conversation_plan"]
        return str(policy[self.phase_keys[self.plan_phase - 1]])

    def _system_message(self) -> dict[str, str]:
        return {
            "role": "system",
            "content": self.system_prompt.format(
                target_behavior=self.behavior,
                strategy=self.strategy,
                max_turns=self.config.get("max_turns", self.num_phases),
            ),
        }

    def _record_generation(
        self,
        *,
        turn_number: int,
        phase_key: str,
        prompt_messages: list[dict[str, str]],
        raw_response: str,
        conversation: str,
    ) -> str:
        self.generated_records.append(
            {
                "turn_number": turn_number,
                "phase_number": self.plan_phase,
                "phase_key": phase_key,
                "phase": self._current_flow(),
                "prompt_messages": prompt_messages,
                "raw_response": raw_response,
                "attacker_message": conversation,
                "target_response": None,
            }
        )
        return conversation

    def _generate_first_turn(self, turn_number: int) -> str:
        phase_key = self.phase_keys[self.plan_phase - 1]
        prompt_messages = [
            self._system_message(),
            {
                "role": "user",
                "content": self.first_turn_template.format(
                    turn_1_conversation_flow=self._current_flow()
                ),
            },
        ]
        response = self.call_api(prompt_messages, temperature=self.config["temperature"])
        prompt_messages.append({"role": "assistant", "content": response})
        self.next_message_candidates.append({"score": 0, "messages": prompt_messages})
        return self._record_generation(
            turn_number=turn_number,
            phase_key=phase_key,
            prompt_messages=prompt_messages,
            raw_response=response,
            conversation=extract_conversation(response),
        )

    def _generate_nth_turn(self, conversation_history: str, turn_number: int) -> str:
        phase_key = self.phase_keys[self.plan_phase - 1]
        prompt_messages = [
            self._system_message(),
            {
                "role": "user",
                "content": self.nth_turn_template.format(
                    turn_number=turn_number,
                    conversation_history=conversation_history,
                    turn_number_conversation_flow=self._current_flow(),
                    previous_turn_number=max(turn_number - 1, 1),
                ),
            },
        ]
        response = self.call_api(prompt_messages, temperature=self.config["temperature"])
        prompt_messages.append({"role": "assistant", "content": response})
        self.next_message_candidates.append({"score": 0, "messages": prompt_messages})
        return self._record_generation(
            turn_number=turn_number,
            phase_key=phase_key,
            prompt_messages=prompt_messages,
            raw_response=response,
            conversation=extract_conversation(response),
        )

    def _generate_final_turn(self, conversation_history: str, turn_number: int) -> str:
        phase_key = self.phase_keys[self.plan_phase - 1]
        prompt_messages = [
            self._system_message(),
            {
                "role": "user",
                "content": self.final_turn_template.format(
                    conversation_history=conversation_history,
                    final_turn_conversation_flow=self._current_flow(),
                ),
            },
        ]
        response = self.call_api(prompt_messages, temperature=self.config["temperature"])
        prompt_messages.append({"role": "assistant", "content": response})
        self.next_message_candidates.append({"score": 0, "messages": prompt_messages})
        return self._record_generation(
            turn_number=turn_number,
            phase_key=phase_key,
            prompt_messages=prompt_messages,
            raw_response=response,
            conversation=extract_conversation(response),
        )


def default_prompts_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "prompts" / "attacker_agent_prompts.yaml"


def ordered_conversation_plan_keys(conversation_plan: dict[str, Any]) -> list[str]:
    if not isinstance(conversation_plan, dict):
        raise ValueError("conversation_plan must be a dictionary")
    if "turn_1" not in conversation_plan:
        raise ValueError("conversation_plan must include turn_1")
    if "final_turn" not in conversation_plan:
        raise ValueError("conversation_plan must include final_turn")

    numbered = [
        key
        for key in conversation_plan
        if re.fullmatch(r"turn_\d+", str(key))
    ]
    numbered.sort(key=lambda key: int(str(key).rsplit("_", 1)[1]))
    return numbered + ["final_turn"]


def extract_conversation(response: str) -> str:
    start_tag = "<conversation>"
    end_tag = "</conversation>"
    start = response.find(start_tag)
    end = response.find(end_tag)
    if start == -1 or end == -1 or end <= start:
        return response.strip()
    return response[start + len(start_tag) : end].strip()
