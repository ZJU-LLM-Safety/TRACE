from __future__ import annotations

from typing import Dict

import tiktoken

from agents.base_agent import BaseAgent


class TargetModel(BaseAgent):
    """Target model that responds to attacker's messages."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.messages = []

    def generate_response(self, attacker_message: str) -> str:
        if not self.messages:
            self.messages = [{"role": "user", "content": attacker_message}]
        else:
            self.messages.append({"role": "user", "content": attacker_message})

        response = self.call_api(
            messages=self.messages,
            temperature=self.config["temperature"],
        )
        self.messages.append({"role": "assistant", "content": response})
        return response


def truncate_response(response_text: str, max_tokens: int = 512) -> str:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        tokens = encoding.encode(response_text)
        if len(tokens) <= max_tokens:
            return response_text
        return encoding.decode(tokens[:max_tokens])
    except Exception as exc:
        print(f"Warning: Error in token counting: {exc}")
        return response_text
