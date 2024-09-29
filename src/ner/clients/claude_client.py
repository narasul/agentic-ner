from dataclasses import dataclass
from enum import Enum
from typing import Any, List
import anthropic

from ner.clients.llm_client import LLMClient


class ClaudeFamily(Enum):
    HAIKU = "claude-3-haiku-20240307"
    SONNET_35 = "claude-3-5-sonnet-20240620"
    OPUS = "claude-3-opus-20240229"


@dataclass
class AnthropicClient(LLMClient):
    model_name: ClaudeFamily

    def get_llm_response(
        self, query: str, system_prompt: str = "", functions: List[Any] = []
    ) -> str:
        client = anthropic.Anthropic()

        message = client.messages.create(
            model=self.model_name.value,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
        )

        return str(message.content)


if __name__ == "__main__":
    client = AnthropicClient(ClaudeFamily.HAIKU)
    print(client.get_llm_response(query="Hey, how are you?"))
