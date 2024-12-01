from time import sleep
import anthropic
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, cast

from ner.clients.llm_client import LLMClient


class ClaudeFamily(Enum):
    HAIKU = "claude-3-haiku-20240307"
    SONNET_35 = "claude-3-5-sonnet-20240620"
    SONNET_35_V2 = "claude-3-5-sonnet-20241022"
    OPUS = "claude-3-opus-20240229"
    HAIKU_35 = "claude-3-5-haiku-20241022"


MAX_RETRIES = 5


@dataclass
class AnthropicClient(LLMClient):
    model_name: ClaudeFamily

    def get_llm_response(
        self, query: str, system_prompt: str = "", functions: List[Any] = []
    ) -> str:
        client = anthropic.Anthropic(max_retries=MAX_RETRIES)
        retries_left = MAX_RETRIES

        message = None
        while retries_left > 0:
            try:
                claude_response = client.messages.create(
                    model=self.model_name.value,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": query}],
                    temperature=0,
                )
                message = cast(
                    anthropic.types.TextBlock, claude_response.content[0]
                ).text
                break
            except Exception as err:
                print(
                    f"Something went wrong while calling Claude: {str(err)}. Retrying."
                )
                sleep(1)
                retries_left -= 1

        if not message:
            raise Exception(
                "Error while calling Claude. Retries 5 times with no success."
            )

        return message


if __name__ == "__main__":
    client = AnthropicClient(ClaudeFamily.HAIKU)
    print(client.get_llm_response(query="Hey, how are you?"))
