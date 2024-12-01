import asyncio
from autogen_core.components.models import (
    ChatCompletionClient,
    UserMessage,
)
from autogen_ext.models import OpenAIChatCompletionClient

from ner.clients.claude_client import ClaudeFamily


def create_chat_completions_client(model_name: str) -> ChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=model_name,
        model_capabilities={
            "function_calling": True,
            "vision": False,
            "json_output": False,
        },
    )


if __name__ == "__main__":

    async def joke():
        client = create_chat_completions_client(ClaudeFamily.SONNET_35_V2.value)
        response = await client.create(
            messages=[UserMessage(content="Tell me a joke", source="Rasul")]
        )
        print(response)

    asyncio.run(joke())
