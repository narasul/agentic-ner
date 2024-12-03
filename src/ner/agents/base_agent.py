from typing import List

from autogen_core.base import MessageContext
from autogen_core.components import (
    DefaultTopicId,
    RoutedAgent,
    message_handler,
)
from autogen_core.components.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown


class GroupChatMessage(BaseModel):
    body: UserMessage


class RequestToSpeak(BaseModel):
    pass


class BaseGroupChatAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
        system_message: str,
    ) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type
        self._model_client = model_client
        self._system_message = SystemMessage(system_message)
        self._chat_history: List[LLMMessage] = []

    @message_handler
    async def handle_message(
        self, message: GroupChatMessage, ctx: MessageContext
    ) -> None:
        self._chat_history.extend(
            [
                UserMessage(
                    content=f"Transferred to {message.body.source}", source="system"
                ),
                message.body,
            ]
        )

    @message_handler
    async def handle_request_to_speak(
        self, message: RequestToSpeak, ctx: MessageContext
    ) -> None:
        Console().print(Markdown(f"### {self.id.type}: "))
        # self._chat_history.append(
        #     UserMessage(
        #         content=f"Transferred to {self.id.type}, adopt the persona immediately and fulfill the request.",
        #         source="system",
        #     )
        # )
        print(f"Taking into account last message:\n {self._chat_history[-1].content}")
        completion = await self._model_client.create(
            [self._system_message] + self._chat_history,
            extra_create_args={"temperature": 0},
        )
        assert isinstance(completion.content, str)
        self._chat_history.append(
            AssistantMessage(content=completion.content, source=self.id.type)
        )
        # Console().print(Markdown(completion.content))

        await self.publish_message(
            GroupChatMessage(
                body=UserMessage(content=completion.content, source=self.id.type)
            ),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type),
        )
