import copy
import json

from autogen_core.base import MessageContext
from autogen_core.components import DefaultTopicId, FunctionCall, message_handler
from autogen_core.components.models import ChatCompletionClient, UserMessage
from autogen_core.components.tools import FunctionTool
from rich.console import Console
from rich.markdown import Markdown

from ner.agents.base_agent import BaseGroupChatAgent, GroupChatMessage, RequestToSpeak
from ner.agents.tools.search import search_with_tavily


RESEARCHER_TOPIC_TYPE = "Researcher"


class ResearchAgent(BaseGroupChatAgent):
    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
        system_prompt: str,
    ) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message=system_prompt,
        )

        self._search_tool = FunctionTool(
            search_with_tavily,
            name="search",
            description="Use this tool to search anything",
        )

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:  # type: ignore
        Console().print(Markdown(f"### {self.id.type}: "))

        self._chat_history.append(
            UserMessage(
                content=f"You are requested to answer the questions inside <search> tags in previous message. If you have received a response from 'search' tool already answer the questions using the response from 'search' tool. Don't respond to anything outside <search> tags, it is not your job.",
                source="system",
            )
        )
        completion = await self._model_client.create(
            [self._system_message] + self._chat_history,
            tools=[self._search_tool],
            extra_create_args={"tool_choice": "required", "temperature": 0},
            cancellation_token=ctx.cancellation_token,
        )

        search_tool_response = ""
        for item in completion.content:
            if isinstance(item, FunctionCall):
                arguments = json.loads(item.arguments)
                Console().print(arguments)
                result = await self._search_tool.run_json(
                    arguments, ctx.cancellation_token
                )
                search_tool_response += result + "\n"

        search_tool_response = "SEARCH_TOOL_RESPONSE: \n" + search_tool_response

        self._chat_history.append(
            UserMessage(content=search_tool_response, source="system")
        )
        chat_history_copy = copy.deepcopy(self._chat_history[-3:])
        completion = await self._model_client.create(
            [self._system_message] + chat_history_copy,
            extra_create_args={"temperature": 0},
        )
        print(f"Search tool response: {search_tool_response}")
        await self.publish_message(
            GroupChatMessage(
                body=UserMessage(content=f"<answer>{completion.content}</answer>", source=self.id.type)  # type: ignore
            ),
            DefaultTopicId(self._group_chat_topic_type),
        )
        print(f"Researcher response: {completion.content}")
