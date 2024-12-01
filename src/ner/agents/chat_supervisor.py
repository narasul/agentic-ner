from typing import List, Any, Dict

from autogen_core.base import MessageContext
from autogen_core.components import (
    DefaultTopicId,
    RoutedAgent,
    message_handler,
)
from autogen_core.components.models import (
    ChatCompletionClient,
    UserMessage,
)
from ner.agents.base_agent import GroupChatMessage, RequestToSpeak
from ner.agents.research_agent import RESEARCHER_TOPIC_TYPE
from ner.agents.reviewer_agent import REVIEWER_TOPIC_TYPE
from ner.agents.tagger_agent import TAGGER_TOPIC_TYPE


MAX_AGENT_TURNS = 10


class ChatSupervisor(RoutedAgent):
    def __init__(
        self,
        participant_topic_types: List[str],
        model_client: ChatCompletionClient,
        participant_descriptions: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        super().__init__("Group chat manager")
        self._participant_topic_types = participant_topic_types
        self._model_client = model_client
        self._chat_history: List[UserMessage] = []
        self._participant_descriptions = participant_descriptions
        self._previous_participant_topic_type: str | None = None
        self._metadata = metadata
        self._num_agent_turns = 0

    @message_handler
    async def handle_message(
        self, message: GroupChatMessage, ctx: MessageContext
    ) -> None:
        assert isinstance(message.body, UserMessage)
        self._chat_history.append(message.body)
        self._num_agent_turns += 1

        if self._num_agent_turns > MAX_AGENT_TURNS:
            print(f"Too many agent invocations!")
            return

        if message.body.source == REVIEWER_TOPIC_TYPE:
            await self.handle_reviewer_message(message)
        elif message.body.source == RESEARCHER_TOPIC_TYPE:
            await self.handle_researcher_message(message)
        elif message.body.source == "User" or message.body.source == "system":
            await self.handle_user_message(message)
        else:
            await self.handle_tagger_message(message)

    async def handle_reviewer_message(self, message: GroupChatMessage):
        print(f"Handling reviewer message. Content: {message.body.content}")
        selected_topic_type = ""
        # If the message is an approval message from the reviewer, stop the chat.
        assert isinstance(message.body.content, str)
        if "<search>" in message.body.content:
            selected_topic_type = RESEARCHER_TOPIC_TYPE
            await self.publish_message(
                RequestToSpeak(), DefaultTopicId(type=selected_topic_type)
            )
            self._previous_participant_topic_type = REVIEWER_TOPIC_TYPE
            return
        if "<feedback>" in message.body.content:
            selected_topic_type = TAGGER_TOPIC_TYPE
            self._chat_history.append(
                UserMessage(
                    source="User",
                    content=self._metadata.get("query", ""),
                )
            )
        elif message.body.content.endswith("APPROVED"):
            return

        await self.publish_message(
            RequestToSpeak(), DefaultTopicId(type=selected_topic_type)
        )
        self._previous_participant_topic_type = selected_topic_type

    async def handle_researcher_message(self, message: GroupChatMessage):
        print(
            f"Handling researcher response. Requesting {self._previous_participant_topic_type} to speak"
        )
        await self.publish_message(
            RequestToSpeak(),
            DefaultTopicId(type=self._previous_participant_topic_type or ""),
        )

    async def handle_user_message(self, message: GroupChatMessage):
        self._previous_participant_topic_type = TAGGER_TOPIC_TYPE
        await self.publish_message(
            RequestToSpeak(), DefaultTopicId(type=TAGGER_TOPIC_TYPE)
        )

    async def handle_tagger_message(self, message: GroupChatMessage):
        selected_topic_type = ""
        print(f"Handling tagger message. Content: {message.body.content}")
        if "<output>" in message.body.content or "<objection>" in message.body.content:
            selected_topic_type = REVIEWER_TOPIC_TYPE
            print(f"Updating tagger output: {message.body.content}")
            self._metadata["last_tagger_output"] = message.body.content  # type: ignore
        elif "<search>" in message.body.content:
            selected_topic_type = RESEARCHER_TOPIC_TYPE
            await self.publish_message(
                RequestToSpeak(), DefaultTopicId(type=RESEARCHER_TOPIC_TYPE)
            )
            self._previous_participant_topic_type = TAGGER_TOPIC_TYPE
            return
        else:
            selected_topic_type = TAGGER_TOPIC_TYPE
            self._chat_history.append(
                UserMessage(
                    source="User",
                    content="You should put your final output inside <output> tags!",
                )
            )

        self._previous_participant_topic_type = selected_topic_type

        await self.publish_message(
            RequestToSpeak(), DefaultTopicId(type=selected_topic_type)
        )