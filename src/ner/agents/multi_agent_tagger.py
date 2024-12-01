import asyncio
from dataclasses import dataclass
from typing import List, Optional, Tuple
import uuid

from autogen_core.base import TopicId
from autogen_core.components.models import ChatCompletionClient, UserMessage

from autogen_core.application import SingleThreadedAgentRuntime
from autogen_core.components import TypeSubscription

from ner.agents.base_agent import GroupChatMessage
from ner.agents.tagger_agent import TAGGER_TOPIC_TYPE, TaggerAgent
from ner.agents.reviewer_agent import REVIEWER_TOPIC_TYPE, ReviewerAgent
from ner.agents.research_agent import RESEARCHER_TOPIC_TYPE, ResearchAgent
from ner.agents.chat_supervisor import ChatSupervisor
from ner.agents.agent_config import AgentConfig
from ner.converter import Converter
from ner.grounding import GroundingEngine
from ner.tagger import Tagger


@dataclass
class MultiAgentTagger(Tagger):
    agent_config: AgentConfig
    llm_client: ChatCompletionClient
    grounding_engine: Optional[GroundingEngine] = None
    group_chat_topic_type = "GroupChat"

    async def initialize_agents(self) -> SingleThreadedAgentRuntime:
        self.runtime = SingleThreadedAgentRuntime()

        research_topic_type = RESEARCHER_TOPIC_TYPE
        tagger_topic_type = TAGGER_TOPIC_TYPE
        reviewer_topic_type = REVIEWER_TOPIC_TYPE

        tagger_agent_type = await TaggerAgent.register(
            self.runtime,
            tagger_topic_type,
            lambda: TaggerAgent(
                description=self.agent_config.tagger_description,
                group_chat_topic_type=self.group_chat_topic_type,
                model_client=self.llm_client,
                system_prompt=self.agent_config.tagger_system_prompt,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(
                topic_type=tagger_topic_type, agent_type=tagger_agent_type.type
            )
        )
        await self.runtime.add_subscription(
            TypeSubscription(
                topic_type=self.group_chat_topic_type, agent_type=tagger_agent_type.type
            )
        )

        reviewer_agent_type = await ReviewerAgent.register(
            self.runtime,
            reviewer_topic_type,
            lambda: ReviewerAgent(
                description=self.agent_config.reviewer_description,
                group_chat_topic_type=self.group_chat_topic_type,
                model_client=self.llm_client,
                system_prompt=self.agent_config.reviewer_system_prompt,
            ),
        )
        await self.runtime.add_subscription(
            TypeSubscription(
                topic_type=reviewer_topic_type, agent_type=reviewer_agent_type.type
            )
        )
        await self.runtime.add_subscription(
            TypeSubscription(
                topic_type=self.group_chat_topic_type,
                agent_type=reviewer_agent_type.type,
            )
        )

        researcher_agent_type = await ResearchAgent.register(
            self.runtime,
            research_topic_type,
            lambda: ResearchAgent(
                description=self.agent_config.researcher_description,
                group_chat_topic_type=self.group_chat_topic_type,
                model_client=self.llm_client,
                system_prompt=self.agent_config.researcher_system_prompt,
            ),
        )

        await self.runtime.add_subscription(
            TypeSubscription(
                topic_type=research_topic_type, agent_type=researcher_agent_type.type
            )
        )
        await self.runtime.add_subscription(
            TypeSubscription(
                topic_type=self.group_chat_topic_type,
                agent_type=researcher_agent_type.type,
            )
        )

        self.metadata = {}
        chat_supervisor_type = await ChatSupervisor.register(
            self.runtime,
            "group_chat_manager",
            lambda: ChatSupervisor(
                participant_topic_types=[
                    tagger_topic_type,
                    reviewer_topic_type,
                    research_topic_type,
                ],
                model_client=self.llm_client,
                participant_descriptions=[
                    self.agent_config.tagger_description,
                    self.agent_config.reviewer_description,
                    self.agent_config.researcher_description,
                ],
                metadata=self.metadata,
                grounding_engine=self.grounding_engine,
            ),
        )

        await self.runtime.add_subscription(
            TypeSubscription(
                topic_type=self.group_chat_topic_type,
                agent_type=chat_supervisor_type.type,
            )
        )

        return self.runtime

    async def recognize_async(
        self, tokens: List[str], left_context: str = "", right_context: str = ""
    ):
        await self.initialize_agents()

        query_template = "{}\n\n<text_to_tag>{}</text_to_tag>\n\n{}\n\nOnly tag this text: <text_to_tag>{}</text_to_tag>"
        query = query_template.format(
            left_context, " ".join(tokens), right_context, " ".join(tokens)
        )
        self.metadata["query"] = (
            f"Remember, you need to tag the following:\n <text_to_tag>{' '.join(tokens)}</text_to_tag>"
        )
        self.metadata["tokens"] = tokens
        self.metadata["entity_types"] = self.entity_types
        self.metadata["convert_to_genia_labels"] = (
            MultiAgentTagger.convert_to_genia_labels
        )

        print(f"Starting the chat with following query: {query}")
        self.runtime.start()
        session_id = str(uuid.uuid4())
        await self.runtime.publish_message(
            GroupChatMessage(
                body=UserMessage(
                    content=query,
                    source="User",
                )
            ),
            TopicId(type=self.group_chat_topic_type, source=session_id),
        )
        await self.runtime.stop_when_idle()

        tagged_string, genia_labels = MultiAgentTagger.convert_to_genia_labels(
            self.metadata.get("last_tagger_output", ""), tokens, self.entity_types  # type: ignore
        )
        print(f"Predicted entities: {genia_labels}")

        return tagged_string, Converter.convert_genia_to_iob2(genia_labels, tokens)

    def recognize(
        self, tokens: List[str], left_context: str = "", right_context: str = ""
    ) -> Tuple[str, List[str]]:
        response = asyncio.run(
            self.recognize_async(tokens, left_context, right_context)
        )
        return response  # type: ignore

    def recognize_with_feedback(self, tokens: List[str]) -> Tuple[str, List[str]]:
        pass
