from autogen_core.components.models import ChatCompletionClient

from ner.agents.base_agent import BaseGroupChatAgent


TAGGER_TOPIC_TYPE = "Tagger"


class TaggerAgent(BaseGroupChatAgent):
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
