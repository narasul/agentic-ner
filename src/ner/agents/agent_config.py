from pydantic import BaseModel


class AgentConfig(BaseModel):
    tagger_system_prompt: str
    reviewer_system_prompt: str
    researcher_system_prompt: str

    tagger_description: str = (
        "Agent responsible for Named Entity Recognition inside given text"
    )
    reviewer_description: str = (
        "Agent responsible for reviewing the output of the tagger"
    )
    researcher_description: str = (
        "Agent responsible for researching the entities requested by the tagger or reviewer"
    )
