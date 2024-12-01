from dataclasses import dataclass
from abc import ABC, abstractmethod

from ner.clients.llm_client import LLMClient


@dataclass
class Feedback:
    grade: float
    feedback: str


@dataclass
class Grader(ABC):
    llm_client: LLMClient

    @abstractmethod
    def grade(self, prediction: str) -> Feedback:
        pass
