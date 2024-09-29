from abc import ABC, abstractmethod
from typing import Any, List


class LLMClient(ABC):
    @abstractmethod
    def get_llm_response(
        self, query: str, system_prompt: str = "", functions: List[Any] = []
    ) -> str:
        raise NotImplementedError()
