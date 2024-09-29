from typing import Any
import openai
from dataclasses import dataclass

from ner.clients.llm_client import LLMClient


@dataclass
class GorillaClient(LLMClient):
    model_name: str = "gorilla-openfunctions-v2"

    def get_llm_response(
        self, query: str, system_prompt: str = "", functions: List[Any] = []
    ) -> str:
        openai.api_key = "EMPTY"
        openai.api_base = "http://luigi.millennium.berkeley.edu:8000/v1"
        try:
            completion = openai.ChatCompletion.create(
                model=self.model_name,
                temperature=0.0,
                messages=[{"role": "user", "content": query}],
                functions=functions,
            )
            return str(completion.choices[0])
        except Exception as e:
            print(e, self.model_name, query)
            return ""


if __name__ == "__main__":
    client = GorillaClient()
    print(
        client.get_llm_response(
            "Give me a function to get weather conditions in New York."
        )
    )
