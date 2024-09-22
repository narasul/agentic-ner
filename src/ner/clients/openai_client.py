import openai
from dataclasses import dataclass


@dataclass
class GorillaClient:
    @staticmethod
    def get_gorilla_response(prompt: str, model: str = "gorilla-openfunctions-v0", functions=[]):
        openai.api_key = "EMPTY"
        openai.api_base = "http://luigi.millennium.berkeley.edu:8000/v1"
        try:
            completion = openai.ChatCompletion.create(
                model="gorilla-openfunctions-v2",
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                functions=functions,
            )
            return completion.choices[0]
        except Exception as e:
            print(e, model, prompt)

print(GorillaClient.get_gorilla_response("Give me a function to get weather conditions in New York."))
