import json
from typing import List
from dataclasses import dataclass
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.converter import Converter
from ner.prompts import SYSTEM_PROMPT_FOR_XML_OUTPUT
from ner.tagger import Tagger


@dataclass
class FewShotTagger(Tagger):
    def recognize(self, tokens: List[str]) -> List[str]:
        query = " ".join(tokens)
        llm_output = self.llm_client.get_llm_response(
            query, SYSTEM_PROMPT_FOR_XML_OUTPUT
        )
        genia_labels = FewShotTagger.convert_to_genia_labels(llm_output, tokens)
        print(f"Predicted entities: {genia_labels}\n\n")

        return Converter.convert_genia_to_iob2(genia_labels, tokens)


if __name__ == "__main__":
    with open("data/genia_test.json") as file:
        raw_references = json.loads(file.read())

    ner_master = FewShotTagger(AnthropicClient(ClaudeFamily.SONNET_35))
    subset = raw_references[:5]
    for reference in subset:
        print(f"Reference entities: {reference["entities"]}")
        ner_master.recognize(reference["tokens"])

        print("\n\n-----------\n\n")
