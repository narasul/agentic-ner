import json
from typing import List, Tuple
from dataclasses import dataclass
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.clients.llm_client import LLMClient
from ner.converter import Converter
from ner.prompts import SYSTEM_PROMPT_FOR_XML_OUTPUT, get_system_prompt_with_feedback
from ner.tagger import Tagger


@dataclass
class FewShotTagger(Tagger):
    system_prompt: str
    llm_client: LLMClient

    def recognize(self, tokens: List[str], left_context: str = "", right_context: str = "") -> Tuple[str, List[str]]:
        query_template = "{}\n\n<text_to_tag>{}</text_to_tag>\n\n{}\n\nOnly tag this text: <text_to_tag>{}</text_to_tag>"
        query = query_template.format(
            left_context, " ".join(tokens), right_context, " ".join(tokens)
        )
        llm_output = self.llm_client.get_llm_response(
            query, self.system_prompt
        )
        tagged_string, genia_labels = FewShotTagger.convert_to_genia_labels(llm_output, tokens, self.entity_types)
        print(f"Predicted entities: {genia_labels}")

        return tagged_string, Converter.convert_genia_to_iob2(genia_labels, tokens)

    def recognize_with_feedback(
        self, tokens: List[str], previous_output: str, feedback: str
    ) -> Tuple[str, List[str]]:
        query = " ".join(tokens)
        system_prompt_with_feedback = get_system_prompt_with_feedback(query, previous_output, feedback)
        llm_output = self.llm_client.get_llm_response(
            query, system_prompt_with_feedback
        )
        tagged_string, genia_labels = FewShotTagger.convert_to_genia_labels(llm_output, tokens)
        print(f"Predicted entities: {genia_labels}\n\n")

        return tagged_string, Converter.convert_genia_to_iob2(genia_labels, tokens)


if __name__ == "__main__":
    with open("data/genia_test.json") as file:
        raw_references = json.loads(file.read())
    
    entity_types = ["DNA", "RNA", "protein", "cell_type", "cell_line"]

    ner_master = FewShotTagger(entity_types, SYSTEM_PROMPT_FOR_XML_OUTPUT, AnthropicClient(ClaudeFamily.SONNET_35))
    subset = raw_references[:5]
    for reference in subset:
        print(f"Reference entities: {reference["entities"]}")
        ner_master.recognize(reference["tokens"])

        print("\n\n-----------\n\n")
