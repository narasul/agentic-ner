import json
import copy
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field

import nltk
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.clients.llm_client import LLMClient
from ner.converter import Converter
from ner.prompts import SYSTEM_PROMPT_FOR_XML_OUTPUT, get_system_prompt_with_feedback
from ner.tagger import Tagger
from ner.helper import extract_tag

@dataclass
class FewShotTagger(Tagger):
    system_prompt: str
    llm_client: LLMClient
    metadata: Dict[str, Any] = field(default_factory=dict)
    def recognize(self, tokens: List[str], left_context: str = "", right_context: str = "") -> Tuple[str, List[str]]:
        self.metadata["distances"] = self.metadata.get("distances", [[], []])
        query_template = "{}\n\n<text_to_tag>{}</text_to_tag>\n\n{}\n\nOnly tag this text: <text_to_tag>{}</text_to_tag>"
        query = query_template.format(
            left_context, " ".join(tokens), right_context, " ".join(tokens)
        )
        llm_output = self.llm_client.get_llm_response(
            query, self.system_prompt
        )
        tokens_copy = copy.deepcopy(tokens)
        tagged_string, genia_labels = FewShotTagger.convert_to_genia_labels(llm_output, tokens, self.entity_types)
        print(f"Predicted entities: {genia_labels}")

        raw_tagged_string = extract_tag(llm_output, "output").replace("\\n", "").strip()

        llm_output_without_tags = Tagger._remove_all_tags(raw_tagged_string, self.entity_types)
        input_sentence = " ".join(tokens_copy)
        print(f"Tagged string without tags: {llm_output_without_tags}")
        print(f"Input string: {input_sentence}")

        distance = nltk.edit_distance(input_sentence, llm_output_without_tags)
        print(f"Edit distance: {distance}")
        print(f"Length diff: {len(input_sentence) - len(llm_output_without_tags)}")
        self.metadata["distances"][0].append(len(input_sentence))
        self.metadata["distances"][1].append(distance)


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
