import json
from dataclasses import dataclass
from typing import Any, Dict, List
from ner.clients.llm_client import LLMClient
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.converter import Converter
from ner.prompts import SYSTEM_PROMPT_FOR_XML_OUTPUT


@dataclass
class NERMaster:
    llm_client: LLMClient

    def recognize(self, tokens: List[str]) -> List[str]:
        query = " ".join(tokens)
        llm_output = self.llm_client.get_llm_response(
            query, SYSTEM_PROMPT_FOR_XML_OUTPUT
        )
        genia_labels = NERMaster.convert_to_genia_labels(llm_output, tokens)
        print(f"Predicted entities: {genia_labels}\n\n")

        return Converter.convert_genia_to_iob2(genia_labels, tokens)

    @staticmethod
    def convert_to_genia_labels(
        llm_output: str, tokens: List[str]
    ) -> List[Dict[str, Any]]:
        tagged_string = NERMaster._extract_tag(llm_output, "output").replace("\\n", "").strip()
        print(f"Prediction: {tagged_string}")
        entities = []
        for entity_type in ["DNA", "RNA", "protein", "cell_type"]:
            while entity := NERMaster._extract_tag(tagged_string, entity_type):
                entity_tokens = entity.split(" ")
                start_pos = NERMaster._get_position(entity_tokens[0], tokens)
                end_pos = NERMaster._get_position(entity_tokens[-1], tokens)
                if start_pos != -1:
                    end_pos = end_pos + 1 if end_pos != -1 else start_pos + 1
                    entities.append(
                        {
                            "start": start_pos,
                            "end": end_pos,
                            "type": entity_type,
                        }
                    )

                    # mask the already tagged tokens so that if entity is in more than one
                    # place in the sentence the second/third/etc. occurence of the entity
                    # is tagged correctly
                    for pos in range(start_pos, end_pos):
                        tokens[pos] = '_'
                tagged_string = NERMaster._remove_tag(tagged_string, entity_type)
        return entities

    @staticmethod
    def _remove_tag(llm_output: str, tag: str) -> str:
        return llm_output.replace(f"<{tag}>", "", 1).replace(f"</{tag}>", "", 1)

    @staticmethod
    def _get_position(token: str, tokens: List[str]):
        try:
            return tokens.index(token)
        except Exception:
            return -1

    @staticmethod
    def _extract_tag(llm_output: str, tag: str) -> str:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        tag_len = len(open_tag)

        start_position = llm_output.find(open_tag)
        if start_position == -1:
            return ""

        end_position = llm_output.find(close_tag)

        try:
            return llm_output[start_position + tag_len : end_position]
        except Exception as err:
            print(f"Error while parsing llm output: {str(err)}")
            return ""


if __name__ == "__main__":
    with open("data/genia_test.json") as file:
        raw_references = json.loads(file.read())

    ner_master = NERMaster(AnthropicClient(ClaudeFamily.SONNET_35))
    subset = raw_references[:5]
    for reference in subset:
        print(f"Reference entities: {reference["entities"]}")
        ner_master.recognize(reference["tokens"])

        print("\n\n-----------\n\n")
