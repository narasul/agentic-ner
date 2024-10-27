from dataclasses import dataclass
from typing import List, Any, Dict
from abc import ABC, abstractmethod
from ner.clients.llm_client import LLMClient


@dataclass
class Tagger(ABC):
    llm_client: LLMClient

    @abstractmethod
    def recognize(self, tokens: List[str]) -> List[str]:
        pass

    @staticmethod
    def convert_to_genia_labels(
        llm_output: str, tokens: List[str]
    ) -> List[Dict[str, Any]]:
        tagged_string = (
            Tagger._extract_tag(llm_output, "output").replace("\\n", "").strip()
        )
        print(f"Prediction: {tagged_string}")
        entities = []
        for entity_type in ["DNA", "RNA", "protein", "cell_type"]:
            while entity := Tagger._extract_tag(tagged_string, entity_type):
                entity_tokens = entity.split(" ")
                start_pos = Tagger._get_position(entity_tokens[0], tokens)
                end_pos = Tagger._get_position(entity_tokens[-1], tokens)
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
                        tokens[pos] = "_"
                tagged_string = Tagger._remove_tag(tagged_string, entity_type)
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
