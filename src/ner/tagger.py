from nltk import word_tokenize
from dataclasses import dataclass
from typing import List, Any, Dict, Tuple
from abc import ABC, abstractmethod
from ner.clients.llm_client import LLMClient
from ner.helper import extract_tag


@dataclass
class Tagger(ABC):
    llm_client: LLMClient
    entity_types: List[str]

    @abstractmethod
    def recognize(
        self, tokens: List[str], left_context: str = "", right_context: str = ""
    ) -> Tuple[str, List[str]]:
        pass

    @abstractmethod
    def recognize_with_feedback(
        self, tokens: List[str], previous_output: str, feedback: str
    ) -> Tuple[str, List[str]]:
        pass

    @staticmethod
    def convert_to_genia_labels(
        llm_output: str,
        tokens: List[str],
        entity_types: List[str] = ["DNA", "RNA", "protein", "cell_type", "cell_line"],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        tagged_string = extract_tag(llm_output, "output").replace("\\n", "").strip()
        tagged_string_to_return = tagged_string
        entities = []
        for entity_type in entity_types:
            while entity := extract_tag(tagged_string, entity_type):
                entity_tokens = word_tokenize(
                    Tagger._remove_all_tags(
                        entity, entity_types
                    )  # handle nested NER correctly by removing extra tags inside extracted string
                )
                start_pos, end_pos = Tagger._get_first_and_last_index(
                    entity_tokens, tokens
                )
                if start_pos != -1 and end_pos != -1:
                    entities.append(
                        {
                            "start": start_pos,
                            "end": end_pos + 1,
                            "type": entity_type,
                        }
                    )

                    # mask the already tagged tokens so that if entity is in more than one
                    # place in the sentence the second/third/etc. occurence of the entity
                    # is tagged correctly
                    for pos in range(start_pos, end_pos):
                        tokens[pos] = "_"
                tagged_string = Tagger._remove_tag(tagged_string, entity_type)
        return tagged_string_to_return, entities

    @staticmethod
    def _remove_tag(llm_output: str, tag: str) -> str:
        return llm_output.replace(f"<{tag}>", "", 1).replace(f"</{tag}>", "", 1)

    @staticmethod
    def _remove_all_tags(text: str, entity_types: List[str]) -> str:
        tagless = text
        for entity_type in entity_types:
            tagless = Tagger._remove_tag(tagless, entity_type)

        return tagless

    @staticmethod
    def _get_first_and_last_index_naive(
        first_token: str, last_token: str, tokens: List[str]
    ) -> Tuple[int, int]:
        lowercase_tokens = [token.lower() for token in tokens]
        first_token = first_token.lower()
        last_token = last_token.lower()

        first_index = -1
        last_index = -1

        for i, token in enumerate(lowercase_tokens):
            if token == first_token:
                first_index = i
            if token == last_token:
                last_index = i

            if first_index != -1 and last_index >= first_index:
                break

        if last_index < first_index:
            return -1, -1

        return first_index, last_index

    @staticmethod
    def _get_first_and_last_index(
        entity_tokens: List[str], tokens: List[str]
    ) -> Tuple[int, int]:
        tokens = [token.lower() for token in tokens]
        entity_tokens = [token.lower() for token in entity_tokens]

        # Find the longest consecutive sequence matching entity_tokens inside tokens
        max_length = 0
        best_start_index = -1

        # Iterate through all possible lengths up to the full entity_tokens length
        for end_b in range(1, len(entity_tokens) + 1):
            candidate = entity_tokens[:end_b]

            for start_b in range(len(tokens) - end_b + 1):
                if tokens[start_b : start_b + end_b] == candidate:
                    if end_b > max_length:
                        max_length = end_b
                        best_start_index = start_b
                    break

        if max_length == 0:
            return -1, -1

        return best_start_index, best_start_index + max_length - 1
