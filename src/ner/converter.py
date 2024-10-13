import copy
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Converter:
    @staticmethod
    def convert_genia_to_iob2(entities: List[Dict[str, Any]], tokens: List[str]):
        tags = ["O"] * len(tokens)
        sorted_entities = sorted(entities, key=lambda x: x["start"])

        for entity in sorted_entities:
            start = entity["start"]
            end = entity["end"]
            tag = entity["type"]

            tags[start] = f"B-{tag}"
            for i in range(start + 1, end):
                if not tags[i].startswith("B-"):
                    tags[i] = f"I-{tag}"

        return tags

    @staticmethod
    def convert_genia_to_example(
        entities: List[Dict[str, Any]], tokens: List[str]
    ) -> str:
        tokens_ = copy.deepcopy(tokens)
        for entity in entities:
            start_pos = entity["start"]
            end_pos = entity["end"] - 1
            entity_type = entity["type"]

            tokens_[start_pos] = f"<{entity_type}>{tokens_[start_pos]}"
            tokens_[end_pos] = f"{tokens_[end_pos]}</{entity_type}>"

        return " ".join(tokens_)


if __name__ == "__main__":
    sample_tokens = [
        "Immunoprecipitation",
        "of",
        "the",
        "gp",
        "160",
        "-induced",
        "nuclear",
        "extracts",
        "with",
        "polyclonal",
        "antibodies",
        "to",
        "Fos",
        "and",
        "Jun",
        "proteins",
        "indicates",
        "that",
        "AP-1",
        "complex",
        "is",
        "comprised",
        "of",
        "members",
        "of",
        "these",
        "family",
        "of",
        "proteins.",
    ]

    sample_entities = [
        {"start": 18, "end": 19, "type": "protein"},
        {"start": 12, "end": 13, "type": "protein"},
        {"start": 9, "end": 11, "type": "protein"},
        {"start": 14, "end": 16, "type": "protein"},
        {"start": 3, "end": 5, "type": "protein"},
        {"start": 18, "end": 20, "type": "protein"},
    ]

    print(f"IOB2: {Converter.convert_genia_to_iob2(sample_entities, sample_tokens)}")
    print(f"Tokens: {sample_tokens}")
    print(f"Entities: {sample_entities}")
    print(
        f"Example: {Converter.convert_genia_to_example(sample_entities, sample_tokens)}"
    )
