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
                tags[i] = f"I-{tag}"

        return tags


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
