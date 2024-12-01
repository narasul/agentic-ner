import copy
import random
import numpy as np

from typing import List, Any, Dict
from datasets import Dataset, load_dataset
from nltk import word_tokenize
from pydantic import BaseModel, Field


from ner.converter import Converter


SAMPLING_SEED = 43


class NERDatasetEntry(BaseModel):
    left_context: str
    right_context: str
    text: str
    tokens: List[str]
    labels: List[str]


class Example(BaseModel):
    left_context: str
    right_context: str
    text_to_tag: str
    tagged_text: str


class NERDataset(BaseModel):
    entity_types: List[str] = Field(default_factory=list)
    entries: List[NERDatasetEntry] = Field(default_factory=list)
    references: List[List[str]]

    def get_examples(self, n: int = 3) -> List[Example]:
        examples = list()
        all_entities_present_in_examples = False
        while not all_entities_present_in_examples:
            examples = list()
            sample = self.sample(n, fix_seed=False).entries
            labels_set = list()
            for entry in sample:
                labels_set.append(entry.labels)
                tagged_text = Converter.convert_iob2_to_example(
                    entry.labels, entry.tokens
                )
                examples.append(
                    Example(
                        left_context=entry.left_context,
                        right_context=entry.right_context,
                        text_to_tag=entry.text,
                        tagged_text=tagged_text,
                    )
                )

            # make sure all entity types are covered in the examples
            covered_entities = set()
            for labels in labels_set:
                for label in labels:
                    if label.startswith("B") or label.startswith("I"):
                        covered_entities.add(label[2:])

            if len(covered_entities) == len(self.entity_types):
                all_entities_present_in_examples = True

        return examples

    def sample(self, n: int, fix_seed: bool = True) -> "NERDataset":
        if fix_seed:
            random.seed(SAMPLING_SEED)
        random_sample = copy.deepcopy(random.sample(self.entries, n))

        references = list()
        for entry in random_sample:
            references.append(copy.deepcopy(entry.labels))

        return NERDataset(
            entries=random_sample, references=references, entity_types=self.entity_types
        )

    @staticmethod
    def from_buster(
        fold: str, sample_size: int = -1, contextify: bool = True
    ) -> "NERDataset":
        dataset: Dataset = load_dataset("expertai/BUSTER")[fold]  # type: ignore

        entities = set()
        entries = list()
        as_list = dataset.data.to_pylist()[:sample_size]

        for raw_entry in as_list:
            batch = NERDataset._from_buster_to_ner_entry(raw_entry, contextify)
            entries.extend(batch)

        references = list()
        for entry in entries:
            for label in entry.labels:
                if label != "O":
                    entities.add(label[2:])

            references.append(entry.labels)

        # shuffle the data
        np.random.seed(SAMPLING_SEED)
        indices = np.random.permutation(len(entries))
        references = [references[i] for i in indices]
        entries = [entries[i] for i in indices]

        return NERDataset(
            entries=entries, entity_types=list(entities), references=references
        )

    @staticmethod
    def from_genia(split: str, sample_size: int = -1) -> "NERDataset":  # type: ignore
        dataset: Dataset = load_dataset("Rosenberg/genia")[split]  # type: ignore

        as_list = dataset.data.to_pylist()[:sample_size]

        entries = [
            NERDatasetEntry(
                left_context=" ".join(raw_entry["ltokens"]),
                right_context=" ".join(raw_entry["rtokens"]),
                tokens=raw_entry["tokens"],
                labels=Converter.convert_genia_to_iob2(
                    raw_entry["entities"], raw_entry["tokens"]
                ),
                text=" ".join(raw_entry["tokens"]),
            )
            for raw_entry in as_list
        ]

        entities = set()
        references = list()
        for entry in entries:
            for label in entry.labels:
                if label != "O":
                    entities.add(label[2:])

            references.append(entry.labels)

        # shuffle the data
        np.random.seed(SAMPLING_SEED)
        indices = np.random.permutation(len(entries))
        references = [references[i] for i in indices]
        entries = [entries[i] for i in indices]

        return NERDataset(
            entries=entries, entity_types=list(entities), references=references
        )

    @staticmethod
    def from_astroner(split: str, sample_size: int = -1) -> "NERDataset":
        pass

    @staticmethod
    def from_musicner(split: str, sample_size: int = -1) -> "NERDataset":
        pass

    @staticmethod
    def _from_buster_to_ner_entry(
        raw_entry: Dict[str, Any], contextify: bool
    ) -> List[NERDatasetEntry]:
        tokens = raw_entry["tokens"]

        labels = list()
        for label in raw_entry["labels"]:
            if label.startswith("B") or label.startswith("I"):
                flattened_label = (
                    label
                    if label.startswith("O")
                    else f"{label[:2]}{Converter.get_buster_entity_type(label)}"
                )
                labels.append(flattened_label)
            else:
                labels.append(label)

        normalized_tokens = list()
        normalized_labels = list()
        for i, token in enumerate(tokens):
            word_tokens = word_tokenize(token)
            word_labels = [labels[i]] * len(word_tokens)

            # some tokens in buster dataset consist of two words. e.g. '3 billion'
            normalized_tokens.extend(word_tokens)
            normalized_labels.extend(word_labels)

        labels = normalized_labels
        tokens = normalized_tokens

        if not contextify:
            return [
                NERDatasetEntry(
                    text=raw_entry["text"],
                    left_context="",
                    right_context="",
                    labels=labels,
                    tokens=tokens,
                )
            ]

        batch = list()
        left = list()
        right = list()
        previous_dot_index = -1
        for i in range(len(tokens)):
            if (
                tokens[i] == "."
                and labels[i] == "O"
                and i != 0
                and tokens[i - 1] != "Inc"
            ) or i == len(tokens) - 1:
                if previous_dot_index != -1:
                    left = tokens[: previous_dot_index + 1]
                if i < len(tokens) - 1:
                    right = tokens[i + 1 :]

                text = " ".join(tokens[previous_dot_index + 1 : i])

                labels_copy = copy.deepcopy(labels[previous_dot_index + 1 : i])
                tokens_copy = copy.deepcopy(tokens[previous_dot_index + 1 : i])

                left_context = " ".join(left)
                right_context = " ".join(right)

                for label in labels_copy:
                    # filter the batch and only keep the ones with tags
                    if label.startswith("I") or label.startswith("B"):
                        batch.append(
                            NERDatasetEntry(
                                text=text,
                                left_context=left_context,
                                right_context=right_context,
                                labels=labels_copy,
                                tokens=tokens_copy,
                            )
                        )
                        break

                previous_dot_index = i

        return batch


if __name__ == "__main__":
    genia_dataset = NERDataset.from_genia("test")
    print(genia_dataset.sample(5))
