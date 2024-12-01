from typing import Dict, List, Set
from collections import namedtuple
from pydantic import BaseModel, Field

from ner.eval.dataset import NERDataset


class GroundingKnowledgeBase(BaseModel):
    data: Dict[str, Set[str]] = Field(default_factory=dict)
    token_separator: str = "___"

    def to_knowledge_key(self, tokens: List[str]) -> str:
        return self.token_separator.join([token.lower() for token in tokens])


WrongPrediction = namedtuple("WrongPrediction", ["predicted_tags", "grounded_tags"])


class GroundingFeedback(BaseModel):
    correct: Dict[str, List[str]] = Field(default_factory=dict)
    wrong: Dict[str, WrongPrediction] = Field(default_factory=dict)


    def get_text_feedback(self, token_separator: str = "___", include_correct: bool = False) -> str:
        feedback = ""

        if include_correct:
            for correct_entity in self.correct:
                for tag in self.correct[correct_entity]:
                    feedback += f"- '{correct_entity.replace(token_separator, " ")}' is correctly tagged as '{tag}'\n"


        for wrong_entity, wrong_prediction in self.wrong.items():
            for tag in wrong_prediction.predicted_tags:
                feedback += f"- '{wrong_entity.replace(token_separator, " ")}' is tagged as '{tag}'. It should likely be {' or '.join(wrong_prediction.grounded_tags)} instead.\n"


        return feedback

class GroundingEngine(BaseModel):
    knowledge_base: GroundingKnowledgeBase = Field(default_factory=GroundingKnowledgeBase)

    @staticmethod
    def from_ner_dataset(dataset: NERDataset) -> "GroundingEngine":
        engine = GroundingEngine()
        for entry in dataset.entries:
            engine.knowledge_base.data = GroundingEngine._upadate_grounding_data(
                engine.knowledge_base, entry.tokens, entry.labels, engine.knowledge_base.data
            )
        
        return engine


    @staticmethod
    def _upadate_grounding_data(knowledge_base: GroundingKnowledgeBase, tokens: List[str], labels: List[str], data: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        for i, label in enumerate(labels):
            if label.startswith("B"):
                start_idx = i
                end_idx = len(labels)
                for j in range(i, len(labels)):
                    if labels[j] == "O":
                        end_idx = j
                        break

                knowledge_key = knowledge_base.to_knowledge_key(
                    tokens[start_idx:end_idx]
                )
                data[knowledge_key] = data.get(
                    knowledge_key, set()
                )
                data[knowledge_key].add(label[2:])

        if '' in data:
            del data['']
        return data


    def verify(self, tokens: List[str], predicted_labels: List[str]) -> GroundingFeedback:
        print(f"Running grounding engine on predicted labels: {predicted_labels}. Tokens: {tokens}")
        grounding_data = GroundingEngine._upadate_grounding_data(self.knowledge_base, tokens, predicted_labels, dict())
        print(f"Running grounding engine. Grounding data: {grounding_data}")
        grounding_feedback = GroundingFeedback()

        for entity, predicted_tags in grounding_data.items():
            grounded_tags = self.knowledge_base.data.get(entity)
            if grounded_tags:
                print(f"Grounded tags: {grounded_tags}")
                diff = list(set(predicted_tags) - set(grounded_tags))
                grounding_feedback.wrong[entity] = WrongPrediction(predicted_tags=diff, grounded_tags=grounded_tags)

                correct = list()
                for tag in predicted_tags:
                    if tag in grounded_tags:
                        correct.append(tag)
                        grounding_feedback.correct[entity] = correct

        # print(f"Grounding feedback with correct tags included: {grounding_feedback.get_text_feedback()}")
        return grounding_feedback



if __name__ == "__main__":
    print("Loading Genia dataset")
    dataset = NERDataset.from_genia("train")

    print("Building Grounding Engine from Genia dataset")
    grounding_engine = GroundingEngine.from_ner_dataset(dataset)

    print(grounding_engine.knowledge_base.data)

    print("Testing grounding engine.")

    prediction = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-DNA', 'I-DNA', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-DNA', 'I-DNA', 'O', 'O']
    tokens = ['Having', 'previously', 'shown', 'that', 'Ca(2+)-', 'and', 'PKC-', 'dependent', 'pathways', 'synergize', 'by', 'accelerating', 'the', 'degradation', 'of', 'pax-5', 'gene', ',', 'we', 'focused', 'on', 'the', 'regulation', 'of', 'IkB', 'alpha', 'phosphorylation', '.']


    feedback = grounding_engine.verify(tokens, prediction)
    print(f"Feedback: {feedback.get_text_feedback(include_correct=True)}")
