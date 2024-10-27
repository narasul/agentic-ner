import json
from typing import Any, Dict, List, Tuple
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.converter import Converter
from ner.tagger import Tagger
from ner.tagger_few_shot import FewShotTagger


def run_eval(tagger: Tagger):
    test_data, references = load_references("data/genia_train_dev.json")
    print(f"Loaded references. Sample reference: {references[0]}")

    subset_size = len(references)
    subset_size = 60
    predictions = get_predictions(
        tagger, test_data, subset_size
    )
    print(f"Loaded predictions. Sample prediction: {predictions[0]}")

    print("\n\nEval result:")
    print(classification_report(references, predictions, scheme=IOB2))

    with open("data/pred.json", "w") as file:
        file.write(json.dumps(predictions))



def load_references(path: str) -> Tuple[Any, Any]:
    with open(path) as file:
        raw_references = json.loads(file.read())

    return raw_references, [
        Converter.convert_genia_to_iob2(reference["entities"], reference["tokens"])
        for reference in raw_references
    ]


def get_predictions(
    tagger: Tagger, test_data: List[Dict[str, Any]], subset_size: int
):
    print(f"Length of test data: {len(test_data)}. Taking subset of {subset_size}")
    subset = test_data[:subset_size]
    predictions = []
    for data in subset:
        print(f"Truth     : {Converter.convert_genia_to_example(data["entities"], data["tokens"])}")
        predictions.append(tagger.recognize(data["tokens"]))

    return predictions


if __name__ == "__main__":
    tagger = FewShotTagger(AnthropicClient(ClaudeFamily.SONNET_35))
    run_eval(tagger)

#            Sample eval results with subset_size of 600
#
#               precision    recall  f1-score   support
#
#          DNA       0.41      0.55      0.47       387
#          RNA       0.41      0.41      0.41        41
#    cell_line       0.00      0.00      0.00         0
#    cell_type       0.58      0.38      0.46       269
#      protein       0.53      0.46      0.50       841
#
#    micro avg       0.44      0.47      0.46      1538
#    macro avg       0.39      0.36      0.37      1538
# weighted avg       0.51      0.47      0.48      1538
