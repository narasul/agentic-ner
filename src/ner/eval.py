import json
from typing import Any, Dict, List, Tuple
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.converter import Converter
from ner.few_shot import NERMaster


def run_eval(predictions: List[List[str]], references: List[List[str]]):
    print(classification_report(references, predictions, scheme=IOB2))


def load_references(path: str) -> Tuple[Any, Any]:
    with open(path) as file:
        raw_references = json.loads(file.read())

    return raw_references, [
        Converter.convert_genia_to_iob2(reference["entities"], reference["tokens"])
        for reference in raw_references
    ]


def get_predictions(
    ner_master: NERMaster, test_data: List[Dict[str, Any]], subset_size: int
):
    print(f"Length of test data: {len(test_data)}. Taking subset of {subset_size}")
    subset = test_data[:subset_size]
    predictions = []
    for data in subset:
        predictions.append(ner_master.recognize(data["tokens"]))

    return predictions


if __name__ == "__main__":
    subset_size = 500
    test_data, references = load_references("data/genia_test.json")
    print(f"Loaded references. Sample reference: {references[0]}")

    predictions = get_predictions(
        NERMaster(AnthropicClient(ClaudeFamily.SONNET_35)), test_data, subset_size
    )
    print(f"Loaded predictions. Sample prediction: {predictions[0]}")

    print("\n\nEval result:")
    run_eval(references[:subset_size], predictions)

#            Sample eval results with subset_size of 10
#
#               precision    recall  f1-score   support
#
#          DNA       0.62      0.61      0.61        38
#          RNA       0.50      0.50      0.50         2
#    cell_line       0.00      0.00      0.00         0
#    cell_type       0.83      0.18      0.29        57
#      protein       0.41      0.68      0.51        47
#
#    micro avg       0.44      0.46      0.45       144
#    macro avg       0.47      0.39      0.38       144
# weighted avg       0.63      0.46      0.45       144
