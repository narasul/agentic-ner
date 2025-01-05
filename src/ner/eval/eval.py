import json
from time import sleep
from datetime import datetime
from typing import List, Tuple
import numpy as np
from seqeval.metrics.v1 import precision_recall_fscore_support
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2

from ner.eval.dataset import NERDataset
from ner.tagger import Tagger


def get_predictions(tagger: Tagger, test_data: NERDataset) -> List[List[str]]:
    print(f"Length of test data: {len(test_data.references)}.")
    predictions = []
    try:
        for entry in tqdm(test_data.entries):
            print(f"\n\nTo tag: {' '.join(entry.tokens)}")
            tagged_string, iob2_tags = tagger.recognize(
                entry.tokens, entry.left_context, entry.right_context
            )
            print(f"Tagged    : {tagged_string}")
            predictions.append(iob2_tags)
            sleep(2)
    except Exception as err:
        print(
            f"Something wrong happened: {str(err)}. Returning predictions gathered so far."
        )
        return predictions

    return predictions


def run_eval(
    tagger: Tagger,
    dataset: NERDataset,
    output_file: str,
    return_scores=False,
):
    print(f"Test dataset size: {len(dataset.references)}")

    predictions = get_predictions(tagger, dataset)
    print(f"Loaded predictions. Sample prediction: {predictions[0]}")

    print("\n\nEval result:")
    print(
        classification_report(
            dataset.references[: len(predictions)], predictions, scheme=IOB2
        )
    )

    with open(f"pred/{output_file}-{datetime.now().isoformat()}.json", "w") as file:
        file.write(json.dumps(predictions))

    if return_scores:
        return precision_recall_fscore_support(
            dataset.references[: len(predictions)],
            predictions,
            scheme=IOB2,
            average="micro",
        )


def calculate_std_dev(scores: List[Tuple[float, float, float, float]]):
    precision_scores = np.array([score[0] for score in scores])
    recall_scores = np.array([score[1] for score in scores])
    f1_scores = np.array([score[2] for score in scores])

    p_dev = np.std(precision_scores, ddof=1)
    r_dev = np.std(recall_scores, ddof=1)
    f1_dev = np.std(f1_scores, ddof=1)

    print(f"Standard deviation for precision: {p_dev}")
    print(f"Standard deviation for recall: {r_dev}")
    print(f"Standard deviation for f1 score: {f1_dev}")
