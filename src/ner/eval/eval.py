import json
from time import sleep
from datetime import datetime
from typing import List
from tqdm import tqdm
from seqeval.metrics import classification_report
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
