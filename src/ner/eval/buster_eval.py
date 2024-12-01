from ner.agents.multi_agent_tagger import MultiAgentTagger
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.ontology import get_buster_ontology
from ner.eval.dataset import NERDataset
from ner.eval.eval import run_eval
from ner.prompts import get_agent_config, get_ner_prompt
from ner.tagger_few_shot import FewShotTagger


def run_few_shot_eval():
    print("Running few shot NER eval")
    dataset = NERDataset.from_buster("FOLD_2").sample(500)
    dev_dataset = NERDataset.from_buster("FOLD_1")
    ontology = get_buster_ontology()
    domain = "Finance, Law, Business"
    output_file = "buster_few_show_eval"

    system_prompt = get_ner_prompt(
        domain=domain, ontology=ontology, examples=dev_dataset.get_examples(3)
    )
    print("System prompt: ")
    print(system_prompt)

    llm_client = AnthropicClient(ClaudeFamily.HAIKU_35)
    tagger = FewShotTagger(
        llm_client=llm_client,
        entity_types=dataset.entity_types,
        system_prompt=system_prompt,
    )

    run_eval(tagger, dataset, output_file)

    #    Eval result:
    #                             precision    recall  f1-score   support
    #
    #           ACQUIRED_COMPANY       0.49      0.62      0.55       326
    #            ANNUAL_REVENUES       0.11      0.52      0.18        31
    #             BUYING_COMPANY       0.53      0.70      0.60       436
    # GENERIC_CONSULTING_COMPANY       0.69      0.90      0.78        81
    #   LEGAL_CONSULTING_COMPANY       0.79      0.90      0.84        29
    #            SELLING_COMPANY       0.33      0.57      0.42        95
    #
    #                  micro avg       0.47      0.68      0.56       998
    #                  macro avg       0.49      0.70      0.56       998
    #               weighted avg       0.50      0.68      0.57       998


def run_multi_agent_eval():
    print("Running multi-agent NER eval")
    dataset = NERDataset.from_buster("FOLD_2").sample(500)
    dev_dataset = NERDataset.from_buster("FOLD_1")
    ontology = get_buster_ontology()
    domain = "Finance, Law, Business"
    output_file = "buster_multi_agent_eval"
    examples = dev_dataset.get_examples(3)

    agent_config = get_agent_config(domain, ontology, examples)

    llm_client = AnthropicClient(ClaudeFamily.HAIKU_35)
    tagger = MultiAgentTagger(
        llm_client,
        dataset.entity_types,
        agent_config,
    )

    run_eval(tagger, dataset, output_file)


if __name__ == "__main__":
    # run_few_shot_eval()
    run_multi_agent_eval()
