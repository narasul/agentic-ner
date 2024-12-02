from ner.agents.multi_agent_tagger import MultiAgentTagger
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.clients.claude_oai_compatible_client import create_chat_completions_client
from ner.grounding import GroundingEngine
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

    #     Eval result, Haiku 3.5:
    #                             precision    recall  f1-score   support
    #
    #           ACQUIRED_COMPANY       0.54      0.60      0.56       322
    #            ANNUAL_REVENUES       0.14      0.54      0.23        48
    #             BUYING_COMPANY       0.56      0.72      0.63       425
    # GENERIC_CONSULTING_COMPANY       0.60      0.66      0.63        68
    #   LEGAL_CONSULTING_COMPANY       0.78      0.89      0.83        28
    #            SELLING_COMPANY       0.32      0.58      0.41       110
    #
    #                  micro avg       0.47      0.66      0.55      1001
    #                  macro avg       0.49      0.67      0.55      1001
    #               weighted avg       0.52      0.66      0.57      1001


def run_multi_agent_eval(enable_grounding: bool = False):
    print("Running multi-agent NER eval")
    dataset = NERDataset.from_buster("FOLD_2").sample(500)
    dev_dataset = NERDataset.from_buster("FOLD_1")
    ontology = get_buster_ontology()
    domain = "Finance, Law, Business"
    output_file = "buster_multi_agent_eval"
    examples = dev_dataset.get_examples(3)

    agent_config = get_agent_config(domain, ontology, examples)

    llm_client = create_chat_completions_client(ClaudeFamily.HAIKU_35.value)
    if enable_grounding:
        grounding_engine = GroundingEngine.from_ner_dataset(dev_dataset)
        tagger = MultiAgentTagger(
            dataset.entity_types, agent_config, llm_client, grounding_engine
        )
    else:
        tagger = MultiAgentTagger(dataset.entity_types, agent_config, llm_client)

    run_eval(tagger, dataset, output_file)

    #     Eval result without grounding engine, Haiku 3.5:
    #                                 precision    recall  f1-score   support
    #
    #               ACQUIRED_COMPANY       0.53      0.63      0.58       322
    #                ANNUAL_REVENUES       0.15      0.46      0.22        48
    #                 BUYING_COMPANY       0.59      0.72      0.65       425
    #     GENERIC_CONSULTING_COMPANY       0.62      0.54      0.58        68
    #       LEGAL_CONSULTING_COMPANY       0.81      0.89      0.85        28
    #                SELLING_COMPANY       0.35      0.57      0.44       110
    #
    #                      micro avg       0.50      0.66      0.57      1001
    #                      macro avg       0.51      0.64      0.55      1001
    #                   weighted avg       0.53      0.66      0.58      1001

    #     Eval result with grounding engine, Haiku 3.5:
    #                                 precision    recall  f1-score   support
    #
    #               ACQUIRED_COMPANY       0.54      0.60      0.57       322
    #                ANNUAL_REVENUES       0.18      0.54      0.27        48
    #                 BUYING_COMPANY       0.60      0.69      0.64       425
    #     GENERIC_CONSULTING_COMPANY       0.69      0.56      0.62        68
    #       LEGAL_CONSULTING_COMPANY       0.78      0.89      0.83        28
    #                SELLING_COMPANY       0.34      0.57      0.42       110
    #
    #                      micro avg       0.50      0.64      0.56      1001
    #                      macro avg       0.52      0.64      0.56      1001
    #                   weighted avg       0.54      0.64      0.58      1001


if __name__ == "__main__":
    # run_few_shot_eval()
    run_multi_agent_eval()
    # run_multi_agent_eval(enable_grounding=True)
