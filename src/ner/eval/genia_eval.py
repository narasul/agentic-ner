from ner.ablation.prompts import (
    get_agent_config_no_internet,
    get_agent_config_no_researcher,
)
from ner.agents.multi_agent_tagger import MultiAgentTagger
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.clients.claude_oai_compatible_client import create_chat_completions_client
from ner.grounding import GroundingEngine
from ner.ontology import get_genia_ontology
from ner.eval.dataset import NERDataset
from ner.eval.eval import run_eval
from ner.prompts import get_agent_config, get_ner_prompt
from ner.tagger_few_shot import FewShotTagger


def run_few_shot_eval(sonnet: bool = False):
    print("Running few shot NER eval")
    dataset = NERDataset.from_genia("test").sample(500)
    dev_dataset = NERDataset.from_genia("train")
    ontology = get_genia_ontology()
    domain = "Biomedical, molecular biology, genomics"
    output_file = "genia_few_shot_eval"

    system_prompt = get_ner_prompt(
        domain=domain, ontology=ontology, examples=dev_dataset.get_examples(3)
    )
    print("System prompt: ")
    print(system_prompt)

    if sonnet:
        llm_client = AnthropicClient(ClaudeFamily.SONNET_35_V2)
    else:
        llm_client = AnthropicClient(ClaudeFamily.HAIKU_35)
    tagger = FewShotTagger(
        llm_client=llm_client,
        entity_types=dataset.entity_types,
        system_prompt=system_prompt,
    )

    run_eval(tagger, dataset, output_file)

    # Eval result, Haiku 3.5:
    #               precision    recall  f1-score   support
    #
    #          DNA       0.51      0.36      0.42       339
    #          RNA       0.40      0.40      0.40        25
    #    cell_line       0.65      0.41      0.50       125
    #    cell_type       0.35      0.60      0.44       176
    #      protein       0.57      0.56      0.57       709
    #
    #    micro avg       0.51      0.50      0.51      1374
    #    macro avg       0.50      0.47      0.47      1374
    # weighted avg       0.53      0.50      0.51      1374

    # Eval result Sonnet 3.5 V2:
    #               precision    recall  f1-score   support
    #
    #          DNA       0.51      0.38      0.44       339
    #          RNA       0.42      0.52      0.46        25
    #    cell_line       0.64      0.43      0.51       125
    #    cell_type       0.46      0.62      0.53       176
    #      protein       0.67      0.56      0.61       709
    #
    #    micro avg       0.59      0.51      0.55      1374
    #    macro avg       0.54      0.50      0.51      1374
    # weighted avg       0.60      0.51      0.55      1374


def run_multi_agent_eval(
    enable_grounding: bool = False,
    sonnet: bool = False,
    internet_access: bool = True,
    researcher: bool = True,
):
    print("Running multi-agent NER eval")
    dataset = NERDataset.from_genia("test").sample(500)
    dev_dataset = NERDataset.from_genia("train")
    examples = dev_dataset.get_examples(3)
    ontology = get_genia_ontology()
    domain = "Biomedical, molecular biology, genomics"
    output_file = "genia_multi_agent_eval"

    if not researcher:
        agent_config = get_agent_config_no_researcher(domain, ontology, examples)
    elif internet_access:
        agent_config = get_agent_config(domain, ontology, examples)
    else:
        agent_config = get_agent_config_no_internet(domain, ontology, examples)

    if sonnet:
        llm_client = create_chat_completions_client(ClaudeFamily.SONNET_35_V2.value)
    else:
        llm_client = create_chat_completions_client(ClaudeFamily.HAIKU_35.value)

    if enable_grounding:
        grounding_engine = GroundingEngine.from_ner_dataset(dev_dataset)
        tagger = MultiAgentTagger(
            dataset.entity_types,
            agent_config,
            llm_client,
            grounding_engine,
            internet_access,
            researcher,
        )
    else:
        tagger = MultiAgentTagger(
            dataset.entity_types,
            agent_config,
            llm_client,
            None,
            internet_access,
            researcher,
        )

    run_eval(tagger, dataset, output_file)

    # Eval result without grounding, Haiku 3.5:
    #               precision    recall  f1-score   support
    #
    #          DNA       0.53      0.35      0.42       339
    #          RNA       0.35      0.44      0.39        25
    #    cell_line       0.71      0.40      0.51       125
    #    cell_type       0.40      0.61      0.48       176
    #      protein       0.67      0.53      0.59       709
    #
    #    micro avg       0.58      0.48      0.52      1374
    #    macro avg       0.53      0.47      0.48      1374
    # weighted avg       0.60      0.48      0.52      1374

    # Eval result with grounding engine, Haiku 3.5:
    #               precision    recall  f1-score   support
    #
    #          DNA       0.51      0.34      0.40       339
    #          RNA       0.41      0.44      0.42        25
    #    cell_line       0.69      0.38      0.49       125
    #    cell_type       0.41      0.59      0.48       176
    #      protein       0.68      0.55      0.61       709
    #
    #    micro avg       0.58      0.48      0.53      1374
    #    macro avg       0.54      0.46      0.48      1374
    # weighted avg       0.60      0.48      0.53      1374

    # Eval result with grounding engine, no internet access, Haiku 3.5:
    #               precision    recall  f1-score   support
    #
    #          DNA       0.54      0.32      0.41       339
    #          RNA       0.29      0.40      0.33        25
    #    cell_line       0.62      0.38      0.48       125
    #    cell_type       0.41      0.62      0.50       176
    #      protein       0.71      0.58      0.64       709
    #
    #    micro avg       0.59      0.50      0.55      1374
    #    macro avg       0.52      0.46      0.47      1374
    # weighted avg       0.62      0.50      0.54      1374

    # Eval result, with grounding engine, no researcher, Haiku 3.5:
    #               precision    recall  f1-score   support
    #
    #          DNA       0.51      0.33      0.40       339
    #          RNA       0.52      0.44      0.48        25
    #    cell_line       0.68      0.42      0.52       125
    #    cell_type       0.41      0.64      0.50       176
    #      protein       0.69      0.58      0.63       709
    #
    #    micro avg       0.59      0.51      0.55      1374
    #    macro avg       0.56      0.48      0.51      1374
    # weighted avg       0.61      0.51      0.54      1374

    # Eval result Sonnet 3.5 V2 and grounding:

    #               precision    recall  f1-score   support
    #
    #          DNA       0.55      0.39      0.46       339
    #          RNA       0.39      0.48      0.43        25
    #    cell_line       0.70      0.46      0.55       125
    #    cell_type       0.50      0.60      0.54       176
    #      protein       0.75      0.59      0.66       709
    #
    #    micro avg       0.65      0.53      0.58      1374
    #    macro avg       0.58      0.50      0.53      1374
    # weighted avg       0.66      0.53      0.58      1374


if __name__ == "__main__":
    # run_few_shot_eval()
    # run_multi_agent_eval()
    # run_multi_agent_eval(enable_grounding=True)
    # run_few_shot_eval(sonnet=True)
    # run_multi_agent_eval(enable_grounding=True, sonnet=True)
    # run_multi_agent_eval(enable_grounding=True, internet_access=False)
    run_multi_agent_eval(enable_grounding=True, researcher=False)
