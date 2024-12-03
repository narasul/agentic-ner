# eval for: https://github.com/deezer/music-ner-eacl2023/tree/mai://github.com/deezer/music-ner-eacl2023/tree/main

from ner.ablation.prompts import (
    get_agent_config_no_internet,
    get_agent_config_no_researcher,
)
from ner.agents.multi_agent_tagger import MultiAgentTagger
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.clients.claude_oai_compatible_client import create_chat_completions_client
from ner.grounding import GroundingEngine
from ner.eval.dataset import NERDataset
from ner.eval.eval import run_eval
from ner.ontology import get_musicner_ontology
from ner.prompts import get_agent_config, get_ner_prompt
from ner.tagger_few_shot import FewShotTagger


dataset = NERDataset.from_musicner(path="data/music_reco_ner/test.bio")
dev_dataset = NERDataset.from_musicner(path="data/music_reco_ner/train.bio")
ontology = get_musicner_ontology()
domain = "Music industry, entertainment"


def run_few_shot_eval(sonnet: bool = False):
    print("Running few shot NER eval")
    output_file = "music_ner_few_shot_eval"

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
    #       Artist       0.89      0.87      0.88       303
    #          WoA       0.74      0.84      0.79       208
    #
    #    micro avg       0.83      0.86      0.84       511
    #    macro avg       0.82      0.85      0.83       511
    # weighted avg       0.83      0.86      0.84       511


def run_multi_agent_eval(
    enable_grounding: bool = False,
    sonnet: bool = False,
    internet_access: bool = True,
    researcher: bool = True,
):
    print("Running multi-agent NER eval")
    output_file = "music_ner_multi_agent_eval"

    if not researcher:
        agent_config = get_agent_config_no_researcher(
            domain, ontology, dev_dataset.get_examples(3)
        )
    elif internet_access:
        agent_config = get_agent_config(domain, ontology, dev_dataset.get_examples(3))
    else:
        agent_config = get_agent_config_no_internet(
            domain, ontology, dev_dataset.get_examples(3)
        )

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

    # Eval result, with grounding. Haiku 3.5: - run again to confirm with no-grounding version
    #               precision    recall  f1-score   support
    #
    #       Artist       0.95      0.83      0.89       303
    #          WoA       0.76      0.82      0.79       208
    #
    #    micro avg       0.86      0.83      0.84       511
    #    macro avg       0.85      0.83      0.84       511
    # weighted avg       0.87      0.83      0.85       511

    # Eval result, without grounding. Haiku 3.5: - run again to confirm with the version above
    #               precision    recall  f1-score   support
    #
    #       Artist       0.93      0.78      0.85       303
    #          WoA       0.83      0.77      0.80       208
    #
    #    micro avg       0.88      0.78      0.83       511
    #    macro avg       0.88      0.78      0.82       511
    # weighted avg       0.89      0.78      0.83       511

    # Eval result, grounding, without internet access, Haiku 3.5:
    #               precision    recall  f1-score   support
    #
    #       Artist       0.91      0.82      0.86       303
    #          WoA       0.77      0.79      0.78       208
    #
    #    micro avg       0.85      0.81      0.83       511
    #    macro avg       0.84      0.81      0.82       511
    # weighted avg       0.85      0.81      0.83       511

    # Eval result, grounding, without researcher, Haiku 3.5:
    #               precision    recall  f1-score   support
    #
    #       Artist       0.95      0.83      0.89       303
    #          WoA       0.72      0.80      0.76       208
    #
    #    micro avg       0.84      0.82      0.83       511
    #    macro avg       0.84      0.81      0.82       511
    # weighted avg       0.86      0.82      0.83       511


if __name__ == "__main__":
    # run_few_shot_eval()
    # run_multi_agent_eval()  # no grounding
    # run_multi_agent_eval(enable_grounding=True)  # - running now
    run_multi_agent_eval(enable_grounding=False)  # - Also, running now
    # run_multi_agent_eval(enable_grounding=True, internet_access=False)
    # run_multi_agent_eval(enable_grounding=True, researcher=False)
