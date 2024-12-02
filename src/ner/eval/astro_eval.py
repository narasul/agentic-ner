# https://arxiv.org/abs/2405.02602 - dataset source

from ner.agents.multi_agent_tagger import MultiAgentTagger
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.clients.claude_oai_compatible_client import create_chat_completions_client
from ner.grounding import GroundingEngine
from ner.eval.dataset import NERDataset
from ner.eval.eval import run_eval
from ner.ontology import get_astroner_ontology
from ner.prompts import get_agent_config, get_ner_prompt
from ner.tagger_few_shot import FewShotTagger


dataset = NERDataset.from_astroner(path="data/astro_ner/test.json").sample(500)
dev_dataset = NERDataset.from_astroner(path="data/astro_ner/train.json")
ontology = get_astroner_ontology()
domain = "Astronomy, astronomy research"


def run_few_shot_eval(sonnet: bool = False):
    print("Running few shot NER eval")
    output_file = "astro_ner_few_shot_eval"

    system_prompt = get_ner_prompt(
        domain=domain, ontology=ontology, examples=dev_dataset.get_examples(12)
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

    # Eval result:
    #                   precision    recall  f1-score   support
    #
    #       AstrObject       0.17      0.64      0.27        14
    #     AstroPortion       0.18      0.20      0.19        15
    #  ChemicalSpecies       0.23      0.31      0.27        93
    #       Instrument       0.60      0.30      0.40        79
    #      Measurement       0.05      0.02      0.03        46
    #           Method       0.53      0.47      0.50       332
    #       Morphology       0.08      0.19      0.11        32
    # PhysicalQuantity       0.20      0.17      0.18        48
    #          Process       0.46      0.36      0.40       141
    #          Project       0.12      0.57      0.20        14
    #  ResearchProblem       0.30      0.34      0.32       361
    #   SpectralRegime       0.32      0.53      0.40        17
    #
    #        micro avg       0.33      0.36      0.35      1192
    #        macro avg       0.27      0.34      0.27      1192
    #     weighted avg       0.37      0.36      0.36      1192


def run_multi_agent_eval(enable_grounding: bool = False, sonnet: bool = False):
    print("Running multi-agent NER eval")
    output_file = "astro_ner_multi_agent_eval"

    agent_config = get_agent_config(domain, ontology, dev_dataset.get_examples(12))

    if sonnet:
        llm_client = create_chat_completions_client(ClaudeFamily.SONNET_35_V2.value)
    else:
        llm_client = create_chat_completions_client(ClaudeFamily.HAIKU_35.value)

    if enable_grounding:
        grounding_engine = GroundingEngine.from_ner_dataset(dev_dataset)
        tagger = MultiAgentTagger(
            dataset.entity_types, agent_config, llm_client, grounding_engine
        )
    else:
        tagger = MultiAgentTagger(dataset.entity_types, agent_config, llm_client)

    run_eval(tagger, dataset, output_file)


if __name__ == "__main__":
    # run_few_shot_eval()
    # run_multi_agent_eval()
    run_multi_agent_eval(enable_grounding=True)
