# https://arxiv.org/abs/2405.02602 - dataset source

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

    # Eval result, Haiku 3.5:
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

    # Eval result, Sonnet 3.5:
    #                   precision    recall  f1-score   support
    #
    #       AstrObject       0.31      0.71      0.43        14
    #     AstroPortion       0.71      0.33      0.45        15
    #  ChemicalSpecies       0.39      0.37      0.38        93
    #       Instrument       0.73      0.34      0.47        79
    #      Measurement       0.16      0.07      0.09        46
    #           Method       0.52      0.46      0.49       332
    #       Morphology       0.18      0.12      0.15        32
    # PhysicalQuantity       0.12      0.12      0.12        48
    #          Process       0.45      0.39      0.42       141
    #          Project       0.15      0.64      0.25        14
    #  ResearchProblem       0.27      0.30      0.28       361
    #   SpectralRegime       0.35      0.53      0.42        17
    #
    #        micro avg       0.37      0.35      0.36      1192
    #        macro avg       0.36      0.37      0.33      1192
    #     weighted avg       0.39      0.35      0.36      1192
    #


def run_multi_agent_eval(
    enable_grounding: bool = False,
    sonnet: bool = False,
    internet_access: bool = True,
    researcher: bool = True,
):
    print("Running multi-agent NER eval")
    output_file = "astro_ner_multi_agent_eval"

    if not researcher:
        agent_config = get_agent_config_no_researcher(
            domain, ontology, dev_dataset.get_examples(12)
        )
    elif internet_access:
        agent_config = get_agent_config(domain, ontology, dev_dataset.get_examples(12))
    else:
        agent_config = get_agent_config_no_internet(
            domain, ontology, dev_dataset.get_examples(12)
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

    # Eval result with grounding, Haiku 3.5:
    #                   precision    recall  f1-score   support
    #
    #       AstrObject       0.17      0.64      0.26        14
    #     AstroPortion       0.50      0.27      0.35        15
    #  ChemicalSpecies       0.20      0.13      0.16        93
    #       Instrument       0.72      0.35      0.47        79
    #      Measurement       0.11      0.11      0.11        46
    #           Method       0.49      0.48      0.49       332
    #       Morphology       0.15      0.12      0.14        32
    # PhysicalQuantity       0.13      0.08      0.10        48
    #          Process       0.38      0.23      0.29       141
    #          Project       0.14      0.64      0.23        14
    #  ResearchProblem       0.26      0.27      0.26       361
    #   SpectralRegime       0.32      0.53      0.40        17
    #
    #        micro avg       0.33      0.31      0.32      1192
    #        macro avg       0.30      0.32      0.27      1192
    #     weighted avg       0.35      0.31      0.32      1192

    # Eval result without grounding, Haiku 3.5:
    #                   precision    recall  f1-score   support
    #
    #       AstrObject       0.17      0.64      0.26        14
    #     AstroPortion       0.44      0.27      0.33        15
    #  ChemicalSpecies       0.24      0.17      0.20        93
    #       Instrument       0.71      0.37      0.48        79
    #      Measurement       0.11      0.11      0.11        46
    #           Method       0.48      0.49      0.49       332
    #       Morphology       0.16      0.12      0.14        32
    # PhysicalQuantity       0.11      0.06      0.08        48
    #          Process       0.35      0.21      0.26       141
    #          Project       0.14      0.64      0.23        14
    #  ResearchProblem       0.25      0.25      0.25       361
    #   SpectralRegime       0.30      0.53      0.38        17
    #
    #        micro avg       0.32      0.31      0.32      1192
    #        macro avg       0.29      0.32      0.27      1192
    #     weighted avg       0.34      0.31      0.32      1192

    # Eval result, with grounding, no internet access, Haiku 3.5:
    #                   precision    recall  f1-score   support
    #
    #       AstrObject       0.15      0.57      0.24        14
    #     AstroPortion       0.29      0.13      0.18        15
    #  ChemicalSpecies       0.26      0.17      0.21        93
    #       Instrument       0.66      0.34      0.45        79
    #      Measurement       0.11      0.11      0.11        46
    #           Method       0.48      0.48      0.48       332
    #       Morphology       0.19      0.16      0.17        32
    # PhysicalQuantity       0.10      0.06      0.08        48
    #          Process       0.35      0.21      0.26       141
    #          Project       0.14      0.64      0.23        14
    #  ResearchProblem       0.25      0.25      0.25       361
    #   SpectralRegime       0.33      0.53      0.41        17
    #
    #        micro avg       0.32      0.31      0.31      1192
    #        macro avg       0.27      0.30      0.25      1192
    #     weighted avg       0.34      0.31      0.31      1192

    # Eval result, with grounding, without researcher, Haiku 3.5:
    #                   precision    recall  f1-score   support
    #
    #       AstrObject       0.14      0.64      0.23        14
    #     AstroPortion       0.38      0.33      0.36        15
    #  ChemicalSpecies       0.20      0.14      0.16        93
    #       Instrument       0.66      0.32      0.43        79
    #      Measurement       0.10      0.11      0.10        46
    #           Method       0.51      0.44      0.47       332
    #       Morphology       0.13      0.12      0.13        32
    # PhysicalQuantity       0.12      0.06      0.08        48
    #          Process       0.35      0.16      0.22       141
    #          Project       0.13      0.57      0.21        14
    #  ResearchProblem       0.25      0.29      0.27       361
    #   SpectralRegime       0.31      0.53      0.39        17
    #
    #        micro avg       0.31      0.30      0.30      1192
    #        macro avg       0.27      0.31      0.25      1192
    #     weighted avg       0.34      0.30      0.31      1192

    # Eval result with grounding, Sonnet 3.5:
    #                   precision    recall  f1-score   support
    #
    #       AstrObject       0.18      0.64      0.28        14
    #     AstroPortion       0.00      0.00      0.00        15
    #  ChemicalSpecies       0.44      0.23      0.30        93
    #       Instrument       0.78      0.46      0.58        79
    #      Measurement       0.22      0.11      0.14        46
    #           Method       0.52      0.44      0.47       332
    #       Morphology       0.14      0.09      0.11        32
    # PhysicalQuantity       0.10      0.12      0.11        48
    #          Process       0.45      0.38      0.41       141
    #          Project       0.19      0.64      0.30        14
    #  ResearchProblem       0.27      0.32      0.29       361
    #   SpectralRegime       0.32      0.47      0.38        17
    #
    #        micro avg       0.36      0.34      0.35      1192
    #        macro avg       0.30      0.32      0.28      1192
    #     weighted avg       0.39      0.34      0.36      1192


if __name__ == "__main__":
    # run_few_shot_eval()
    # run_multi_agent_eval()
    run_multi_agent_eval(enable_grounding=True)
    # run_few_shot_eval(sonnet=True)
    # run_multi_agent_eval(enable_grounding=True, sonnet=True)

    # run_multi_agent_eval(enable_grounding=False)
    # run_multi_agent_eval(enable_grounding=True, internet_access=False)
    # run_multi_agent_eval(enable_grounding=True, researcher=False)
