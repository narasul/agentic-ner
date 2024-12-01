from ner.agents.multi_agent_tagger import MultiAgentTagger
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.clients.claude_oai_compatible_client import create_chat_completions_client
from ner.grounding import GroundingEngine
from ner.ontology import get_genia_ontology
from ner.eval.dataset import NERDataset
from ner.eval.eval import run_eval
from ner.prompts import get_agent_config, get_ner_prompt
from ner.tagger_few_shot import FewShotTagger


def run_few_shot_eval():
    print("Running few shot NER eval")
    dataset = NERDataset.from_genia("test").sample(500)
    dev_dataset = NERDataset.from_genia("train")
    ontology = get_genia_ontology()
    domain = "Biomedical, molecular biology, genomics"
    output_file = "genia_few_show_eval"

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


# Eval result:
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


def run_multi_agent_eval(enable_grounding: bool = False):
    print("Running multi-agent NER eval")
    dataset = NERDataset.from_genia("test").sample(500)
    dev_dataset = NERDataset.from_genia("train")
    examples = dev_dataset.get_examples(3)
    ontology = get_genia_ontology()
    domain = "Biomedical, molecular biology, genomics"
    output_file = "genia_few_show_eval"

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

    # Eval result without grounding:
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


#
if __name__ == "__main__":
    # run_few_shot_eval()
    # run_multi_agent_eval()
    run_multi_agent_eval(enable_grounding=True)

#            Old sample eval results with subset_size of 600
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
