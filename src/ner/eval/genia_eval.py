from ner.agents.multi_agent_tagger import MultiAgentTagger
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
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


def run_multi_agent_eval():
    print("Running multi-agent NER eval")
    dataset = NERDataset.from_genia("test").sample(500)
    dev_dataset = NERDataset.from_genia("train")
    examples = dev_dataset.get_examples(3)
    ontology = get_genia_ontology()
    domain = "Biomedical, molecular biology, genomics"
    output_file = "genia_few_show_eval"

    agent_config = get_agent_config(domain, ontology, examples)

    llm_client = AnthropicClient(ClaudeFamily.HAIKU_35)
    tagger = MultiAgentTagger(
        llm_client,
        dataset.entity_types,
        agent_config,
    )

    run_eval(tagger, dataset, output_file)


if __name__ == "__main__":
    run_few_shot_eval()

    # TODO: change the sleep to 1 or 2 seconds from 5 seconds
    # run_multi_agent_eval()

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
