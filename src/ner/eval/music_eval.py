# eval for: https://github.com/deezer/music-ner-eacl2023/tree/mai://github.com/deezer/music-ner-eacl2023/tree/main

from ner.agents.multi_agent_tagger import MultiAgentTagger
from ner.clients.claude_client import AnthropicClient, ClaudeFamily
from ner.eval.dataset import NERDataset
from ner.eval.eval import run_eval
from ner.prompts import get_agent_config, get_ner_prompt
from ner.tagger_few_shot import FewShotTagger


def run_few_shot_eval():
    print("Running few shot NER eval")
    dataset = NERDataset.from_musicner("test").sample(500)
    dev_dataset = NERDataset.from_musicner("train")
    ontology = get_musicner_ontology()
    domain = "Music and entertainment"
    output_file = "musicner_few_show_eval"

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
    dataset = NERDataset.from_musicner("test").sample(500)
    dev_dataset = NERDataset.from_musicner("train")
    examples = dev_dataset.get_examples(3)
    ontology = get_musicner_ontology()
    domain = "Music and entertainment"
    output_file = "musicner_few_show_eval"

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

    # run_multi_agent_eval()
