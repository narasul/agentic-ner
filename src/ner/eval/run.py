import click
from typing import Tuple, Callable

from ner.eval.genia_eval import run_few_shot_eval as run_genia_few_shot
from ner.eval.genia_eval import run_multi_agent_eval as run_genia_multi_agent
from ner.eval.music_eval import run_few_shot_eval as run_music_few_shot
from ner.eval.music_eval import run_multi_agent_eval as run_music_multi_agent
from ner.eval.buster_eval import run_few_shot_eval as run_buster_few_shot
from ner.eval.buster_eval import run_multi_agent_eval as run_buster_multi_agent
from ner.eval.astro_eval import run_few_shot_eval as run_astro_few_shot
from ner.eval.astro_eval import run_multi_agent_eval as run_astro_multi_agent

BENCHMARKS = ["genia", "music", "buster", "astro"]
VARIANTS = [
    "few-shot",
    "agentic-ner-no-grounding",
    "agentic-ner-grounding",
    "agentic-ner-grounding-no-internet",
    "agentic-ner-grounding-no-researcher",
]
LLM_MODELS = ["haiku", "sonnet"]


def get_benchmark_runners(benchmark: str) -> Tuple[Callable, Callable]:
    """Get the appropriate few-shot and multi-agent runners for a benchmark."""
    runners = {
        "genia": (run_genia_few_shot, run_genia_multi_agent),
        "music": (run_music_few_shot, run_music_multi_agent),
        "buster": (run_buster_few_shot, run_buster_multi_agent),
        "astro": (run_astro_few_shot, run_astro_multi_agent),
    }
    return runners[benchmark]


@click.command()
@click.option(
    "--benchmark",
    type=click.Choice(BENCHMARKS),
    required=True,
    help="Benchmark dataset to evaluate on",
)
@click.option(
    "--variant",
    type=click.Choice(VARIANTS),
    required=True,
    help="Evaluation variant to run",
)
@click.option(
    "--llm",
    type=click.Choice(LLM_MODELS),
    default="haiku",
    help="LLM model to use (default: haiku)",
)
@click.option(
    "--sample-size",
    type=int,
    default=500,
    help="Number of samples to evaluate (default: 500)",
)
def run(benchmark: str, variant: str, llm: str, sample_size: int):
    """Run NER evaluation for specified benchmark and variant.

    Examples:
        python run.py --benchmark genia --variant few-shot --llm haiku
        python run.py --benchmark music --variant agentic-ner-grounding --llm sonnet --sample-size 100
    """
    use_sonnet = llm == "sonnet"
    click.echo(f"Running {variant} evaluation on {benchmark} benchmark")
    click.echo(f"Using {llm.upper()} model with sample size {sample_size}")

    few_shot_runner, multi_agent_runner = get_benchmark_runners(benchmark)

    if variant == "few-shot":
        few_shot_runner(sonnet=use_sonnet, sample_size=sample_size)

    elif variant == "agentic-ner-no-grounding":
        multi_agent_runner(
            enable_grounding=False, sonnet=use_sonnet, sample_size=sample_size
        )

    elif variant == "agentic-ner-grounding":
        multi_agent_runner(
            enable_grounding=True, sonnet=use_sonnet, sample_size=sample_size
        )

    elif variant == "agentic-ner-grounding-no-internet":
        multi_agent_runner(
            enable_grounding=True,
            internet_access=False,
            sonnet=use_sonnet,
            sample_size=sample_size,
        )

    elif variant == "agentic-ner-grounding-no-researcher":
        multi_agent_runner(
            enable_grounding=True,
            researcher=False,
            sonnet=use_sonnet,
            sample_size=sample_size,
        )


if __name__ == "__main__":
    run()
