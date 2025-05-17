# AgenticNER

This repository hosts source code for my master's thesis ['Collaborative Multi-Agent Architecture for Domain-Agnostic Named Entity Recognition'](https://dspace.ut.ee/items/434e6981-09fe-4e1e-a65a-6b02173f4ad7). Instructions to run the benchmarks and reproduce the results from thesis can be found below.

---

## Installation

* Requirements:
  * Python 3.12+

* Install Poetry:
  ```sh
  curl -sSL https://install.python-poetry.org | python3 -
  ```

* Create a virtual environment and install the dependencies

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
```

## Running Benchmarks

The repo includes a command-line interface to run various benchmarks and evaluate different variants of AgenticNER.

### Prerequisites

* Get API key from [Anthropic](https://console.anthropic.com/) and export the API key

```bash
export ANTHROPIC_API_KEY="<your api key>"
```

* Get API key from [tavily.com](https://tavily.com/) to enable internet access for research agent and export the API key

```bash
export TAVILY_API_KEY="<your api key>"
```

* Set Anthropic API key in `config.yaml` to start LiteLLM proxy to create OpenAI compatible interface for Anthropic models. This is required by `AutoGen`.

* Start LiteLLM proxy in separate command line session

```bash
poetry run litellm --config config.yaml
```

* Export LiteLLM server address as OpenAI base URL

```bash
export OPENAI_BASE_URL=http://0.0.0.0:4000
```

### Command Structure

```bash
poetry run python src/ner/eval/run.py --benchmark <benchmark> --variant <variant> [--llm <llm>] [--sample-size <size>]
```

### Parameters

- `--benchmark`: Choose the benchmark dataset 
  - Options: `genia`, `music`, `buster`, `astro`
- `--variant`: Choose the evaluation variant
  - `few-shot`: Basic few-shot learning approach
  - `agentic-ner-no-grounding`: AgenticNER without grounding
  - `agentic-ner-grounding`: AgenticNER with grounding enabled
  - `agentic-ner-grounding-no-internet`: AgenticNER with grounding but no internet access
  - `agentic-ner-grounding-no-researcher`: AgenticNER with grounding but no researcher agent
- `--llm`: Choose the LLM model (default: haiku)
  - Options: `haiku`, `sonnet`
- `--sample-size`: Number of samples to evaluate (default: 500)

### Example Commands

```bash
# Run few-shot evaluation on GENIA using Haiku
poetry run python src/ner/eval/run.py --benchmark genia --variant few-shot

# Run full AgenticNER on MusicRecoNER using Sonnet
poetry run python src/ner/eval/run.py --benchmark music --variant agentic-ner-grounding --llm sonnet

# Run AgenticNER without internet on Buster with 100 samples
poetry run python src/ner/eval/run.py --benchmark buster --variant agentic-ner-grounding-no-internet --sample-size 100
```

### Reproducing results from the thesis

To reproduce the results from the thesis, run the following commands for each benchmark:

#### GENIA Benchmark
```bash
# Baseline (Few-shot single LLM call with Haiku)
poetry run python src/ner/eval/run.py --benchmark genia --variant few-shot

# Baseline (Few-shot single LLM call with Sonnet)
poetry run python src/ner/eval/run.py --benchmark genia --variant few-shot --llm sonnet

# AgenticNER variants
poetry run python src/ner/eval/run.py --benchmark genia --variant agentic-ner-no-grounding
poetry run python src/ner/eval/run.py --benchmark genia --variant agentic-ner-grounding
poetry run python src/ner/eval/run.py --benchmark genia --variant agentic-ner-grounding-no-internet
poetry run python src/ner/eval/run.py --benchmark genia --variant agentic-ner-grounding-no-researcher
poetry run python src/ner/eval/run.py --benchmark genia --variant agentic-ner-grounding --llm sonnet
```

#### MusicRecoNER Benchmark
```bash
poetry run python src/ner/eval/run.py --benchmark music --variant few-shot
poetry run python src/ner/eval/run.py --benchmark music --variant few-shot --llm sonnet
poetry run python src/ner/eval/run.py --benchmark music --variant agentic-ner-no-grounding
poetry run python src/ner/eval/run.py --benchmark music --variant agentic-ner-grounding
poetry run python src/ner/eval/run.py --benchmark music --variant agentic-ner-grounding-no-internet
poetry run python src/ner/eval/run.py --benchmark music --variant agentic-ner-grounding-no-researcher
poetry run python src/ner/eval/run.py --benchmark music --variant agentic-ner-grounding --llm sonnet
```

#### Buster Benchmark
```bash
poetry run python src/ner/eval/run.py --benchmark buster --variant few-shot
poetry run python src/ner/eval/run.py --benchmark buster --variant few-shot --llm sonnet
poetry run python src/ner/eval/run.py --benchmark buster --variant agentic-ner-no-grounding
poetry run python src/ner/eval/run.py --benchmark buster --variant agentic-ner-grounding
poetry run python src/ner/eval/run.py --benchmark buster --variant agentic-ner-grounding-no-internet
poetry run python src/ner/eval/run.py --benchmark buster --variant agentic-ner-grounding-no-researcher
poetry run python src/ner/eval/run.py --benchmark buster --variant agentic-ner-grounding --llm sonnet
```

#### AstroNER Benchmark
```bash
poetry run python src/ner/eval/run.py --benchmark astro --variant few-shot
poetry run python src/ner/eval/run.py --benchmark astro --variant few-shot --llm sonnet
poetry run python src/ner/eval/run.py --benchmark astro --variant agentic-ner-no-grounding
poetry run python src/ner/eval/run.py --benchmark astro --variant agentic-ner-grounding
poetry run python src/ner/eval/run.py --benchmark astro --variant agentic-ner-grounding-no-internet
poetry run python src/ner/eval/run.py --benchmark astro --variant agentic-ner-grounding-no-researcher
poetry run python src/ner/eval/run.py --benchmark astro --variant agentic-ner-grounding --llm sonnet
```

