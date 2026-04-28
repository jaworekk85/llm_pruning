# LLM Pruning Experiments

This repository explores whether domain-specific behavior in language models is localized in identifiable model components, such as layers, MLP blocks, neurons, or attention heads.

The current experimental model is `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. The first milestone is to collect and compare activation statistics across domains before attempting causal ablation or pruning.

## Project Layout

```text
src/llm_pruning/
  models.py      # model/tokenizer loading and prompt formatting
  prompts.py     # prompt dataset loading
  hooks.py       # activation collection hooks
  metrics.py     # simple domain selectivity summaries

experiments/
  generate_text.py        # small generation sanity check
  collect_activations.py  # collect domain activation statistics

data/prompts/
  agriculture.txt
  astronomy.txt
  math.txt
  medicine.txt
  politics.txt
```

## Quick Start

Activate the local virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Run a generation sanity check:

```powershell
python main.py
```

Collect MLP activation summaries for the seed prompt set:

```powershell
python activations.py
```

The activation experiment writes a CSV file to `results/activations.csv` by default and prints simple domain selectivity rankings.

## Research Notes

The prompt files in `data/prompts/` are only a small seed set. They are useful for testing the code path, not for making scientific claims. A serious experiment needs larger, balanced, held-out prompt sets and causal validation through ablation or pruning.
