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
  analyze_activations.py  # compute localization metrics from activation CSVs
  ablate_components.py    # disable selected components and measure loss changes

data/prompts/
  agriculture.txt
  astronomy.txt
  math.txt
  medicine.txt
  politics.txt

data/prompt_sets/
  seed.jsonl  # metadata-rich prompt records
  seed_qa.jsonl  # metadata-rich prompt records with target answers
  qa_v1.jsonl  # deterministic balanced QA dataset, generated from blueprints
```

## Quick Start

Activate the local virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Optional: install a CUDA-enabled PyTorch build on machines with an NVIDIA GPU:

```powershell
pip uninstall torch -y
pip install -r requirements-cuda-cu124.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Run a generation sanity check:

```powershell
python main.py
```

Collect MLP activation summaries for the seed prompt set:

```powershell
python activations.py
```

Use the metadata-rich prompt set:

```powershell
python activations.py --prompt-jsonl data/prompt_sets/seed.jsonl --split discovery
python activations.py --prompt-jsonl data/prompt_sets/qa_v1.jsonl --split discovery
```

Choose a measurement level:

```powershell
# Layer-wise hidden-state summaries.
python activations.py --granularity layer --output-csv results/layer_activations.csv

# MLP-block summaries.
python activations.py --granularity mlp_module --output-csv results/mlp_module_activations.csv

# MLP neuron proxy: gate projection units. Use top-k for small exploratory CSVs.
python activations.py --granularity mlp_neuron --top-units-per-module 32 --output-csv results/mlp_neuron_top32.csv

# Attention-head proxy: per-head vectors before the attention output projection.
python activations.py --granularity attention_head --output-csv results/attention_head_activations.csv
```

The activation experiment writes a CSV file to `results/activations.csv` by default and prints simple domain selectivity rankings.

Analyze a collected activation CSV:

```powershell
python experiments/analyze_activations.py --input-csv results/activations.csv --output-dir results/analysis
```

The analysis writes component scores, concentration scores, and domain decodability results.

Run a first ablation:

```powershell
python experiments/ablate_components.py --prompt-jsonl data/prompt_sets/seed_qa.jsonl --component-scores results/analysis_mlp_modules/component_scores.csv --granularity mlp_module --target-domain agriculture --component-count 5 --random-control-repeats 10 --output-dir results/ablation_agriculture_mlp_modules
```

Run tests:

```powershell
python -m unittest discover -s tests
```

Validate prompt metadata:

```powershell
python experiments/validate_prompts.py --prompt-jsonl data/prompt_sets/seed.jsonl
python experiments/validate_prompts.py --prompt-jsonl data/prompt_sets/seed_qa.jsonl --require-targets
python experiments/validate_prompts.py --prompt-jsonl data/prompt_sets/qa_v1.jsonl --require-targets
```

Regenerate the deterministic QA v1 dataset:

```powershell
python experiments/build_qa_v1.py --output data/prompt_sets/qa_v1.jsonl
```

## Research Notes

The prompt files in `data/prompts/` are only a small seed set. They are useful for testing the code path, not for making scientific claims. A serious experiment needs larger, balanced, held-out prompt sets and causal validation through ablation or pruning.

See [docs/experiment_plan.md](docs/experiment_plan.md) for the first research roadmap.
See [docs/metrics.md](docs/metrics.md) for the metric definitions.
See [docs/prompt_dataset_protocol.md](docs/prompt_dataset_protocol.md) for the prompt collection protocol.
See [docs/ablation.md](docs/ablation.md) for ablation details and caveats.
