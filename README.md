# LLM Pruning Research Framework

This repository studies how knowledge, capabilities, domains and languages are organized inside pretrained LLMs, and how that organization relates to causal interventions, specialist compression and adaptation. Localization and compressibility are questions to test, not assumptions built into the framework.

ICML 2027 is the primary publication objective: first screen whether pruning impairs skill composition beyond constituent-skill loss. Predicting specialist compressibility remains a secondary candidate. Planned UCT multilingual and Waikato streaming collaborations share tools while keeping independent questions. Priorities and decision gates: [research strategy](docs/research_strategy.md).

Current code includes activation collection, component rankings, causal ablations, MC likelihood evaluation, and linked short-answer composition screens with independent Wanda pruning. Phase 1 is complete locally: data builder/importer, evaluator, controls, paired analysis and server config/preflight are available. Pretrained validation and targeted restoration remain outstanding; offline random models provide engineering checks only.

Current composition task candidate: [controlled benchmark design](docs/composition_benchmark_design.md), with connected lookup and simple-addition families, shared evidence and paired input probes. Older station-count tasks are engineering exercises. Scientific dataset acceptance requires pretrained baseline behavior; the runner supports `--baseline-only`.
For a fresh GPU server, start with [server handoff](docs/server_handoff.md): environment,
regeneration of ignored datasets, pinned model/tokenizer and dense baseline before pruning.

Start with [STATUS.md](STATUS.md) and [docs/code_map.md](docs/code_map.md). The new [target baseline pipeline](docs/projects/icml2027.md) evaluates domains, languages, capabilities or tasks from a JSON configuration and saves reproducible run artifacts.

[Wanda frontiers and organization diagnostics](docs/pruning_and_organization.md) add independent target-calibrated pruning, held-out quality/cost curves and discovery activation/causal concentration, overlap and stability. Dense storage and nonzero weights are measured separately. These tools have been checked on tiny random-init models, without pretrained research runs.

## Config-driven Target Baseline

For the current linked-task pilot, use the build/preflight/server instructions in
[docs/projects/icml2027.md](docs/projects/icml2027.md). The commands below evaluate MC targets.

Validate the example config and all dataset splits without loading a model:

```powershell
python experiments/icml2027/run_target_baseline.py --config configs/experiments/icml2027/target_baseline.json --validate-only
```

To evaluate, remove `--validate-only`. The example uses cached TinyLlama weights and a small validation sample; model downloads are disabled by default. Choose a fresh `experiment_name` for each run. Outputs include `manifest.json`, `config.json`, `summary.md`, per-value metrics, detailed candidate/decision CSVs and sampled record IDs. Existing result directories are rejected.

This runner currently supports multiple-choice records. Language/capability/task labels must be provided explicitly in the dataset. The legacy commands below remain available; their result handling has not yet migrated to manifests and overwrite protection.

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

Install the test dependency in the active environment and run offline tests:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q -x
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

Use [docs/research_strategy.md](docs/research_strategy.md) for current priorities and [docs/projects/icml2027.md](docs/projects/icml2027.md) for the ICML pilot. [docs/experiment_plan.md](docs/experiment_plan.md) preserves the historical first roadmap.
See [docs/metrics.md](docs/metrics.md) for the metric definitions.
See [docs/prompt_dataset_protocol.md](docs/prompt_dataset_protocol.md) for the prompt collection protocol.
See [docs/ablation.md](docs/ablation.md) for ablation details and caveats.
