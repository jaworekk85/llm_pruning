# Independent pruning and organization diagnostics

These are shared tools, independent of the ICML hypothesis. They currently evaluate
MC records with explicit domain/language/capability/task labels. Offline tests use
random-init models; no pretrained research result has been produced.

## Wanda and specialist frontiers

`src/llm_pruning/wanda.py` implements the standard row-wise importance
`abs(weight) * L2(input activation)`, with stable column-index tie breaking.
It follows [the authors' sequential-block algorithm](https://github.com/locuslab/wanda/blob/main/lib/prune.py):
collect norms within a block, prune its projections, propagate the pruned output
to the next block. Each output row loses `floor(input_width * sparsity)` entries
(with a small floating-point rounding tolerance). Norms exclude padding; variable
length calibration uses a common token-sum scale, which leaves row rankings unchanged.
Decoder q/k/v/o and gate/up/down weights are eligible; biases, embeddings, norms
and the LM head stay dense. Supported: ordinary single-device HF Llama/Qwen2 with
full attention. Quantized, mixed/sliding attention and distributed layouts fail.

Calibration uses **target-only discovery prompts**, including MC options but no
correct-answer continuation. Evaluation uses validation/test records and the same
prompt policy. Dataset-wide ID/input split checks happen before sampling. No
activation ranking or causal probe result enters Wanda.

```powershell
python experiments/run_compression_frontier.py --config configs/experiments/icml2027/compression_frontier.json --validate-only
python experiments/run_compression_frontier.py --config configs/experiments/icml2027/compression_frontier.json --budgets 1 .9 .75 .6 .5 --calibration-records 32
```

Use the existing venv Python on Windows. Preflight performs no model loading.
The example config is a small validation template with local-only loading; pin
`revision` and freeze budgets, calibration sampling, prompt policy and quality
thresholds before a test run. Change `experiment_name` for every run. Repeat
calibration with several seeds; one seed is not a robustness study.

`frontier.csv` has per-value, pooled non-target and optional core panels. Core
requires explicit `--core-values` labels on the configured target field; it is not
automatically a general language/reasoning benchmark. A separate orthogonal core
dataset is a future extension. Aggregates give each example equal weight.
`--max-accuracy-drop` (default .02 absolute) and optional `--core-min-accuracy`
produce the **minimum observed feasible budget** in `feasibility.json`; this is
descriptive, with no confidence/power guarantee. No test-driven adaptive search.

Each budget starts dense, recalibrates sequentially, and restores weights even
after failure. Requested budgets refer to eligible projection weights; realized
nonzero fractions are reported for those weights and for the whole model. Row
rounding and preexisting zeros can change the realized budget. Masks at different
budgets need not be nested. CPU backups require roughly the eligible weight size;
calibration hidden states/decoder arguments are also cached on CPU.

Dense parameter count, parameter bytes and actual resident tensor storage bytes
are recorded separately from nonzeros. Zeroing does not shrink the dense model.
Storage excludes temporary tensors and CPU backups. On CUDA, an additional
evaluation-only PyTorch allocation peak is measured; it excludes pruning peak and
does not equal total process memory. FLOPs, latency and energy are not measured.
Per-example decisions/deltas, sampled metadata, projection zero-pattern hashes,
model revision, versions and protocol are saved. A single-pruner curve is an
empirical lower bound, not the optimal specialist compression frontier.

## Organization: activations and causal probes

Activations already come from `collect_component_activations.py`. Single-component
causal measurements use the discovery-only runner:

```powershell
python experiments/probe_component_importance.py --config configs/experiments/icml2027/component_probes.json --components layer.0.attn.head.0 layer.0.attn.head.1 --validate-only
python experiments/probe_component_importance.py --config configs/experiments/icml2027/component_probes.json --components layer.0.attn.head.0 layer.0.attn.head.1
python experiments/analyze_component_organization.py --source-runs results/icml2027_mmlu_discovery_probes_v1 --experiment-name discovery_diagnostics_v1 --top-k 5
```

Omitting `--components` probes every component at the selected granularity:
inference grows approximately with component count. No large probes were run.
Each mask contains one component, and all use the same examples/baseline. Combined
mask effects cannot be decomposed into per-component importance. Single-component
effects are conditional on the intact network and need not add up.

The offline analyzer accepts completed discovery activation and/or probe runs:
- Signed activation selectivity `(target_mean - other_mean)/(target_mean + other_mean + eps)`.
- Signed causal damage and target-minus-pooled-other damage, for an explicit metric.
- Positive-score entropy, normalized entropy, effective count, Gini including zero
  mass, and top-k share; null if no positive signal.
- Positive top-k Jaccard, normalized weighted overlap and signed cosine.
- Across matching repeated runs: top-k/weighted overlap, cosine and Spearman with
  average tied ranks. Same model/revision/dtype/inventory/prompt protocol required;
  datasets/seeds may differ. Identical measurement files are flagged.

Heads and MLP groups are separate universes. Partial probes report their measured
and full inventory counts; their concentration does not describe the full model.
Coverage, labels, finite values and singleton conditions are validated. Stability
needs multiple runs; these descriptive outputs have no inferential intervals.
Primitive scores/means stay in `component_scores.json`; definitions, concentration,
overlap and stability are in `diagnostics.json`. Source files remain untouched.
