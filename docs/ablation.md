# Ablation Experiments

Ablation tests whether components that look domain-selective are causally important.

Activation analysis asks:

> Which components behave differently for a domain?

Ablation asks:

> What changes when those components are disabled?

## Current Implementation

The ablation script selects top-ranked components from an analysis CSV, disables them during inference, and compares language-model loss against:

- baseline model
- top target-domain components ablated
- repeated random control component sets ablated

Example:

```powershell
python experiments/ablate_components.py `
  --prompt-jsonl data/prompt_sets/seed_qa.jsonl `
  --component-scores results/analysis_mlp_modules/component_scores.csv `
  --granularity mlp_module `
  --target-domain agriculture `
  --component-count 5 `
  --random-control-repeats 10 `
  --output-dir results/ablation_agriculture_mlp_modules
```

Supported granularities:

- `mlp_module`: zero the output of selected MLP blocks.
- `mlp_neuron`: zero selected `mlp.gate_proj` output units.
- `attention_head`: zero selected head vectors before `self_attn.o_proj`.

## Outputs

Each run writes:

```text
components.csv  # selected top-target and random-control components
losses.csv      # per-prompt loss values
summary.csv     # mean loss and delta from baseline per domain/condition
contrasts.csv   # top-target deltas compared against random-control deltas
```

## Interpretation

The key quantity is `delta_loss`:

```text
delta_loss = ablated_mean_loss - baseline_mean_loss
```

If agriculture-selective components are causally important for agriculture, then ablating them should increase agriculture loss more than:

- loss on other domains
- random control ablations
- components selected for unrelated domains

`contrasts.csv` reports:

```text
top_minus_random_delta_loss =
  top_target_delta_loss - mean(random_control_delta_loss)
```

For the target domain, a positive value means the selected domain components caused more damage than average random controls. This is stronger than comparing to a single random sample.

It also reports:

```text
target_specificity_over_other_domains =
  target_domain_top_delta - mean(other_domain_top_deltas)
```

This asks whether the ablation hurts the target domain more than it hurts other domains.

## Important Caveat

The preferred path is answer-target loss. The prompt is used as context, but only the target answer tokens are scored.

Question-only prompt loss is still available with `--loss-scope prompt`, but it is only useful for legacy smoke tests and is not sufficient for a paper claim about domain knowledge.

Answer-bearing evaluation records look like:

```json
{
  "prompt": "What is crop rotation?",
  "target": "Crop rotation is the practice of growing different crops..."
}
```

The implementation masks the prompt tokens in the labels, so the loss is computed only over the answer.

## Current Protocol

For a serious run:

1. Select components using activation analysis on the discovery split.
2. Evaluate ablation on validation or test records with target answers.
3. Use repeated random controls, at least 10 and preferably more.
4. Compare `top_minus_random_delta_loss`.
5. Check `target_specificity_over_other_domains`.

## Next Improvements

- Add matched random controls from the same layer range.
- Add many more answer-bearing prompt records.
- Add ablation of components selected on discovery split and evaluated on validation/test splits.
- Add confidence intervals or bootstrap estimates over prompts.
