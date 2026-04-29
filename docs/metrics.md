# Metrics

The first experiments use activation summaries as evidence for possible domain localization. These metrics are descriptive first; causal evidence comes later through ablation and pruning.

## Component Selectivity

For a component `c` and target domain `d`:

```text
selectivity(c, d) =
  (mean_activation(c, d) - mean_activation(c, not d))
  /
  (abs(mean_activation(c, d)) + abs(mean_activation(c, not d)) + eps)
```

Interpretation:

- Positive values mean the component is more active for the target domain.
- Values close to zero mean weak domain preference.
- Negative values mean the component is less active for the target domain than for other domains.

## Effect Size

Effect size normalizes the difference by activation variability:

```text
effect_size(c, d) =
  (mean_activation(c, d) - mean_activation(c, not d))
  /
  pooled_std(c, d vs. not d)
```

This is useful because raw activation differences can be larger in some layers or component types.

If there is not enough variability to estimate a pooled standard deviation, the current implementation reports effect size as `0.0`. In practice, effect size should only be interpreted after collecting multiple prompts per domain.

## Concentration

Concentration asks whether domain signal is spread across many components or concentrated in a small subset.

The analysis currently uses only positive scores for a target domain, because those are the components that are more active for that domain than for others.

### Entropy

Entropy treats positive component scores as a distribution:

```text
p_i = score_i / sum(score)
entropy = -sum(p_i * log(p_i))
```

Low entropy suggests concentrated signal. High entropy suggests distributed signal.

The normalized entropy is:

```text
normalized_entropy = entropy / log(number_of_components)
```

This ranges from `0` to `1` when there is positive signal:

- closer to `0`: more localized
- closer to `1`: more distributed

The effective number of components is:

```text
effective_components = exp(entropy)
```

### Top-k Share

Top-k share measures how much positive signal is carried by the highest-ranked components:

```text
top_5_share = sum(top 5 percent positive scores) / sum(all positive scores)
```

Large top-k shares suggest stronger localization.

### Gini

Gini measures inequality of positive component scores:

- closer to `0`: scores are evenly spread
- closer to `1`: scores are concentrated in a few components

## Domain Decodability

Domain decodability asks whether the domain can be predicted from activation vectors.

The current implementation uses a leave-one-out nearest-centroid classifier:

1. Represent each prompt as a vector of component activation means.
2. Hold out one prompt.
3. Compute one centroid per domain from the remaining prompts.
4. Predict the held-out prompt's domain using nearest centroid distance.

This is deliberately simple and dependency-free. It should be treated as a sanity check, not as final evidence of localization.

## Recommended Interpretation

Good evidence for localization requires more than one metric:

- high selectivity or effect size for a small set of components
- low normalized entropy
- high top-k share
- high decodability using a limited component set
- causal performance drops when the selected components are ablated

The last point is critical. Activation metrics can suggest where to look, but ablation is needed to test whether the components matter.

For ablation, prefer target-answer loss over prompt loss. Prompt loss measures how well the model predicts the question text. Target-answer loss measures how well the model predicts the answer conditioned on the question, which is closer to the domain-knowledge question.
