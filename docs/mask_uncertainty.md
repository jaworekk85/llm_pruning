# Mask budgets and uncertainty

## Configure a run

The causal runner exposes:

- `--control-repeats R`: requested **unique** masks per strategy; capped at the
  exact finite population. Default 3 is an engineering pilot, not a power calculation.
- `--bootstrap-samples B` (default 2000) and `--confidence` (default 0.95).
- `--multiplicity bonferroni` (default) or `none`. Default comparison family is
  target values x control strategies x three metrics. `--comparison-count` can
  specify a larger preregistered family across models/budgets/granularities.
- `--desired-control-half-width H` with `--planning-metric` (default
  `loss_increase`): estimate a future mask budget from validation-pilot variance.
  Accuracy uses fraction units, loss uses nats. This does not adaptively stop a run.

For example, add `--control-repeats 30 --bootstrap-samples 5000
--desired-control-half-width 0.02` to the causal command. These numbers illustrate
parameters, not a scientifically sufficient design.

Controls are sampled uniformly without replacement using combinatorial ranks;
large mask spaces are not enumerated. `conditions.json` records requested,
available and sampled counts, and whether the pool was exhausted. Identical
masks across strategies share an inference result. Increasing the budget with
the same seed does not promise the previous draw is a prefix.

## Analyze saved results without a model

```powershell
venv/Scripts/python.exe experiments/analyze_mask_uncertainty.py --source-run results/icml2027_mmlu_component_causal_v1 --experiment-name causal_precision_v1 --bootstrap-samples 5000 --desired-control-half-width 0.02
```

The analysis creates a fresh directory with config, manifest, source fingerprints,
`summary.md` and `uncertainty.json`. Source results stay unchanged. No weights,
datasets or model inference are needed. The causal runner also writes this report
automatically. Legacy repeated masks are collapsed; unknown sampling metadata
is flagged and receives no finite-population correction.

## Meaning and limits

The estimand is selected damage minus mean control damage, conditional on the
mask chosen on discovery. Bootstrap draws resample example rows with the same
weights for selected and every control, and whole mask columns shared across
examples. Reports contain example-only, mask-only and joint percentile intervals.
Small example/mask counts, degenerate observed spread and poorly resolved
bootstrap tails are flagged. A single non-exhaustive mask cannot estimate mask
uncertainty; a single example cannot estimate example uncertainty. Such joint
intervals are null, not zero-width evidence of precision.

Sampling without replacement uses approximate finite-population variance
scaling. An exhaustive mask pool has zero mask-sampling error on the observed
examples, while example uncertainty remains. Bonferroni-adjusted interval levels
address the declared comparison family; they do not make bootstrap intervals exact.
Assumptions include independent evaluation examples. Shared passages/facts need
cluster-aware resampling, not treating correlated questions as independent.

Validation-only planning uses observed sample variance of per-mask mean damage:
`ceil(z^2*s^2 / (H^2 + z^2*s^2/K))`, capped at population K (omit the K term when
unknown). It targets **mask-mean precision on these examples**, not total effect
precision or statistical power. Zero pilot variance is flagged and is not proof
of zero population variance. Test reports do not suggest new budgets; freeze
budgets using validation before test. There are no permutation p-values,
guaranteed significance, optional significance stopping or automatic power claims.
