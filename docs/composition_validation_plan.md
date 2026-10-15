# Composition uncertainty and confirmation plan

This file owns statistical design. Task semantics and implemented input controls:
[benchmark design](composition_benchmark_design.md). Setup/commands:
[server handoff](server_handoff.md). All size numbers below are design scenarios;
no pretrained model results have been observed. The earlier station-count review
is [preserved in Git](README.md#historical-versions).

## Estimands and confirmation rule

For each episode and variant v, define D_v = dense_correct_v − pruned_correct_v.
Positive D is harm, but improvements are retained as negative values. The primary
contrast Z = D_composed − D_staged lies in [-2, 2]; A/B damage lies in [-1, 1].
Use all linked cases and resample whole source groups jointly across variants and
conditions. Z measures change in the direct/staged performance gap; it does not
uniquely identify failed neural composition. Oracle/B and prompt controls are
required to interpret it. Do not multiply separate A/B accuracies as an error null.

Choose ONE primary model, retained budget and calibration condition on validation.
One proposed frozen confirmation family has three one-sided claims: extra gap
greater than .10, A damage below .05 and B damage below .05. Allocate familywise
alpha .05 across the three (Bonferroni is sufficient without independence), or
pre-specify another justified joint procedure. Report all secondary grid points;
if multiple primary conditions are kept, include them in the family. An estimate
of exactly .10 cannot reliably certify a lower confidence bound greater than .10.

The existing runner's intervals are exploratory, unadjusted percentile bootstrap;
its flags are point estimates. They do not implement the above confirmation rule.
Implement and bind the chosen confirmation analysis before freezing a scientific
test. Bootstrap intervals can have zero width when the pilot shows no variation;
that does not prove zero population damage. In that case use a justified finite-
sample bound or exact discrete analysis rather than claiming equivalence.

## Offline size scenarios

Reproduce with a fresh output directory:

```powershell
python experiments/icml2027/plan_composition_precision.py --output-dir results/composition_precision_plan_new --groups 32 128 512 1024 2048 --comparisons 3 --half-width 0.05
```

Available artifact: `results/composition_precision_plan_20261013_v2/` (manifest,
plan.json, summary). The calculator assumes one episode per independent source
group and a fixed model/mask protocol. Different group structures need a cluster
design; repeated calibration seeds are not additional independent cases.

| Independent cases | Approx. simultaneous Z half-width, assumed SD=.5 / 1 | Upper probability of a worsened A or B item if zero observed, per allocated claim |
|---|---|---|
| 32 | .212 / .423 | .120 |
| 128 | .106 / .212 | .031 |
| 512 | .053 / .106 | .008 |
| 1024 | .037 / .075 | .004 |
| 2048 | .026 / .053 | .002 |

Normal scenarios use z_(1−alpha/(2m))·SD/√n, with m=3. Assumed SD is not a pilot
estimate or a power guarantee. For expected Z=.20 versus meaningful margin .10,
80% approximate power for the Z claim alone needs 221 groups at SD=.5 or 882 at
SD=1, using the one-sided normal approximation. This is not 80% joint power for
all three claims. Smaller effect-to-margin separation needs more.

The zero-worsened bound is 1−(alpha/m)^(1/n). It concerns the probability of a
dense-correct case becoming wrong; that conservatively upper-bounds net damage.
It applies only when zero such losses are observed, not whenever net damage is
zero. At n=32, even zero losses cannot rule out 5% harm. At n=128 it can in this
specific zero-loss scenario; ordinary nonzero-loss settings need separate analysis.

Distribution-free Hoeffding widths use R·√(log(2m/alpha)/(2n)), with R=4 for Z
and R=2 for damage. A simultaneous .05 half-width would need 15,320 independent
groups for Z or 3,830 for damage. These deliberately conservative bounds show
the worst case; they are not a recommendation to run that many now.
All case-sampling bounds condition on the selected mask. They do not cover
variation between calibration seeds, models or task families.

## Execution decision

The old exercise's 32-case screen was competence/runtime only. The current controlled
configs select eight reference groups plus their probes (40 role episodes), also
competence/runtime only; primary uncertainty uses only the eight reference cases.
If feasible, evaluate more validation source groups (at least the available 128),
inspect paired discordance, cluster variance, malformed answers and calibration
seed spread. Choose 512–2048 as candidate confirmation sizes only if validation
supports the assumed variability and resource cost. Otherwise revise the design;
never interpret insufficient precision as a negative finding.

Reserve confirmation cases before model-driven choices. The current controlled
builder preserves existing facts and splits when increasing `groups_per_split`.
Audit expansions against previously used discovery/validation groups. The historical
station-count builder's `episodes_per_split` does not preserve splits and must not
be used to expand a used scientific cohort.

Before test: freeze primary conditions, alpha allocation, sample size, grouping,
eligibility, parser/prompts, control contrasts and analysis code. For a claim over
calibration randomness, use more seeds and a pre-specified crossed case/seed
analysis; the current three seeds support a robustness screen, not precise seed
population inference. More bootstrap draws only reduce resampling Monte Carlo
noise; they do not add cases or masks. Later random-mask controls need their own
finite-pool/precision plan (`mask_uncertainty.md`).
