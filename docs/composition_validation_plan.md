# Composition tasks and precision — local review, 13 Oct 2026

This file owns the pre-GPU task review and statistical design. Run commands and
the implemented contract remain in [icml2027.md](projects/icml2027.md).
Novelty decision: [focused review](composition_novelty_review_2026-10-13.md).
No pretrained outputs have been observed; all numbers below are design scenarios.
Current task design and implemented controls:
[composition_benchmark_design.md](composition_benchmark_design.md). The station-count
review below is historical; its v2 exercise is not the current research candidate.

## Task review and changes

Inspected complete linked variants for two ID-sorted validation cases in v1,
then a linked validation case and validation-wide structural counts in v2.
Gold formulas, all source boundaries and staged wiring are checked offline.
This is a logic/prompt review, not human annotation or model feasibility evidence.

The v1 task admits shortcuts: target sentence always first, target entity always
the non-X name, distractor count always target+7. The composed prompt also asks
the retrieval question and arithmetic question in succession, which can invite
two outputs despite the integer-only contract.

`count_affine_v2` removes these shortcuts: target and distractor use the same
name format, target identity and sentence order are shuffled, and distractor
counts are sampled independently with equal counts rejected. The composed
instruction explicitly suppresses the intermediate output and requests the
final result. Difficulty metadata distinguishes lookup, affine and lookup-affine.
Validation target sentence positions: first 62 / second 66, out of 128 cases.
The v1 artifacts are preserved; configs now refer to fresh v2 folders.

Example from v2 validation (`context.0`):

| Variant | Content | Gold |
|---|---|---|
| A | Station-00001 holds 92; Station-00000 holds 486. Query Station-00000. | 486 |
| B | Given count 486, compute 5×count+3. | 2433 |
| Composed | Retrieve Station-00000's count, output only 5×answer+3. | 2433 |
| Oracle | Same composed input, plus the correct intermediate 486. | 2433 |
| Staged | Query A, pass its actual parsed integer to B; malformed A fails the case. | 2433 |

B is a skill bundle (multiplication plus addition), not necessarily a primitive
neural operation. The two stages mean lookup followed by affine computation.
Numbers, templates and English instruction remain narrow; identical arithmetic
can recur across independent station cases. Case IDs prevent exact prompt collisions
but do not establish a held-out arithmetic-template or skill-combination split.

## Controls still requiring pretrained validation

| Risk | Required validation comparison before a neural claim |
|---|---|
| Different input lengths/context | B supplied with the same irrelevant station context; a length-matched lookup-only task; exact token counts under the selected tokenizer. |
| Order or identity shortcut | Evaluate counterbalanced target positions, renamed entities and shuffled context at dense and selected pruned points; retain source grouping. |
| Arithmetic difficulty/ceiling | Inspect multiplier, digit length and carry difficulty; select an adequate family on validation, not test. Dense ≥.70 is only initial feasibility. |
| Answer formatting vs ability | Report invalid separately from valid-but-wrong outputs, retained raw text and truncation/context diagnostics. Do not silently relax the parser after test. |
| Extra computation/information | Staged uses two calls; oracle supplies a gold intermediate. Track inference cost and interpret their advantages diagnostically. |
| Calibration mismatch | Compare composed vs mixed discovery pools at equal tokens, repeat seeds; later reasoning-trace controls if the claim concerns decode-time reasoning. |
| Unsupported factual-storage claim | Context lookup is knowledge use; imported closed-book facts require a separate audited parametric-recall cohort. |

The new controlled benchmark implements shared evidence and target/distractor/
order/name probes in complete source groups. Exact token-length controls, real-fact
audits and pretrained feasibility remain open; see the current design document.
Select no test examples by pruned-model success. Review a larger human sample
when importing real facts, including aliases, arithmetic units and licenses.

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

Reserve confirmation cases before model-driven choices and expand using fresh
source/fact populations or preserving existing split assignments. Rebuilding the
generator with a different `episodes_per_split` changes its shuffle and can move
the same IDs/facts across splits: it is not a safe expansion of a used dataset.
Audit new data against all previously used discovery/validation groups, rather
than relying only on the runner's within-file checks.

Before test: freeze primary conditions, alpha allocation, sample size, grouping,
eligibility, parser/prompts, control contrasts and analysis code. For a claim over
calibration randomness, use more seeds and a pre-specified crossed case/seed
analysis; the current three seeds support a robustness screen, not precise seed
population inference. More bootstrap draws only reduce resampling Monte Carlo
noise; they do not add cases or masks. Later random-mask controls need their own
finite-pool/precision plan (`mask_uncertainty.md`).
