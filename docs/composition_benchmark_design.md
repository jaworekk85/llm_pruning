# Composition benchmark design — 13 Oct 2026

This is the current dataset design. The earlier station-count/affine sets are
engineering exercises; fixing their obvious shortcuts did not establish a sound
research benchmark. Implemented code: `experiments/icml2027/controlled_composition.py`.
Statistics: [validation plan](composition_validation_plan.md). Novelty:
[focused review](composition_novelty_review_2026-10-13.md).

## Question and admissible conclusions

Does pruning damage connected use of two available skills beyond constituent
damage, and can discovery-selected support protect that use at matched cost?
The initial dataset must distinguish this from hard arithmetic, context removal,
formatting failures, position/name shortcuts and extra staged computation.

Controlled context tasks let us perturb facts with known consequences. They
measure information use and task execution, not the storage of world knowledge
inside weights. Behavioral dependence on two facts does not prove that the model
uses two distinct neural modules or a particular sequential internal algorithm.
The initial benchmark is a measurement instrument, not the proposed paper's novelty.

## Two task families, evaluated separately

| Family | A | B with supplied correct intermediate | Composition | Purpose |
|---|---|---|---|---|
| `lookup_lookup`, primary | Entity → numeric code | Code → numeric value | Entity → code → value | Connected retrieval without arithmetic difficulty. |
| `lookup_add`, secondary | Entity → numeric count | Add one supplied constant to count | Retrieve count, then add | Checks whether the observation extends beyond two retrievals. |

Each reference context contains four plausible entities; the lookup family also
contains four code/value mappings. Keys and values are distinct within their
respective tables, mapping order is shuffled independently, and a mapped value
does not equal its own key. Names are fictional and have equal character length.
Intermediate and final answers are two-digit integers; addition has one operation,
not the earlier multiply-and-add bundle. Carry cases are recorded explicitly.

All A/B/composed/oracle variants within an episode receive the **same full evidence**.
B does not silently lose distractors or the retrieval context. The questions
differ, so exact tokenizer length is not equal by construction. Check query length
and tokenization under the pinned model; do not claim this controls every difficulty
or positional effect. Staged B uses its actual A output with that episode's evidence.
Oracle supplies the correct intermediate; both controls remain diagnostic and
have extra information/computation compared with direct composition.

Example from lookup validation: an entity has code 41, and code 41 has value 78.
A should return 41; B with code 41 and composition should return 78. Swapping
that entity's code with another entity's code 32 changes composition's answer to
75, while preserving the second-stage table. Swapping two other entities' codes
must leave 78 unchanged. In the addition family, count 35 plus 3 gives 38;
changing the relevant count to 51 gives 54.

## Probe block: observable predictions before any neural interpretation

Every independent source group contains five linked episodes:

| Role | Intervention on the input | Required outcome |
|---|---|---|
| Reference | Original facts and query | Correct A, B and composition. |
| Changed target | Swap queried entity's intermediate with another entity's | A and final answer change to their new gold values. |
| Changed distractor | Swap values of two unqueried entities | Correct outputs remain unchanged. |
| Reordered | Rotate evidence rows without changing facts | Correct outputs remain unchanged. |
| Renamed | Bijectively rename every entity, including the query | Correct outputs remain unchanged. |

The target intervention preserves the multiset of facts' numeric values and the
second-stage rule, reducing crude digit/count confounds. Correctness of both
reference and probe is the principal paired diagnostic: a constant wrong answer
can be invariant but does not pass. Also report the valid-pair count and prediction
change rate; exclude no cases from accuracy because the model failed a probe.
These are input counterfactuals, not causal evidence about a neural component.

Reference target positions are counterbalanced across four rows. Both sentence
and table layouts include every position; layout is not tied to position parity.
The two layouts are a robustness comparison, not a held-out-template experiment:
both appear on discovery/validation/test. New templates and composition families
must be reserved separately for a transfer claim.

## Truth, splits and primary estimates

An independent symbolic executor produces and rechecks every gold answer.
Preflight verifies canonical questions, exact shared evidence, distinct functional
keys/values, all probe roles and their intended transformations. Imported real
facts need their own evidence/source/license audit; they are not silently converted
into this controlled benchmark.

Each split and source-group index has a separate deterministic RNG. Increasing
`groups_per_split` preserves existing IDs, facts, prompts and split assignments.
Namespace, seed, family and context size form the population identity. A new
namespace identifies a fresh generated population; it does not certify independence
of externally imported aliases. Retain/audit all previous discovery/validation use
before constructing the eventual confirmation dataset.

All role episodes and all variants stay in one source group. A group cap retains
complete blocks; an episode cap that cuts a block is rejected. **Primary estimates
use reference episodes only.** Probe roles receive separate paired readouts and
source-group bootstrap intervals. Five correlated copies are not five independent
examples. Families are not pooled, and calibration uses discovery reference
A/B/composed prompts only, with no oracle/probe/output-target calibration.

## What was checked locally

Built each family with 128 independent groups per split: 640 role episodes per
split, 7680 static records per family. The current screen selects eight whole groups
(40 role episodes), but its primary estimate contains only eight reference cases.
That is a runtime/competence check, with less precision than the older 32-reference
screen; expand validation before interpreting a pruning effect.

Reviewed 16 reference cases across both families, both layouts and all four target
positions. All gold/probe contracts are checked across the complete generated
datasets; tests verify changing relevant facts, invariant irrelevant edits, intact
blocks, no probe calibration, safe expansion and runner/weight restoration.
Lookup-add validation has 56 no-carry and 72 carry reference cases. This is a
documented distribution, not an assertion that both strata are equally difficult.

The exact Qwen2.5-7B tokenizer was unavailable in the local cache; no download was
attempted. Its token-length audit and all pretrained competence remain pending.
Tests use tiny random models and verify implementation only. Neither automatic
contracts nor this review can certify that the tasks elicit the intended computation
in a pretrained model.

## Acceptance gate and later factual track

On discovery/validation, first run the dense model: start with eight groups, then
all available 128 reference groups. Require adequate A/B/composed competence
(candidate .90 on each), low malformed rates and correct target-change/invariance
behavior; inspect each layout, target position and carry stratum. A pooled average
can hide an unusable subgroup. Choose exact gate thresholds before examining test.
The .90 config flag is only a point-estimate screening aid, not certification.

If dense behavior fails the probes, revise elicitation/data or change model before
running the pruning grid. If a pruned effect appears, check its dependence on layout,
length, context size and constituents, then test neural interventions and matched
selectors. Absence of an effect on this easy family is not evidence that all skill
composition is robust. More difficult tasks must preserve the same identification
controls rather than merely add longer calculations.

For stored factual knowledge, the next separate track should use source-audited
closed-book atomic/composed queries with aliases and bridge types, plus grounded
versions of the same cases. A candidate source is SOCRATES; MuSiQue is a candidate
for natural grounded transfer. They were read for methodology, not downloaded or
approved as ready imports. Neither may be labeled contamination-free for our model
without an audit; output evaluation there will require more than an integer parser.

Methodological basis: [MuSiQue](https://aclanthology.org/2022.tacl-1.31/), §3–4,
requires dependence between connected steps and tests shortcut risks;
[SOCRATES](https://aclanthology.org/2025.findings-acl.205/), introduction/shortcut
discussion, motivates separate factual/composition evaluation and co-occurrence
controls. [Min et al.](https://aclanthology.org/P19-1416/), abstract-only check,
warns that compositional questions can admit single-hop solutions. We adopt these
design lessons; our small generated task is not a new version or reproduction of
those benchmarks. Full corpus/empirical checks of the proposed imports remain open.
