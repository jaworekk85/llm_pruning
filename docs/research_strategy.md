# Research strategy — 2026-10-13

## Aim and priority

Map knowledge and computation inside pretrained LLMs and use causal evidence to
understand and improve selective compression. **Primary publication objective:
ICML 2027.** UCT (South Africa) and Waikato (New Zealand) are independent planned
collaboration tracks. Their questions below are proposals, not partner commitments.

This file owns project priorities and decisions. Technical reference: research
spec §8–§10/§15. ICML pilot protocol: [projects/icml2027.md](projects/icml2027.md).
Read these only when the task needs scientific design; ordinary fixes need STATUS.

## Three projects, shared measurements

| Project | Candidate question | Current priority |
|---|---|---|
| ICML 2027 | Does pruning preserve constituent skills while impairing their composition, and can localized support explain and mitigate that loss? | First candidate to screen; not a confirmed phenomenon or novelty claim. |
| UCT / South Africa | Which shared and language-specific components support low-resource languages, and what must a compact specialist preserve? | Prepare language/data/evaluation choices with partners; reuse the static pipeline. |
| Waikato / New Zealand | When do changing competence requirements invalidate a specialist, and what capacity action is worth its cost? | Prepare a stream/evaluation protocol with partners; implement a controller after static validation. |

Common tools: target metadata, split isolation, evaluators, component IDs/masks,
activation/causal measurements, independent pruning, uncertainty and manifests.
Share tools rather than claim one paper answers all three projects. Neither a
language benchmark nor drift engineering is required to finish the ICML screen.

## ICML alternatives and decision gate

1. **First screen: composition under pruning.** Evaluate linked A, B and A→B tasks
   on the same underlying cases. Establish unpruned competence, then distinguish
   constituent failure, error propagation and extra composition loss. A convincing
   paper needs causal localization and a useful held-out consequence at matched cost.
   The current [benchmark design](composition_benchmark_design.md) starts with
   connected lookup→lookup and a separate lookup→addition family, shared evidence
   and behavioral probes. Earlier station-count/affine data are engineering exercises.
2. **Secondary analysis: organization → independent pruning frontier.** Keep the
   previous hypothesis; reuse the same runs where suitable. It is not an automatic
   fallback paper if the composition screen fails, and a few task bundles cannot
   support a population-level prediction claim.
3. **Later alternatives:** conditional/backup support during compression, or
   functional grouping for physical quantization. Do not build both before a pilot.

Relevant precedents to read before freezing a claim:
- [Circuit Compositions, ACL 2025](https://aclanthology.org/2025.acl-long.727/):
  compositional circuits already exist as a research question.
- [CoAx, v3](https://arxiv.org/html/2607.01940v3): importance conditional on primary
  removal and backup completion; discovering backup components alone is not new.
- [ProxySPEX](https://arxiv.org/html/2505.17495): head interactions already guide pruning.
- [TAGQuant](https://aclanthology.org/2026.eacl-industry.18/): channel shuffling already
  improves group quantization; rearrangement alone is not a novelty claim.

The [13 October focused audit](composition_novelty_review_2026-10-13.md) adds close
quantization and sparse-circuit precedents: PRISM, Through a Compressed Lens,
Circuit Claims and CircuitLoRA. Compression-sensitive composition, extra support
beyond atomic circuits and circuit-guided adaptation are already occupied. Keep
the pilot; require distinct causal protection under matched compression cost and
held-out transfer. The audit documents reading depth and remaining citation gaps.
Prior predictive audit: [literature_novelty_review_2026-10-11.md](literature_novelty_review_2026-10-11.md).

**Scale only after:** usable pretrained baseline; effect precise enough to assess
against a validation-fixed meaningful threshold; controls rule out simpler failure
explanations; concrete distinction from closest work. Causal support and transfer
must then substantiate a stronger contribution. A non-significant result or a
different protocol alone is not an ICML contribution. Revise the question if needed;
do not automatically move the project to another venue or declare a negative paper.

## Next five tasks

Phase 1 is complete locally: linked-data contract/builder/importer, deterministic
short-answer evaluator, staged/oracle controls, paired analysis and server config/preflight.
Pretrained feasibility and scientific confirmation remain open.
New-server execution order and data/model setup: [server handoff](server_handoff.md).
The controlled benchmark was redesigned after the initial shortcut review; its
engineering checks are complete, but scientific acceptance requires dense model behavior.

1. Pin/prepare a pretrained 7–8B model and validate A/B/A→B competence on the server.
2. Measure runtime/calibration feasibility and run the small validation screen.
3. Apply the [task/precision review](composition_validation_plan.md) to pretrained
   validation; finish necessary prompt controls and confirmation analysis, then
   freeze before test. Local task review, v2 generator fixes and size scenarios are done.
4. If warranted, localize support and implement matched-cost causal retention/restoration.
5. Replicate on another model/pruner and new combinations, and audit the exact ICML claim.

Tasks 1–3 need explicit run scope and hardware; no large run has started. Context-supplied
cases do not locate stored factual knowledge. Numeric-fact imports support a separate
parametric track. Targeted restoration remains unimplemented.

## Collaboration preparation

UCT: agree a small language set with adequate baseline ability, data access/licenses,
instruction language, comparable task formats and source-grouped splits. Test
perplexity and downstream quality separately. Comparisons with small from-scratch
models require matched scope and transparent data/training/recovery costs.

Waikato: specify domain/language/composition shifts, label delay, prequential order
and cost axes. Learn only from data/labels available at time t; keep future data out
of mask calibration. Compare fixed specialist, fixed generalist and capacity actions.
Count detector/recalibration overhead and any stored generalist needed for fallback
or restoration. Zeroing dense weights does not free VRAM or guarantee speedup.

Partner deadlines and deliverables remain unset until confirmed. Collaboration
planning can proceed alongside ICML; implementation expands when a concrete shared
use case needs it. The ICML submission dates are unverified; milestones in the spec
are internal targets and must be revised after hardware availability is known.
