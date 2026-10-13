# Status — 2026-10-13

Primary publication objective: ICML 2027. Shared tools also support independent UCT and Waikato projects.

## Implemented

- Legacy activation collection/rankings, causal ablation with random/layer-matched controls, QA loss and MC evaluation.
- Generic `TargetSpec`: domain, language, capability or task; metadata-preserving QA/MC records.
- Config-driven MC baselines and linked short-answer/Wanda screens; deterministic seeded cohort selection.
- Dataset-wide split checks for duplicate IDs and normalized inputs (MC question + options).
- Fresh run directories with config, manifest, dataset SHA-256, selected IDs, versions, model revision/device/dtype, metrics and ≤40-line summary; failed runs recorded.
- Linked generation/Wanda, staged intermediates, strict integers and test lock; controlled shared-evidence families/probes, intact group caps and reference-only primary readouts; `--baseline-only`.
- Stable head/MLP-group IDs and exportable channel inventory for Llama/Qwen2; explicit contiguous grouping.
- Reversible GQA-aware head and gated-MLP-group zero masks; validated selections and exception-safe cleanup.
- Target-aware activation summaries at the same sites; prompt/answer masks, checked token boundaries and streamed MC run artifacts.
- Explicit legacy/plain/native-tokenizer chat prompt policies shared by baseline, collection and causal scoring; template fingerprints.
- Discovery-only activation selection and held-out mask evaluation, disjoint random/layer/cost-matched controls and paired per-target damage.
- Unique finite-pool controls, shared-mask inference reuse, paired bootstrap intervals and validation-only mask precision planning; offline reanalysis.
- Independent sequential-block Wanda, target-only discovery calibration and held-out quality frontiers with actual dense storage/nonzero counts.
- Discovery singleton probes and offline activation/causal concentration, overlap and repeated-run stability diagnostics.

## Verified

- Offline `venv/Scripts/python.exe -m pytest -q -x`: 282 passed, including controlled gold/probe semantics, split-preserving expansion, reference calibration, baseline-only and weight restoration.
- Controlled lookup→lookup / lookup→add: 128 groups/split each, 7680 records/family; both preflights pass. Start: 8 reference groups + probes (40 episodes); exact 7B tokenizer unavailable locally.
- Native template/boundary checks passed on cached TinyLlama/Qwen2.5 tokenizers; likelihoods checked against manual references.
- Tests and tiny fixtures are engineering checks, not scientific evidence; no pretrained research run.

## Next milestone

Phase 1 engineering complete; benchmark redesigned, scientific acceptance pending. Phase 2: pin/cache 7–8B model/tokenizer, check dense competence/probes with `--baseline-only`, then validation pruning.
New server: `docs/server_handoff.md` (setup, ignored-data rebuild, pin/cache, 8-group dense baseline, review before expansion). Design: `docs/composition_benchmark_design.md`; station-count v1/v2 are exercises.

## Open limits

Legacy collectors/ablations need manifest/overwrite migration and generic target scoring; see `docs/projects/icml2027.md`.
Masks are interventions, not physical compression; pretrained, compiled/distributed and quantized models remain unvalidated.
Native prompt mechanics are checked; pretrained prompt quality/multilingual controls remain unvalidated. Legacy format remains opt-in/default for compatibility.
Declared source/fact/episode groups cannot cross splits; undeclared aliases/paraphrases still need source audits. Context-supplied tasks do not locate stored facts.
New Wanda/frontier/diagnostics usage and scope: `docs/pruning_and_organization.md`.
