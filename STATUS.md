# Status — 2026-10-11

This is a shared research framework; ICML, UCT and Waikato remain independent projects.

## Implemented

- Legacy activation collection/rankings, causal ablation with random/layer-matched controls, QA loss and MC evaluation.
- Generic `TargetSpec`: domain, language, capability or task; metadata-preserving QA/MC records.
- Config-driven ICML target baseline runner reusing shared MC scoring; seeded per-value sampling.
- Dataset-wide split checks for duplicate IDs and normalized inputs (MC question + options).
- Fresh run directories with config, manifest, dataset SHA-256, selected IDs, versions, model revision/device/dtype, metrics and ≤40-line summary; failed runs recorded.
- Offline tests include a tiny random-init Llama and local tokenizer; no model downloads.
- Stable head/MLP-group IDs and exportable channel inventory for Llama/Qwen2; explicit contiguous grouping.
- Reversible GQA-aware head and gated-MLP-group zero masks; validated selections and exception-safe cleanup.
- Target-aware activation summaries at the same sites; prompt/answer masks, checked token boundaries and streamed MC run artifacts.
- Explicit legacy/plain/native-tokenizer chat prompt policies shared by baseline, collection and causal scoring; template fingerprints.
- Discovery-only activation selection and held-out mask evaluation, disjoint random/layer/cost-matched controls and paired per-target damage.
- Unique finite-pool controls, shared-mask inference reuse, paired bootstrap intervals and validation-only mask precision planning; offline reanalysis.
- Independent sequential-block Wanda, target-only discovery calibration and held-out quality frontiers with actual dense storage/nonzero counts.
- Discovery singleton probes and offline activation/causal concentration, overlap and repeated-run stability diagnostics.

## Verified

- `venv/Scripts/python.exe -m pytest -q -x` with HF offline flags: 216 passed; Llama/Qwen2 mask, Wanda frontier and diagnostics pipelines tested.
- MMLU preflight: baseline 40 validation records; collection 8 discovery records across 4 domains. No pretrained run.
- Native template/boundary checks passed on cached TinyLlama/Qwen2.5 tokenizers; likelihoods checked against manual references.
- Tests and tiny fixtures are engineering checks, not scientific evidence; no pretrained research run.

## Next milestone

Freeze a concrete pilot protocol: targets/data, meaningful effect, calibration seeds, budget grid and power/clustered-data design.
Then validate pretrained prompt quality and run the pilot on the GPU server; add a second pruner for robustness.

## Open limits

Legacy collectors/ablations need manifest/overwrite migration and generic target scoring; see `docs/projects/icml2027.md`.
Masks are interventions, not physical compression; pretrained, compiled/distributed and quantized models remain unvalidated.
Native prompt mechanics are checked; pretrained prompt quality/multilingual controls remain unvalidated. Legacy format remains opt-in/default for compatibility.
Text checks cannot detect paraphrases/semantic leakage; imported datasets need source-level audits.
New Wanda/frontier/diagnostics usage and scope: `docs/pruning_and_organization.md`.
