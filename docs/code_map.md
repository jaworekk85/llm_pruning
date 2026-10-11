# Code map

Read the named module for a task; avoid opening datasets/results wholesale.

## Shared core (`src/llm_pruning/`)

| Module | Purpose / entrypoints |
|---|---|
| `models.py` | HF model/tokenizer loading, device/dtype resolution; `load_model` supports revision and local-only loading. |
| `prompts.py` | Legacy domain text-file prompt loading. |
| `prompt_records.py` | QA metadata schema, JSONL loading/validation and domain filtering/grouping. |
| `choice_records.py` | MC metadata schema, JSONL loading/validation, domain/subject filtering, normalized input identity. |
| `targets.py` | `TargetSpec`, generic field grouping and dataset-wide `validate_split_isolation`. |
| `baseline_config.py` | Strict `BaselineConfig` JSON loading; config-relative paths and explicit target/evaluation settings. |
| `run_artifacts.py` | `RunArtifacts` lifecycle and overwrite protection, dataset fingerprints, commit/environment metadata. |
| `mc_prompts.py` | Explicit legacy/plain/native-tokenizer chat MC policies, aligned letter/choice-text instructions and template fingerprints. |
| `mc_evaluation.py` | Shared batched candidate likelihood scoring, `CandidateScore`/`RecordDecision`, decision construction. |
| `hooks.py` | Activation hooks/records for layers, MLP modules/neuron proxies and attention-head proxies. |
| `metrics.py` | Descriptive selectivity, effect size, concentration and decodability summaries. |
| `ablations.py` | Legacy component reference parsing and hook-based `AblationManager`. |
| `components.py` | `ComponentId` and model-bound `ComponentRegistry`: stable query-head/MLP-group IDs, validated Llama/Qwen2 projection/channel mapping, JSON-ready inventory. |
| `component_masks.py` | `ComponentMask`: reversible zeroing before attention output/MLP down projections; GQA-aware, validated selections and exception-safe cleanup. See `docs/component_masks.md`. |
| `component_activations.py` | Target-aware `ComponentActivationCollector` with stable IDs and explicit prompt/answer masks; compact statistics at mask sites, checked tokenization boundaries. |
| `record_selection.py` | Shared MC dataset preflight, split isolation and seeded per-target-value sampling for baseline/activation runners. |
| `token_regions.py` | Shared exact-prefix prompt/answer tokenization, padding masks and position IDs; chat special-token ownership. |
| `discovery_selection.py` | Completed discovery-run provenance/coverage checks and generic per-target activation selectivity rankings. |
| `causal_evaluation.py` | Seeded disjoint random/layer/cost-matched masks, same-example MC evaluation, paired per-target damage and control contrasts. |
| `mask_uncertainty.py` | Paired crossed bootstrap, finite-mask-pool correction, multiplicity and validation-only mask precision planning. See `docs/mask_uncertainty.md`. |
| `wanda.py` | Reversible sequential-block, row-wise Wanda; full-attention Llama/Qwen2 projection adapter, padding-aware norms, actual dense storage/nonzero counts. |
| `compression_frontier.py` | Prompt-only discovery calibration and held-out per-value/non-target/explicit-core quality at fixed independent pruning budgets. |
| `organization_diagnostics.py` | Discovery-only activation/causal vectors, concentration, overlap and repeated-run stability with coverage and signed-score preservation. |

## Experiment entrypoints (`experiments/`)

| Script(s) | Purpose |
|---|---|
| `icml2027/run_target_baseline.py` | Config-driven baseline per target-field value; `--validate-only` performs config/data preflight. |
| `collect_component_activations.py` | MC target-aware head/MLP-group collection; shared baseline config + collection flags; exclusive artifacts and `--validate-only`. See `docs/component_activations.md`. |
| `evaluate_component_masks.py` | Held-out MC mask runner from a discovery run; provenance preflight, controls, paired outputs and exclusive artifacts. See `docs/component_causal_evaluation.md`. |
| `analyze_mask_uncertainty.py` | Analyze saved paired deltas without inference; fresh artifacts, intervals and pilot precision estimates. |
| `run_compression_frontier.py` | Independent target-calibrated Wanda, held-out `frontier.csv`, paired examples and dense-memory/sparsity separation; `--validate-only`. |
| `probe_component_importance.py` | Discovery-only singleton head/MLP-group causal probes; explicit subsets supported; `--validate-only`. |
| `analyze_component_organization.py` | Offline activation/probe concentration and overlap, plus stability across matching discovery runs. See `docs/pruning_and_organization.md`. |
| `generate_text.py` | Generation sanity check; top-level `main.py` is a legacy entrypoint. |
| `collect_activations.py` | QA/prompt activation collection; top-level `activations.py` is a legacy entrypoint. |
| `collect_choice_activations.py`, `collect_choice_answer_activations.py` | MC prompt/teacher-forced answer activation collection. |
| `analyze_activations.py`, `compare_component_rankings.py`, `plot_neuron_heatmap.py` | Activation scores, ranking comparisons and visualization. |
| `ablate_components.py`, `run_ablation_grid.py` | QA loss interventions, controls and budget sweeps. |
| `evaluate_mc_ablation.py` | Legacy domain-based MC ablations; shares scoring with the new baseline runner. |
| `bootstrap_mc_results.py`, `summarize_mc_curve.py`, `summarize_ablation_budgets.py` | Existing uncertainty and curve/budget summaries. |
| `import_mmlu_mc.py`, `import_sciq_mc.py`, `build_qa_v1.py` | Benchmark imports and deterministic QA seed builder. |
| `validate_prompts.py`, `validate_choice_prompts.py` | Existing metadata validation entrypoints. |

`tests/` contains cheap core/offline pipeline checks. `configs/experiments/icml2027/` contains the first runner configuration. `docs/projects/icml2027.md` describes its limits and next steps.
