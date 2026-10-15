# Documentation index

Read `AGENTS.md` and `STATUS.md` first. Open only the owner of the current question.
A routine fix does not need the research strategy, long spec or literature audit.

## Current owners

| Question | Owning file |
|---|---|
| Current state and next action | [STATUS](../STATUS.md) |
| New GPU server: setup and ordered commands | [Server handoff](server_handoff.md) |
| Research priorities, phases and partner scope | [Research strategy](research_strategy.md) |
| ICML runner contract, artifacts and lock | [ICML project](projects/icml2027.md) |
| Task semantics, probes and scientific acceptance | [Benchmark design](composition_benchmark_design.md) |
| Uncertainty, size and confirmation planning | [Validation plan](composition_validation_plan.md) |
| Current candidate's novelty and source log | [Composition review](composition_novelty_review_2026-10-13.md) |
| Earlier prediction hypothesis and literature | [Main audit](../ICML2027_RELATED_WORK_AND_NOVELTY_AUDIT.md), §18–21 |
| Find relevant code | [Code map](code_map.md) |
| General methodology beyond this pilot | [Research spec](../LLM_PRUNING_AGENT_RESEARCH_SPEC.md), selected sections |

Keep instructions in their owner; elsewhere link to them. STATUS reports milestones,
not the complete feature inventory. CLI implementations remain authoritative for
accepted flags/config fields. Preserve code/API paths and reproducibility artifacts.

## Historical versions

Superseded text is preserved **in full, byte for byte, in commit**
`58e27a0c126cd31458c6ef73e6dc369e67535a4a`. No history rewrite or result deletion was performed.
The following immutable links show the pre-cleanup versions:

| Material | Preserved version |
|---|---|
| Earlier README / legacy commands | [README](https://github.com/jaworekk85/llm_pruning/blob/58e27a0c126cd31458c6ef73e6dc369e67535a4a/README.md) |
| Full previous status / feature inventory | [STATUS](https://github.com/jaworekk85/llm_pruning/blob/58e27a0c126cd31458c6ef73e6dc369e67535a4a/STATUS.md) |
| Mixed composition/MC protocol and exercise details | [ICML project](https://github.com/jaworekk85/llm_pruning/blob/58e27a0c126cd31458c6ef73e6dc369e67535a4a/docs/projects/icml2027.md) |
| Station-count task review and earlier precision document | [Validation plan](https://github.com/jaworekk85/llm_pruning/blob/58e27a0c126cd31458c6ef73e6dc369e67535a4a/docs/composition_validation_plan.md) |
| Seed/TinyLlama roadmap | [Experiment plan](https://github.com/jaworekk85/llm_pruning/blob/58e27a0c126cd31458c6ef73e6dc369e67535a4a/docs/experiment_plan.md) |
| Earlier paper assessment | [Readiness plan](https://github.com/jaworekk85/llm_pruning/blob/58e27a0c126cd31458c6ef73e6dc369e67535a4a/docs/paper_readiness_plan.md) |
| Earlier result labels and thesis | [Result index](https://github.com/jaworekk85/llm_pruning/blob/58e27a0c126cd31458c6ef73e6dc369e67535a4a/docs/paper_result_index.md) |
| Earlier runtime notes and measurements | [Experiment log](https://github.com/jaworekk85/llm_pruning/blob/58e27a0c126cd31458c6ef73e6dc369e67535a4a/docs/experiment_log.md) |

Offline retrieval: `git show <commit>:<path>`; inspect only the relevant lines.
A full clone contains this history; a shallow clone may need to fetch the commit.
Historical assessments are not current scientific evidence. Raw ignored `results/`
remain local and require separate transfer; Git history does not back them up.

## Local size and housekeeping

Measured 15 Oct 2026 before cleanup: tracked files 6.92 MiB; local `venv/` 4.74 GiB,
`results/` 599.51 MiB. Environment/results dominate disk, not code or Markdown.
They are ignored by Git and are not sent when cloning/pushing. Keep them until
their reproducibility/backup requirements have been handled; this cleanup retains
both. Do not remove packages, rewrite history or introduce a framework refactor
merely to reduce document-reading cost.
