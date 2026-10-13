# AGENTS.md — llm_pruning

Short entrypoint for coding agents. Loaded every session, so keep it under 150 lines.
Long-form reference: `LLM_PRUNING_AGENT_RESEARCH_SPEC.md` — read **only the section your task needs** (map in its §0).

## What this repo is

An experimental field for studying how knowledge, capabilities, domains and languages are organized inside pretrained LLMs, and how that relates to selective compression and specialization.

Independent projects share this code; **ICML 2027 is the primary publication objective**:
- **ICML 2027**: first screen composition loss under pruning; organization→compressibility remains a secondary candidate (`experiments/icml2027/`, protocol: `docs/projects/icml2027.md`). Neither hypothesis is confirmed.
- **UCT / South Africa**: planned collaboration on multilingual / low-resource specialists (`experiments/uct_multilingual/`). Partner scope remains to be agreed.
- **Waikato / New Zealand**: planned collaboration on specialists under competence drift; separate from ICML (`experiments/waikato_streaming/`).

Current state and next step: `STATUS.md`. Priorities/decision gates: `docs/research_strategy.md` (read only for research planning). Literature: `ICML2027_RELATED_WORK_AND_NOVELTY_AUDIT.md`.
For a new GPU server: read `docs/server_handoff.md` after STATUS. First task is dense
`--baseline-only` on controlled lookup; generated `results/` data must be rebuilt.

## Layout

- `src/llm_pruning/` — reusable core (models, hooks, metrics, ablations, prompts)
- `experiments/` — scripts and project-specific code
- `configs/` — experiment configs
- `data/` — prompt sets and imported benchmarks (large; do not read wholesale)
- `results/` — experiment outputs (large; do not read wholesale)
- `docs/` — methodology, metrics, per-project notes; `docs/code_map.md` lists modules

## Rules

1. Framework, not hypothesis: never hard-code assumptions like "domains are localized" or "heads are the right unit".
2. Reusable modules over project-specific hacks. Generalize when a second use case needs it.
3. Separate discovery / validation / test data. Never select components and claim results on the same split.
4. Activation ≠ causal importance ≠ compressibility. Do not claim scientific results from seed/demo data.
5. Every run writes `results/<name>/manifest.json` (commit, config, model revision, seed, versions) and `summary.md` (≤ 40 lines). Never overwrite result directories.
6. Add tests for new core functionality. Smoke tests use tiny random-init models, no downloads.
7. No large experimental runs unless explicitly asked.
8. Update `STATUS.md` after meaningful milestones; keep it to one screen.

## Token budget

- Read `STATUS.md` once per session; use `docs/code_map.md` only when the relevant code path is unknown. Do not reread material already in context.
- Routine code/docs fixes do not require the research spec or novelty audit. Read only relevant sections for scientific design, metric/split/protocol changes, or research claims.
- Do not `cat` CSV, JSONL, parquet, or logs. Use `summary.md`, `head`, `wc -l`, or a short pandas query.
- Scripts print ≤ 40 lines to stdout; details and progress bars go to files.
- Long runs: background + log file; check with `tail -n 20`.
- Tests: `pytest -q -x`.
- Find unfamiliar code via `docs/code_map.md` and docstrings; search symbols and read targeted ranges before expanding to whole modules. Never load whole code/docs directories.
- One task per session; prefer specific instructions over open-ended exploration.

## Commands

```bash
pytest -q -x                     # tests
python experiments/<script>.py --help
```
