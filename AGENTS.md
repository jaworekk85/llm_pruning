# AGENTS.md — llm_pruning

Short agent entrypoint; keep under 150 lines. Read `STATUS.md` once per session.
New GPU server: `docs/server_handoff.md`. First task: dense `--baseline-only` on
controlled lookup. Generated `results/` data must be rebuilt after cloning.

## Project and navigation

Study knowledge/computation organization and selective LLM compression.
Primary objective: **ICML 2027**, currently composition under pruning (unconfirmed).
Organization→compressibility is secondary. Planned UCT multilingual and Waikato
streaming projects are independent; partner scope remains to be agreed.

`docs/README.md` routes each question to its owning document and historical versions.
`docs/code_map.md` locates code. For research design only, use the relevant sections
of `LLM_PRUNING_AGENT_RESEARCH_SPEC.md` (§0 map); literature only for research claims.
Routine fixes do not require the spec, strategy or literature audit.

- `src/llm_pruning/`: reusable core; `experiments/` and `configs/`: project entrypoints.
- `data/`: prompt sets/imports; `results/`: ignored local artifacts. Never read wholesale.
- `docs/`: current methodology and short pointers to archived history in Git.

## Rules

1. Framework, not hypothesis: do not hard-code localization or component assumptions.
2. Reusable modules over project hacks; generalize when a second use case needs it.
3. Separate discovery / validation / test; never select and claim on the same split.
4. Activation ≠ causal importance ≠ compressibility. Seed/demo tests are not science.
5. Every run writes `results/<name>/manifest.json` (commit/config/model revision/seed/versions)
   and `summary.md` (≤40 lines). Never overwrite result directories.
6. Add tests for new core functionality; smoke tests use tiny random models, no downloads.
7. No large experimental runs unless explicitly asked.
8. Update `STATUS.md` at milestones; keep ≤40 lines.
9. Keep one owning document per topic; link instead of copying operational instructions.
   Preserve superseded research material in Git and index its immutable version.

## Context and commands

- Find symbols with `rg` and the code map; read targeted ranges, not whole directories.
- Do not dump CSV/JSONL/parquet/logs; use manifests, summaries or short queries.
- Scripts print ≤40 lines. Long runs: background + log; inspect at most 20 recent lines.
- One task per session; do not reread material already in context.
- Tests: `python -m pytest -q -x`. CLI: `python experiments/<script>.py --help`.
