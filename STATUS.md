# Status — 2026-10-15

Primary goal: ICML 2027. UCT/Waikato are independent planned collaborations.

## Current state

Phase 1 engineering complete; benchmark scientific acceptance pending.
No pretrained research run, confirmed compression effect or test-set result.
Available: linked integer generation, staged/oracle controls, reference-only
analysis, complete behavioral-probe groups, Wanda and frozen-protocol checks.
Other framework tools: `docs/code_map.md`; task design: `docs/composition_benchmark_design.md`.

## Verified

- Last code verification: 282 tests passed offline; tiny models only.
- Lookup→lookup / lookup→add: 128 source groups/split each, 7680 records/family.
- Both preflights passed; first screen uses 8 reference groups plus probes (40 episodes).
- Exact 7B tokenizer/model competence remain unvalidated.
- Documentation cleanup preserves previous versions in Git; code, data and results unchanged.

## Next task

On the GPU server follow `docs/server_handoff.md`: environment, ignored-data rebuild,
pin/cache model+tokenizer, then `--baseline-only`. Review dense answers/probes before
expanding validation or running the pruning grid. No large run without explicit scope.

Limits: exploratory uncertainty/point flags; context use ≠ factual storage;
masks ≠ physical compression; targeted restoration and confirmation analysis pending.
