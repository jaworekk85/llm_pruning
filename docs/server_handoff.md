# Server handoff — ICML 2027, 13 October 2026

Start here after `AGENTS.md` and `STATUS.md`. This file is sufficient to identify
the next task without the chat history. Read detailed design only when interpreting
data/results: [benchmark](composition_benchmark_design.md),
[precision](composition_validation_plan.md), [runner](projects/icml2027.md).
Implementation checkpoint: `0b52d96`; use latest `main` containing this handoff too.

## Objective and present state

Primary goal: ICML 2027. Study knowledge/computation organization and its relation
to selective compression. First candidate: whether constituent skills survive
pruning while their composition suffers, and whether localized support can explain
and mitigate that harm at matched cost. This is an unconfirmed hypothesis and
novelty claim. Organization→compressibility prediction is secondary; planned UCT
and Waikato collaborations are independent and not prerequisites for this pilot.

Phase 1 engineering is complete. Last local offline suite: **282 passed**. No
pretrained research run, GPU feasibility result, test-set performance, confirmed
compression effect or targeted restoration exists. The next task is **dense
pretrained competence and behavioral-probe validation**, before a pruning grid.
Do not restart framework development or a full literature audit as server setup.

Current candidates: `lookup_lookup` first, `lookup_add` after primary feasibility.
Earlier station-count/affine datasets and `composition_screen.json` are engineering
exercises; use `composition_lookup_screen.json` for the first server pilot.
Context tasks measure use of supplied facts, not storage of factual knowledge.

## 1. Fresh checkout and environment

Work from repository root. Do not copy the local Windows virtual environment.
The repo declares Python >=3.10 and dependencies in `pyproject.toml`.
On a Linux server, create/activate an environment, install a CUDA-compatible
PyTorch build for that server, then install the project and test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -x
```

Local reference versions: Python 3.10.11, torch 2.6.0+cu124, transformers 5.5.0,
tokenizers 0.22.2, numpy 2.2.6. These are verified local versions, not proof of a
server/CUDA environment. `pyproject.toml` does not pin them. Record the actual
server versions and resolve environment/test failures before research inference.
Inspect GPU model, free VRAM, CUDA visibility and BF16 support. No specific server,
GPU capacity, credentials or runtime estimate has been agreed in the repository.

## 2. Regenerate data omitted from Git

**`results/` is ignored. A clone has no generated datasets or result manifests.**
Build the primary dataset, from repo root:

```bash
python experiments/icml2027/build_controlled_composition.py --output-dir results/composition_lookup_v1 --family lookup_lookup --groups-per-split 128 --seed 2027 --namespace composition_v1 --items 4
```

Expected: 128 independent source groups per split; each has five role episodes and
four static variants, totaling 7680 records. The builder writes `manifest.json`,
`summary.md`, `audit.json` and `records.jsonl`. All gold and probe contracts are
checked without models. When needed, repeat with `--family lookup_add` and fresh
`--output-dir results/composition_add_v1`. Never overwrite an existing directory;
reuse verified data or build into a fresh folder and adjust the config path.
Dataset hashes are recorded by builds/runs; verify them if transferring data.
Fresh builders write UTF-8/LF on every OS. With the exact defaults above, expected
`records.jsonl` SHA-256 values are:

| Family | SHA-256 |
|---|---|
| lookup_lookup | `581e54ee6c319975d41c097e2d5b6ecb3e2b0f530be0608ac5001f58796d1f01` |
| lookup_add | `f352a14670a261247fb98f35c53cca7e5951c3443e39c06c976709f3440c2660` |

Earlier local Windows artifacts use CRLF and have different byte hashes despite
identical records; they were preserved. Transfer them unchanged if bound to a
frozen protocol. There is currently no frozen scientific confirmation protocol.

## 3. Pin and cache the model on the server

Candidate: `Qwen/Qwen2.5-7B-Instruct`. Draft config has `revision=null`, which is
deliberate: research execution rejects an unpinned revision. Resolve an immutable
40-hex model commit, record it and cache **both tokenizer and safetensors weights
at that revision** on the server. Downloading weights can be large; none should
be downloaded onto the local phone-tethered machine as part of this handoff.

Use the installed Hugging Face Hub client/model repository to resolve and cache
the revision; do not put tokens in configs or tracked files. Keep inference config
`local_files_only=true`. Verify tokenizer chat-template identity and prompt-token
lengths at the pinned revision: the exact 7B tokenizer was unavailable locally.
Model code and native chat prompting are engineering-tested, not pretrained-validated.

Create a fresh JSON copy next to the template config, e.g.
`configs/experiments/icml2027/composition_lookup_server_8.json`, preserving relative
dataset paths. Set `revision` and a unique `experiment_name` such as
`lookup_server_validation_8_001`. Keep `split=validation`, `fixture_run=false`,
`max_eval_source_groups=8`, and no episode cap. Check device/dtype/batch size against
the real GPU; record any changes. This copy is an experiment setting, not a new
research hypothesis. Baseline run names get an additional `_dense` suffix.

## 4. First inference: dense model only, eight complete groups

```bash
python experiments/icml2027/run_composition_screen.py --config configs/experiments/icml2027/composition_lookup_server_8.json --baseline-only --validate-only
python experiments/icml2027/run_composition_screen.py --config configs/experiments/icml2027/composition_lookup_server_8.json --baseline-only
```

Preflight must report **40 validation episodes, 8 reference cases / source groups**
and 640 discovery episodes. `--baseline-only` evaluates one dense condition and
never constructs a pruner. It still checks the discovery token budget after model
loading. For longer runs launch in background with a separate log file; inspect
the compact summary and at most 20 recent log lines. Each retry needs a fresh name.

Review `results/<experiment_name>_dense/summary.md`, then concise `metrics.json`
queries and selected raw answers in `decisions.csv`. Primary metrics use reference
cases only; `behavioral_controls.roles` contains the four paired probe readouts.
Check A/B/composed accuracy, malformed outputs, both-correct probe rates, valid
pair counts, input changes, timing and peak/storage limits. Eight reference cases
are a competence/runtime check, not evidence of the hypothesis or adequate power.

## 5. Decision after the first baseline

- If loading/tokenization/runtime fails, fix the concrete issue and retry in a fresh
  result directory. A failed run must retain its manifest; do not replace artifacts.
- If dense answers/probes are poor, inspect validation examples and revise prompts,
  task difficulty or model before pruning. Do not cherry-pick cases after pruning.
- If usable, prepare a fresh validation config/name with 128 source groups and run
  `--baseline-only` to assess competence across layouts/positions and probe roles.
  Candidate A/B/composed threshold is .90, alongside low malformed rates and correct
  target-change/invariance behavior. Exact acceptance thresholds need validation
  review; the existing flag does not certify the benchmark or statistical significance.

Then scope a small validation Wanda screen using the reviewed cohort/config and a
fresh experiment name, omitting `--baseline-only`. The template grid is retained
fractions [1,.9,.75,.5], composed/mixed calibration, seeds [11,23,37], 4096 discovery
tokens per condition. Estimate its cost from baseline/runtime first. Read the
benchmark and precision plans before interpreting it. This handoff is a planned
workflow, not blanket authorization for large runs; follow `AGENTS.md` rule 7.

## Boundaries and report back

Keep discovery/validation/test separate. Do not open test performance or freeze a
scientific confirmation protocol before choosing size, multiplicity and analysis
on validation. Existing bootstrap intervals/flags are exploratory; confirmation
analysis, neural localization, matched-cost retention/restoration and transfer are
later work. Masks do not demonstrate physical VRAM/storage/speed reductions.

After each meaningful server milestone update `STATUS.md` (<=40 lines) with config,
revision, result path, checks and next decision. Report actual versions/hardware,
dense reference/probe performance, malformed rates, time/memory, failure types and
whether the task is usable for a scoped pruning screen. Do not label it an ICML
result, novel mechanism or successful hypothesis from engineering tests alone.
