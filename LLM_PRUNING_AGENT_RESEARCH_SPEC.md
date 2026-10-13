# LLM Pruning Research Framework — Agent Development Spec

> **Agents:** do not read this whole file at the start of every session.
> Start from `AGENTS.md` (short, always loaded). Read only the section of this spec
> that matches your task (see §0 for the map). This file is the long-form reference.
> Routine code/docs fixes need no spec reading unless they change scientific semantics.

## 0. How to use this document

| If the task is about… | Read |
|---|---|
| scientific design or new research abstractions | `AGENTS.md`, then §1–§2 of this file |
| measurement, metrics, interventions | §3–§5, §14 |
| compression, frontiers | §6–§7, §14 |
| project priorities / collaboration scope | `docs/research_strategy.md`; do not load the whole spec |
| ICML 2027 project | §8 + `docs/projects/icml2027.md`; literature only when evaluating claims |
| UCT multilingual project | §9 |
| Waikato streaming project | §10 |
| configs, repo layout, reproducibility | §11–§13 |
| agent workflow and token budget | §16 |

---

## Purpose

Develop `llm_pruning` as a **general experimental field for studying how knowledge, capabilities, domains, and languages are organized inside pretrained language models, and how that organization relates to selective compression, specialization, and adaptation**.

The repository must **not** be hard-coded around one paper, one domain, or one hypothesis.

It hosts several **independent research projects** that share infrastructure but have their own questions, timelines, and collaborators:

1. **ICML 2027 paper** (own project)
   - primary publication objective,
   - first screen composition loss under pruning, then causal support/mitigation,
   - organization predicting specialist size remains a secondary candidate.

2. **UCT / South Africa collaboration** (own project; overlaps with ICML only through shared tools and possibly language targets)
   - multilingual and low-resource language specialization,
   - shared vs language-specific support,
   - language-specific specialist extraction.

3. **Waikato / New Zealand collaboration** (own project; **not part of the ICML paper**)
   - specialists under changing data distributions,
   - stream-based model-adequacy monitoring,
   - dynamic masks / capacity / precision under drift.

Projects reuse tools, not a compulsory shared paper or hypothesis. Current priority
and collaboration scope are owned by `docs/research_strategy.md`; partner deliverables
are proposals until confirmed. ICML drives near-term development, without encoding
its hypothesis in the shared core.

The common abstraction is:

\[
\text{model}
\rightarrow
\text{internal organization}
\rightarrow
\text{causal validation}
\rightarrow
\text{selective compression}
\rightarrow
\text{adaptation}
\]

Each project uses a different subset of this pipeline.

---

# 1. Scientific design principles

## 1.1. Separate framework from hypothesis

Core code should implement reusable operations:

- dataset loading,
- activation collection,
- component scoring,
- causal intervention,
- compression,
- evaluation,
- shift generation / streaming evaluation,
- experiment logging.

Research hypotheses should live in:
- experiment configs,
- analysis scripts,
- project-specific directories (`experiments/<project>/`).

Do **not** encode assumptions such as:
- "medicine neurons exist",
- "domains are localized",
- "heads are the right granularity",
- "causal modularity predicts compressibility".

These are experimental questions.

## 1.2. Build what the current experiment needs

The framework grows incrementally. Generalize an abstraction when a second use case actually needs it, not in advance. A working pilot beats a complete but unused framework.

---

## 1.3. Treat datasets as first-class research objects

Every example should have structured metadata.

Minimum schema:

```json
{
  "id": "...",
  "domain": "...",
  "capability": "...",
  "language": "...",
  "task": "...",
  "split": "discovery|validation|test",
  "source_name": "...",
  "source_type": "...",
  "prompt": "...",
  "target": "...",
  "difficulty": "...",
  "metadata": {}
}
```

Not every field must be populated, but the schema should allow all projects.

Important dimensions:

- domain: medicine, agriculture, astronomy, law, etc.
- capability: arithmetic, coding, factual recall, instruction following, etc.
- language: English, isiZulu, Afrikaans, etc.
- task: QA, classification, generation, reasoning.
- split: discovery / validation / test.

Useful metadata for confound control (when available): baseline model accuracy, entity/fact popularity (head vs tail knowledge), source benchmark.

---

# 2. Core scientific entities

## 2.1. Model

A wrapper around Hugging Face-style causal LMs.

Responsibilities:
- load tokenizer/model,
- expose architecture metadata,
- normalize access to layers, heads, MLP modules,
- support hooks and interventions,
- expose device / dtype / quantization metadata.

The wrapper must not assume only TinyLlama.

Target families:
- tiny random-init or TinyLlama for smoke tests,
- Qwen,
- Llama-compatible models,
- Gemma.

---

## 2.2. Component

A component is a structural unit that can be measured or intervened upon.

Supported granularities:

```text
layer
attention_head
mlp_module
mlp_group
mlp_neuron
optional_future: subspace/pathway
```

Each component has a stable identifier, e.g.:

```text
layer.7
layer.7.attn.head.3
layer.12.mlp.group.04
```

No experiment should depend on ad-hoc array indices only.

---

## 2.3. Target concept

A target concept can be:

```text
domain
capability
language
task
```

Examples:

```text
domain=medicine
language=isiZulu
capability=arithmetic
task=code_generation
```

All scoring and evaluation APIs accept a generic target specification rather than hard-coded domains.

---

# 3. Measurement layer

## 3.1. Activation collection

For each component \(u_i\), input \(x\), and target \(d\), collect configurable summaries such as \(a_i(x)\).

Possible statistics:
- mean activation,
- absolute activation,
- RMS,
- max,
- sparsity,
- token-position summaries (prompt tokens vs answer tokens — earlier pilots showed this choice matters a lot for MLP neurons).

Store raw large tensors only when explicitly requested.

Default workflow:
- collect compact summaries,
- version metadata,
- write deterministic parquet/CSV/NPZ outputs.

---

## 3.2. Descriptive organization metrics

### Selectivity

\[
S_i(d)=
\frac{\mu_i(d)-\mu_i(\neg d)}
{|\mu_i(d)|+|\mu_i(\neg d)|+\epsilon}.
\]

### Effect size

\[
E_i(d)=
\frac{\mu_i(d)-\mu_i(\neg d)}
{\sigma_{\mathrm{pooled}}}.
\]

### Concentration

Given positive importance/selectivity distribution \(p_i\):

\[
H(d)=-\sum_i p_i\log p_i.
\]

Also:
- normalized entropy,
- effective number of components,
- top-k share,
- Gini coefficient.

### Overlap

For two targets \(d_1,d_2\):

- top-k Jaccard,
- weighted overlap,
- cosine similarity between score vectors.

### Decodability

Linear probe accuracy for target identity from component summaries, at each granularity.

---

# 4. Causal intervention layer

Activation statistics alone are not sufficient.

Support reusable interventions:

```text
zero_ablation
mean_ablation
component_mask
head_mask
mlp_group_mask
layer_drop
optional_future: activation_patch
```

For target \(d\):

\[
I_i(d)
=
Q_d(M)-Q_d(M\setminus u_i).
\]

Experiments must support:
- top-ranked components,
- random controls,
- layer-matched and cost-matched controls,
- repeated random masks,
- held-out evaluation data.

Do not use the same examples for:
- discovering components,
- evaluating causal importance,
- final claims.

---

# 5. Functional modularity

Compute candidate modularity descriptors at different scales.

For target \(d\): \(\mathcal M_d^{(g)}\), where \(g\) is component granularity.

Candidate descriptors:
- concentration of causal importance,
- fraction of target damage explained by top-k components,
- overlap with other targets,
- target-vs-nontarget damage ratio,
- conditional importance,
- redundancy / compensation,
- stability across datasets/seeds.

Do not define one universal "modularity score" too early.

Store primitive measurements first; derive competing metrics later.

---

# 6. Selective compression layer

Compression under target-specific objectives.

For target \(d\), define \(Q_d(M)\) and optionally a minimum general-core quality \(Q_{\mathrm{core}}(M)\).

A specialist is allowed to lose non-target knowledge.

\[
\min_S \operatorname{Cost}(M_S)
\quad\text{s.t.}\quad
Q_d(M_S)\ge Q_d(M)-\epsilon,
\quad
Q_{\mathrm{core}}(M_S)\ge\tau.
\]

Non-target domains do **not** need to be preserved unless an experiment explicitly requires it.

Supported compression policies (add as needed, not all up front):
- **established pruners with target-specific calibration**: Wanda, SparseGPT (primary tools for frontiers),
- structured masks from component rankings (heads, MLP groups, layers),
- optional: low-rank, quantization, recovery fine-tuning.

---

# 7. Specialist compression frontier

For each target \(d\):

\[
F_d(b)
=
\max_{S:\operatorname{Cost}(S)\le b}
Q_d(M_S)
\]

for a sequence of budgets \(b\): `100%, 90%, 75%, 60%, 50%, 40%`.

Any empirical \(F_d\) is a lower bound produced by a specific pruner. Always record which pruner produced it, and prefer conclusions that hold for at least two pruners.

Also record:
- non-target quality,
- core language/reasoning quality,
- parameter count,
- memory,
- theoretical FLOPs,
- real latency/energy when hardware experiments are available.

Each experiment produces a standard `frontier.csv`.

---

# 8. ICML 2027 project

Priorities/decision gates: `docs/research_strategy.md`.
Draft operational protocol: `docs/projects/icml2027.md`.
Literature/history: `ICML2027_RELATED_WORK_AND_NOVELTY_AUDIT.md` (not a frozen protocol).

## Main scientific question

> **Can pruning preserve constituent skills while damaging their composition,
> and can localized support explain and mitigate that damage at matched cost?**

First candidate to screen, not a claimed effect. The previous question — whether
discovery-only organization predicts independent pruning curves beyond baseline
quality/global sensitivity on unseen targets — remains a secondary analysis.
The broad localization→compression framing is occupied; neither candidate has
confirmed novelty. Keep necessary support, sufficient computation and compressibility distinct.
The 13 Oct composition audit also finds close quantization and sparse-circuit
precedents (PRISM, Compressed Lens, Circuit Claims, CircuitLoRA). See
`docs/composition_novelty_review_2026-10-13.md`; compression-sensitive composition
or extra circuit nodes alone are insufficient novelty claims.

## Non-negotiable design requirements

1. **Linked cases.** A, B, A→B and staged/oracle-intermediate controls share case
   identity; source/fact-group splitting keeps all variants in one split.
2. **Baseline feasibility.** Establish pretrained competence, and control constituent
   errors, propagation, prompt format/length and difficulty before claiming extra
   composition loss. Context-supplied facts do not establish stored-knowledge localization.
3. **Discovery/validation/test.** Discover support on discovery, fix choices on
   validation, confirm on untouched test. Resampling respects episode groups.
4. **Independent pruning.** Use established Wanda first. Causal probing and pruning
   outcomes are distinct; targeted methods need held-out, equal-cost comparisons.
5. **Practical controls.** Match calibration token budgets/source and repeat seeds;
   distinguish necessity from sufficiency and nonzero weights from bytes/FLOPs.
6. **Scope of evidence.** A small screen tests feasibility, not universal prediction.
   Larger claims need new task combinations, model/pruner transfer and meaningful
   precision; existing singleton/interaction methods are baselines, not new inventions.

## Granularities

Start with attention heads and grouped MLP channels. Add layers/subspaces/edges
only when a concrete test needs them. Avoid exhaustive individual-neuron experiments.

## Pilot and decision

One pretrained 7–8B model, initially connected lookup→lookup with a separate
lookup→addition family, linked controls and a
small validation-fixed Wanda budget grid. Tiny random-init models only test code.
Precise sources, evaluator, effect thresholds and sample sizes must be fixed before
confirmation. The linked data/evaluator/paired-analysis pipeline and server preflight are
complete locally. Context-supplied cases do not locate factual storage. Do not import
every benchmark from the previous 15-target plan before feasibility.
The controlled benchmark supersedes the station-count exercises; shared evidence,
paired target/distractor/order/name probes and reference-only primary estimates are
implemented. Design: `docs/composition_benchmark_design.md`; sample precision:
`docs/composition_validation_plan.md`. Scientific acceptance requires dense competence
and probe behavior; confirmation analysis and exact token controls precede test.

Internal pilot review target: ~20 Nov 2026, conditional on server availability;
this is not an ICML submission date. Gate: interpretable, precise signal beyond
simpler controls and a concrete contribution beyond closest literature. If not met,
revise the question or test another candidate; no automatic venue change or claim
that a non-significant finding is a publishable negative result.

## Outputs

Linked per-example decisions, A/B/composed/staged/oracle quality changes and intervals,
calibration provenance, pruning cost curves, discovery support measurements and
matched causal comparisons. All runs keep manifests and compact summaries.
The short-answer evaluator, linked data/importer, paired analysis and independent
Wanda screen are implemented and tested offline. Targeted restoration is later work.

---

# 9. UCT / South Africa project

Planned independent collaboration; questions/data/deliverables need partner agreement.
Reuses the same abstractions with `target = language`; it is not required for the
first ICML screen. Language × capability/composition can become a shared use case.

Before choosing a claim, check language-specific neuron, language-targeted pruning
and translation-specialist precedents. Dataset/model access and baseline ability
must be established; do not assume every South African language is already supported.

Candidate questions:

> Is a specialist extracted from a large multilingual LLM better than a small model trained from scratch for a given South African language?

> Do low-resource languages have their own causal support, or do they rely on shared / related-language components?

> Does language-support overlap predict specialist compression robustness?

> Are low-resource languages less functionally modular or less compressible?

For language \(\ell\): \(B_\ell(\epsilon,\tau)\) is the minimum specialist budget preserving that language and shared core abilities.
Observed pruning curves bound achievable cost for specific methods; they do not
identify the ideal minimum. From-scratch comparisons require transparent data,
training/recovery cost, task and tokenizer differences, not just equal parameter counts.

Compare:
- high-resource vs low-resource languages,
- related (e.g. Nguni group) vs unrelated languages,
- shared vs language-specific component support.

Data: Belebele and other multilingual benchmarks first; UCT / Mzansi datasets later, without changing core code.
Agree languages, licenses/source splits, instruction language and downstream
evaluators with partners. Perplexity and downstream ability are separate outcomes.

---

# 10. Waikato / streaming project

**Independent planned collaboration, not part of the ICML paper.** Shares
infrastructure (models, masks, evaluation) and potentially controlled competence
tasks. Partner scope/deadlines are not confirmed here.

Prepare protocol in parallel; implement streaming after static tooling is validated.

A specialist is optimized for a competence envelope \(\mathcal C_t\). The stream may later require \(\mathcal C_{t+1}\neq\mathcal C_t\):
- new topic/domain appears,
- new language appears,
- task mixture changes so that previously removed competence becomes required,
- new subtask is introduced.

The scientific problem is not a change in class frequency. It is:

> **When does the current specialist cease to cover the competence required by the stream, and how should its active capacity change?**

Possible actions:
- switch to generalist,
- restore removed components,
- change mask,
- change precision,
- select another specialist.

```text
stream -> adequacy monitor -> capacity controller -> specialist model
```

Prequential evaluation must score before adaptation, with explicit label delay;
calibration uses only available past data. Compare fixed specialist/generalist
and capacity actions. Count monitoring, recalibration and any stored dense model
needed for fallback/restoration. A removed weight cannot be restored without a
stored source; zero masks do not themselves save physical memory.

Novelty notes (check before committing to a framing):
- "specialist with fallback to a large model" overlaps heavily with **LLM routing / cascades**,
- previously listed DART/context-adaptive pruning is a related-work candidate
  requiring separate verification; context shift within generation is distinct
  from a prequential data stream,
- candidate contribution: competence-drift methodology, controlled shifts and
  cost-aware capacity control; novelty and venue must be agreed/checked separately.

---

# 11. Experiment configuration

All experiments are config-driven.

```text
configs/
  models/
  datasets/
  diagnostics/
  compression/
  experiments/
    icml2027/
    uct_multilingual/
    waikato_streaming/
```

Example:

```yaml
experiment_name: icml_pilot_medicine_qwen7b
model:
  name: ...
  dtype: ...
dataset:
  path: ...
  target_field: domain
  target_value: medicine
granularity:
  - attention_head
  - mlp_group
diagnostics:
  - decodability
  - activation_selectivity
  - causal_ablation
compression:
  pruner: wanda
  calibration: target
  budgets: [1.0, 0.9, 0.75, 0.6, 0.5]
evaluation:
  splits:
    discovery: discovery
    validation: validation
    test: test
seeds: [1, 2, 3]
```

---

# 12. Repository structure target

```text
AGENTS.md                 # short agent entrypoint (<150 lines), always loaded
STATUS.md                 # one screen: current state, next step, open problems
LLM_PRUNING_AGENT_RESEARCH_SPEC.md   # this file, read by section

src/llm_pruning/
  models/
  data/
  components/
  activations/
  diagnostics/
  interventions/
  compression/
  evaluation/
  streaming/              # only when the Waikato project starts
  utils/

experiments/
  icml2027/
  uct_multilingual/
  waikato_streaming/

configs/
  ...

docs/
  code_map.md             # one line per module: what it does, main entry points
  metrics.md
  datasets.md
  interventions.md
  compression.md
  projects/               # per-project notes, decisions, logs
    icml2027.md
    uct.md
    waikato.md

results/                  # large; never read wholesale by agents (see §16)
  <experiment_name>/
    config.yaml
    manifest.json
    summary.md            # <= 40 lines, written by the run script
    metrics.json
    component_scores.parquet
    frontier.csv
    plots/
```

Do not reorganize everything in one destructive refactor. Migrate incrementally while keeping current scripts operational.

---

# 13. Reproducibility requirements

Every experiment saves (in `manifest.json`):
- git commit hash,
- config,
- model identifier/revision,
- dataset version,
- seed,
- device,
- torch/transformers versions,
- intervention method,
- compression policy and pruner,
- evaluation metric definitions.

Do not overwrite existing result directories. Each result folder is self-describing.

---

# 14. Analysis rules

Avoid common methodological errors:

1. **Activation ≠ causal importance.**
2. **Decodability ≠ detachability.**
3. **Detachability ≠ compressibility.**
4. **Pruning mask ≠ real hardware speedup.**
5. **One dataset ≠ capability.**
6. **One random seed ≠ stable finding.**
7. **Top-k overlap alone ≠ functional modularity.**
8. Do not tune a metric on the same test set used to claim predictive validity.
9. **Do not measure modularity and build the frontier from the same component scores** (circularity).
10. **A predictor is only interesting if it beats baseline predictors** (accuracy, knowledge frequency, decodability, global pruning sensitivity).
11. **A frontier is a lower bound from one pruner**; check conclusions with a second pruner.
12. **Few targets ≠ predictive claim**; use held-out targets.
13. Prompt-token and answer-token measurements can select different components; record which one was used.

---

# 15. Development priorities

Dates below are internal planning targets, conditional on hardware availability;
ICML 2027 submission dates have not been verified. Phase 1 is complete locally,
Phase 2 has not started; current evidence/status is in `STATUS.md`.

## Phase 1 — completed locally, 12 Oct 2026

Available offline: target spec, component IDs/masks, MC scoring, Wanda frontiers,
uncertainty, diagnostics and run artifacts. Pretrained validation is outstanding.
Also available: linked-task contract, controlled data/numeric-fact importer, integer
generation evaluator, staged/oracle controls, paired source-group analysis, server
config/preflight and frozen-test lock. Pretrained feasibility/precision are Phase 2.
Do not rebuild completed infrastructure or import all benchmark families up front.

## Phase 2 — ICML pilot (→ ~20 Nov)

See §8 and `docs/projects/icml2027.md`. Validate pretrained feasibility, then the
composition screen and controls. Review candidate strength before causal scale-up.

## Phase 3 — ICML scale-up (Dec–Jan), if GO

Targeted causal retention/restoration, equal-cost controls, second model/pruner and
new task compositions as justified by the pilot. Prior predictive analysis remains
optional and needs its own target-level sample/held-out design.

If NO-GO: document precision/controls and revisit the question. ICML remains the
publication objective until explicitly changed; no automatic NeurIPS fallback.

## Phase 4 — UCT project

Partner preparation may proceed now; implementation follows agreed language/data
scope and reuses validated static tools. Language-specific/shared metrics and
carefully matched low-resource/from-scratch comparisons are candidate deliverables.

## Phase 5 — Waikato project

Partner protocol preparation may proceed now. Later implementation: stream interfaces,
controlled competence shifts, prequential evaluation and cost-aware capacity actions.

---

# 16. Agent behavior and token budget

## 16.1. Behavior

When modifying the repository, the agent should:

1. Use `AGENTS.md` (auto-loaded) and read `STATUS.md` once per session. Read sections of this spec only when scientific semantics need them; skip it for routine fixes.
2. Preserve the framework orientation.
3. Prefer reusable modules over project-specific hacks.
4. Add tests for new core functionality.
5. Keep smoke tests cheap and deterministic (tiny random-init models, no downloads).
6. Avoid large experimental runs unless explicitly requested.
7. Update `STATUS.md` after meaningful milestones (keep it to one screen; move history to `docs/projects/<project>.md`).
8. Record scientific caveats in docs, not only code comments.
9. Do not silently redefine metrics.
10. Do not claim a scientific result from seed/demo datasets.

## 16.2. Context economy

The repository will grow large. Keep the agent's context small:

1. **Do not read `results/`, `data/`, or logs wholesale.** Use `summary.md` files, `head`, `wc -l`, or a short pandas query. Never `cat` a CSV or JSONL.
2. **Every experiment script prints a compact summary** (≤ 40 lines) and writes details to files. Progress bars and per-batch logs go to a log file, not stdout.
3. **Long runs go to the background** with output redirected to a log; check with `tail -n 20`.
4. **Tests run quietly**: `pytest -q -x`, and on failure show only the failing test.
5. **Navigate unfamiliar code via `docs/code_map.md`** and module docstrings. Read targeted symbol ranges; expand only as needed. Do not reread material already in context or load whole code/docs directories.
6. **Keep always-loaded files short**: `AGENTS.md` < 150 lines, `STATUS.md` one screen. Long material lives in docs read on demand.
7. **One task per session.** Start a fresh session for unrelated work instead of continuing a long one.
8. **Specific task prompts** ("add head masking for Qwen2 in `interventions/masks.py`, test with tiny config") instead of open-ended ones ("improve the framework").
9. **Do not spawn subagents by default.** Delegate only when explicitly requested
   or required by applicable instructions; use targeted reads first.

---

# 17. Immediate next milestone

### **Phase 2: pretrained feasibility and validation screen**

Phase 1 deliverables are implemented offline; exact commands/configs live in
`docs/projects/icml2027.md`. A fresh server starts with `docs/server_handoff.md`:
rebuild ignored datasets, pin/cache model and tokenizer, then an eight-group dense
baseline (`--baseline-only`). Next: validate pretrained 7–8B baseline
task/prompt competence, benchmark runtime and run the small validation screen.
Then assess effect/precision and freeze choices before test. Restoration, stream
controllers and functional quantization are later work, not screen prerequisites.

The goal is not to prove the ICML hypothesis. The goal is to find out quickly whether it is worth a paper, while building tools that the UCT and Waikato projects can reuse.

---

# 18. Long-term scientific identity of the repository

\[
\text{Where is knowledge represented?}
\;\downarrow\;
\text{Which parts are causally necessary?}
\;\downarrow\;
\text{Which capabilities can be isolated or compressed?}
\;\downarrow\;
\text{How small can a specialist become?}
\;\downarrow\;
\text{How should specialist capacity change when requirements change?}
\]

This progression should remain visible in the architecture of the codebase.
