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
| ICML 2027 project | §8 + `ICML2027_RELATED_WORK_AND_NOVELTY_AUDIT.md` §7a, §13 |
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
   - functional / causal modularity at several structural scales,
   - selective specialist compression,
   - does internal organization predict achievable specialist size?

2. **UCT / South Africa collaboration** (own project; overlaps with ICML only through shared tools and possibly language targets)
   - multilingual and low-resource language specialization,
   - shared vs language-specific support,
   - language-specific specialist extraction.

3. **Waikato / New Zealand collaboration** (own project; **not part of the ICML paper**)
   - specialists under changing data distributions,
   - stream-based model-adequacy monitoring,
   - dynamic masks / capacity / precision under drift.

Projects may reuse each other's components, but no project's deadlines or hypotheses should shape another project's code.

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

Full rationale and literature: `ICML2027_RELATED_WORK_AND_NOVELTY_AUDIT.md`.

## Main scientific question

> **Does functional modularity at some structural scale predict how compressible a specialist capability/domain/language is — beyond trivial predictors?**
> (Working framing: *Does Localization Inform Compression?*)

\[
\mathcal M_d^{(g)}
\rightarrow
B_d(\epsilon,\tau),
\qquad
B_d(\epsilon,\tau)
=
\min_S\operatorname{Cost}(S)
\]

subject to target/core quality constraints.

## Non-negotiable design requirements

1. **No circularity.** Modularity is measured on the discovery split from cheap local probes (single-component ablations, small subsets, activations). Frontiers are computed with an **independent** pruner (Wanda/SparseGPT with target calibration) and evaluated on the test split.
2. **Baseline predictors.** Report whether modularity predicts \(B_d\) beyond: baseline accuracy on \(d\), knowledge frequency (head/tail), decodability, global pruning sensitivity, evaluation-set size/difficulty.
3. **Many targets.** Aim for 25–40 targets (MMLU/MMLU-Pro subject clusters, Belebele languages, math, code), not 4.
4. **Targets span the modularity axis.** Include targets expected to be modular at neuron level (languages), head level (math/code), and not at neuron level (academic subjects).
5. **Held-out evaluation.** Fit on some targets, predict held-out targets; if ≥2 model families, fit on one, predict the other.
6. **Robustness.** Bootstrap over examples and random controls; several calibration seeds; at least two pruners.

## Granularities

- attention heads,
- grouped MLP channels,
- layers.

Avoid exhaustive individual-neuron experiments.

## Pilot (decision by ~20 Nov 2026)

```text
model:        one 7–8B model (e.g. Qwen2.5-7B or Llama-3.1-8B); TinyLlama only for smoke tests
targets:      ~15 (6–8 MMLU subject clusters, 2–3 Belebele languages, math, code)
granularity:  heads + MLP groups
diagnostics:  decodability, selectivity, single-component ablation, top-k vs layer-matched random
frontier:     Wanda with target calibration; budgets 90, 75, 60, 50%
key plot:     modularity (per granularity) vs minimal specialist budget, next to baseline predictors
```

Decision rules: see audit §13.

## Outputs per `(model, target, granularity)`

1. activation organization metrics,
2. causal importance metrics,
3. target/non-target damage matrix,
4. compression frontier (per pruner),
5. modularity-vs-compressibility analysis with baseline predictors.

---

# 9. UCT / South Africa project

Reuses the same abstractions with `target = language`.

Context:
- UCT NLP group (MzansiText/MzansiLM, 2026): a 125M decoder trained from scratch on 11 South African languages; small decoders stay near chance on reasoning/reading comprehension; corpus ~84% Afrikaans+English.
- Prior work to position against: language-specific neurons (ACL 2024), CRANE (2026), language-targeted pruning limitations (TACL 2026: target-language calibration keeps perplexity but does not reliably help downstream tasks), translation specialists from MoE (2026).

Candidate questions:

> Is a specialist extracted from a large multilingual LLM better than a small model trained from scratch for a given South African language?

> Do low-resource languages have their own causal support, or do they rely on shared / related-language components?

> Does language-support overlap predict specialist compression robustness?

> Are low-resource languages less functionally modular or less compressible?

For language \(\ell\): \(B_\ell(\epsilon,\tau)\) is the minimum specialist budget preserving that language and shared core abilities.

Compare:
- high-resource vs low-resource languages,
- related (e.g. Nguni group) vs unrelated languages,
- shared vs language-specific component support.

Data: Belebele and other multilingual benchmarks first; UCT / Mzansi datasets later, without changing core code.

---

# 10. Waikato / streaming project

**Independent of the ICML paper.** Shares only infrastructure (models, masks, evaluation).

Start after static specialist tooling is stable.

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

Novelty notes (check before committing to a framing):
- "specialist with fallback to a large model" overlaps heavily with **LLM routing / cascades**,
- "drift-adaptive inference-time pruning" is partly taken by **DART (2026)**, though there drift means context shift within a single generation,
- the likely distinctive contribution is **data-stream methodology**: prequential evaluation, label-free competence-drift detection, controlled competence-shift benchmarks, cost-aware capacity control. Natural venues: ECML-PKDD, KDD, Machine Learning journal.

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

## Phase 1 — minimal pipeline for the ICML pilot (now → ~25 Oct)

Only what the pilot needs, built generically:
1. generic target spec (`domain`, `language`, `capability`, `task`),
2. stable component IDs for heads and MLP groups,
3. head and MLP-group masking for Llama/Qwen 7–8B,
4. Wanda (target-calibrated) integration and per-target evaluation,
5. result manifest + `summary.md` per run,
6. data import: MMLU subject clusters, Belebele subset, GSM8K, HumanEval/MBPP.

## Phase 2 — ICML pilot (→ ~20 Nov)

See §8. Decision go/no-go.

## Phase 3 — ICML scale-up (Dec–Jan), if GO

Second model family, 25–40 targets, second pruner, baseline predictors, held-out analysis, layer granularity.

If NO-GO: document the result, test another scale (subspace/pathway), retarget to NeurIPS 2027.

## Phase 4 — UCT project

Multilingual target datasets, language-specific/shared metrics, low-resource comparisons, comparison with small from-scratch models.

## Phase 5 — Waikato project

Stream interfaces, controlled competence shifts, adequacy monitoring, adaptive capacity actions.

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
9. **Delegate bulky exploration** (searching many files, reading long logs) to a cheaper subagent that returns a short summary.

---

# 17. Immediate next milestone

### **Minimal generic pipeline for the ICML pilot**

Deliverables:
- target abstraction supporting `domain`, `language`, `capability`, `task`,
- stable component IDs for heads and MLP groups,
- head / MLP-group masking for a 7–8B Llama/Qwen model,
- target-calibrated Wanda + per-target evaluation,
- discovery/validation/test-aware evaluation,
- experiment config format,
- result manifest + `summary.md`.

Then run the pilot in §8.

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
