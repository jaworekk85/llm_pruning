# Composition under compression — novelty review, 13 Oct 2026

## Decision

Subsequent dataset redesign: [controlled benchmark](composition_benchmark_design.md).
The instrument uses shared evidence and input probes; its design itself is not
claimed as the ICML contribution. The literature verdict below remains conditional.

**Keep the small pilot, narrow the claim, do not scale yet.** Neither constituent
success with composition failure nor compression-sensitive multi-hop reasoning is
new. A possible ICML contribution is discovery-selected causal support that protects
composition in an independently pruned pretrained model, transfers to new task
combinations and improves quality at equal retained capacity. This is a candidate,
not established novelty or an acceptance prediction.

The 12 October check missed close compression precedents. In particular, PRISM
already includes context composition and component-wise interventions; Circuit
Claims already studies atomic/compositional circuits in dense and weight-sparse
models. Moving from quantization to Wanda, increasing model size or renaming extra
components as “bridges” would not establish a substantial contribution.

## Scope and provenance

18 keyword/title queries, targeted backward reference checks and 16 primary-source
records. Relevant methods/results or limitations were inspected for 12 papers,
one additional paper received a method-only check and three an abstract-only check.
This is a focused claim
audit, not exhaustive citation coverage or reproduction. Exact access levels,
URLs, queries and remaining gaps: [source log](composition_audit_20261013_sources.json).
No model weights or benchmark downloads were needed. Search snippets with outdated
titles were checked against current primary pages; absence of a keyword is not
evidence that a paper lacks an experiment.

## Closest precedents

| Work and actual reading | What is already occupied | Consequence for our experiment |
|---|---|---|
| [Press et al., Compositionality Gap](https://arxiv.org/html/2210.03350), introduction and §2 | Correct constituent answers can coexist with a wrong composed answer; explicit decomposition helps. | Our atomic-correct/composed-wrong rate is a related descriptive readout, not a new concept. A staged benefit also changes inference computation. |
| [Through a Compressed Lens, v3](https://arxiv.org/html/2505.13963v3), §3–5 and limitations | Qwen/Llama weight-only quantization, one-/two-hop factual queries, bridge-entity recall and neuron/layer attribution. Authors associate two-hop degradation strongly with first-hop failure. | Distinguish that explanation from additional composition loss. This is the April 2026 camera-ready revision, not the older explainability title returned by some indexes. |
| [PRISM](https://link.springer.com/article/10.1007/s10791-026-10580-5), §4 comprehension, §5 component/depth controls and long-context probes | Separates stored recall, context copying and context composition under PTQ; interventions include MLP/attention precision and parameter-mass/calibration controls. Early-MLP equal-mass recall effects are RTN-specific and vanish under GPTQ. | Closest context-compression diagnostic precedent. “MLP matters more” alone is insufficient. Its mixed-precision deployment analysis also occupies coarse component allocation. |
| [Circuit Claims, v1](https://arxiv.org/html/2607.18921), §3, conclusion, supplementary comparisons and limitations | Synthetic atomic/compositional Lean tasks in dense and sparsely trained checkpoints; circuit conclusions depend on extraction object, Q/K representation and threshold. Extra support beyond atomic circuits accompanies compositional recovery. | “Composition needs extra circuit nodes” is not new. Sparse training is distinct from post-training Wanda, but that distinction alone is too small. Require intervention-tested, stable support. |
| [Circuit Compositions, ACL 2025](https://aclanthology.org/2025.acl-long.727.pdf), §3–4 and §7 | Activation masks in a trained 58M seq2seq model; primitive circuits overlap and unions gain other string-edit behaviors while sometimes losing constituent performance. No completeness guarantee; zero/mean ablation changes masks. | Include the union of constituent supports as a meaningful control. Activation sparsity is not physical weight compression. Merely redoing set unions in a larger model is weak. |
| [CircuitLoRA, ICML 2025](https://arxiv.org/html/2502.11812v2), §5–6 | Directly compares arithmetic composition circuits with primitive-edge unions and uses either to allocate LoRA ranks, with random-layer controls. | Circuit-informed improvement of arithmetic composition already exists. Our distinction must involve compression, a stronger causal test and held-out consequences; avoid claiming the first circuit-guided composition repair. |
| [CoAx, v3](https://arxiv.org/html/2607.01940v3), §2–6 | Conditional importance after primary removal, causal backup tests and held-out circuit completion across model families. | Include a conditional-importance comparator at matched discovery budget. Rediscovering importance changes or adding dormant backup heads is not enough. Circuit completion is not the same cost metric as our retained weights. |
| [ProxySPEX](https://arxiv.org/html/2505.17495), §3 and §5.2/Fig. 10 | Interaction-aware head selection improves held-out MMLU pruning over marginal LASSO and random-mask search. | Interaction-aware pruning is occupied. Match candidate sites, calibration data and search budget before attributing gains to composition-specific support. |
| [Attribution-guided Pruning](https://arxiv.org/html/2506.13727), §3.1–3.3 | LRP ranks parameters/components and extracts sparse behavior-preserving circuits. | Parameter-level circuit extraction already exists. Attribution/activation/singleton controls remain needed alongside interaction-aware controls. Later correction experiments were not fully reviewed here. |
| [RESP / Think Before You Prune](https://arxiv.org/html/2512.02185), §3 and §4.1–4.2 | Pruning reasoning models, task/self-generated calibration and decode-only importance; generic calibration can destroy reasoning performance. | Calibration gains are a confound, not evidence of a unique composition mechanism. Our prompt-only Wanda screen is narrower than its reasoning-trace protocol. |
| [Quantization Trap](https://arxiv.org/html/2602.13595), §4.1–4.3 | Studies multi-step math accuracy together with backend/hardware energy and latency. | Accuracy and physical cost are separate. Its chained conditional-probability expression does not justify multiplying separately measured A/B accuracies. Backend-specific results are not universal quantization laws. |
| [Circuit explained](https://pmc.ncbi.nlm.nih.gov/articles/PMC12871980/), setup, circuit discovery and discussion | Small symbolic transformer: causal circuit extraction and activation edits explain function/argument routing. | A mechanistic composition account itself is occupied; natural pretrained tasks and intervention stability still require evidence. |
| [Grokked Transformers](https://arxiv.org/html/2405.15071), §3–4 | Controlled factual composition, depth-specific retrieval and cross-layer memory-sharing explanation/mitigation. | Missing downstream access can explain failure without a separately stored “composition skill”. Do not assume a dedicated component class in advance. |

Additional abstract-only neighbors are
[Localize-and-Stitch](https://arxiv.org/abs/2408.13656) (sparse task merging),
[RL compositional reasoning](https://arxiv.org/abs/2609.19465) (decomposed/composed
training asymmetry) and [CoT composition theory](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9990a18179e862042c18dfc856b0e53f-Abstract-Conference.html).
Merging task performance is not automatically sequential A→B reasoning.

## Candidate contribution and falsifiers

Operational question: after matching constituent damage and pruning cost, can
discovery-only measurements identify a small support whose retention improves
composed performance on held-out cases and new combinations, beyond constituent
unions, singleton/activation and conditional/interaction-aware selectors?

This requires behavioral and neural evidence separately. The screen's
`damage(composed) - damage(staged)` is a change in the direct-versus-staged gap,
not an estimator that uniquely identifies a neural composition mechanism. Also
inspect oracle and B controls, malformed outputs and dense baselines. A finding
must survive prompt/context/difficulty controls before being called extra
composition loss. The lookup→affine exercise reviewed here has since been superseded
by a controlled lookup→lookup / lookup→addition benchmark, still requiring pretrained acceptance.

Three concrete ways the candidate fails:

1. The apparent loss is constituent failure, answer formatting, distractor position,
   longer input or unequal inference computation; revised controls remove it.
2. Primitive-union or existing conditional/interaction selection provides the same
   held-out improvement at equal retained weight count and selection cost.
3. Selected support fails transfer/stability or removing unrelated matched support
   has the same effect. A map or correlation alone does not rescue the claim.

Targeted restoration must add and remove capacity to preserve the budget; count
the retained nonzeros/structure and any stored dense source checkpoint. Evaluate
atomic and non-target harm too. Physical storage/latency improvements need a
physical compression implementation, beyond the current reversible masks.

## Gate before a larger research run

First run the small pretrained validation screen. If promising, expand validation
for the [precision plan](composition_validation_plan.md), add matched-context/
format controls, and compare the discovery selectors above. Freeze the final
cohort size, primary conditions, thresholds and analysis before test.

The stronger contribution needs transfer to another model/pruner and genuinely
new combinations, plus a practical or mechanistic consequence. Those are research
requirements proposed here, not formal ICML 2027 acceptance criteria. Publication
dates/decisions for unverified preprints and the 2027 deadline remain unverified.

Remaining audit work before claiming priority: full forward-citation search for
the closest quantization/circuit papers; the original TwoHop-Fact methodology;
method/results checks of the three abstract-only neighbors; full supplementary
pruning/correction protocols; source-code reproduction where the exact claim
depends on an implementation detail. OpenReview blocked CircuitLoRA's PDF, but
its v2 arXiv methods and composition results were accessible and read.
