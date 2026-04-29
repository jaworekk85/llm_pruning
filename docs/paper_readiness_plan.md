# Paper Readiness Plan

Current pilot claim:

> In TinyLlama on QA v1, domain-selective attention heads show a dose-dependent causal ablation effect on held-out agriculture QA loss, while highly discriminative MLP neuron proxies do not show comparable causal effects.

This is promising, but not yet paper-level. The next work should make the result harder to explain away as a dataset, model, or control artifact.

## 1. Expand QA v1 Evaluation Size

Increase validation and test coverage from 10 examples per domain to at least 100 examples per domain. Keep discovery, validation, and test disjoint.

Purpose: reduce variance and make prompt-level confidence intervals meaningful.

## 2. Freeze Decisions Before Test

Use discovery for component ranking and validation for method selection. Touch the test split only after the control strategy, component counts, domains, and metrics are fixed.

Purpose: preserve a clean final confirmation set.

## 3. Add Matched Controls

Add controls beyond unrestricted random samples:

- layer-matched controls for attention heads and MLP neurons
- activation-strength-matched controls
- possibly rank-band controls, such as components ranked 50-100

Purpose: test whether top-ranked components are domain-specific rather than merely high-impact or layer-position artifacts.

## 4. Replicate Across Domains

Run the same ablation curves for all domains, not just agriculture.

Purpose: distinguish a general domain-localization pattern from an agriculture-specific result.

## 5. Replicate Across Models

Run the protocol on additional small models before scaling:

- a non-chat baseline, such as GPT-2 small or medium
- another small chat model if local or server memory allows
- larger models on the compute server if available

Purpose: test whether the head-vs-neuron contrast is model-family specific.

## 6. Add Behavioral Generation Evaluation

Evaluate generated answers before and after ablation, using exact-match where possible and semantic/manual or model-assisted scoring where needed.

Purpose: show that loss shifts correspond to answer quality changes, not only token-probability artifacts.

## 7. Inspect Mechanisms

For the most robust attention heads, inspect:

- attended token positions
- whether heads attend to domain terms, question words, or answer-relevant context
- how attention patterns differ before and after ablation

Purpose: turn the empirical effect into a mechanistic story.

## Near-Term Priority

Run the agriculture attention-head curve again with layer-matched random controls. If the dose-response survives, the current result becomes much stronger.
