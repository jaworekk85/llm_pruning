# Held-out component mask evaluation and prompt policies

## Prompt policies

`BaselineConfig.prompt_policy` is explicit and shared by baseline, activation
and causal runners. The compatibility default remains `legacy_mc_v1`;
example configs now request `tokenizer_chat_v1`.

| Policy | Use and behavior |
|---|---|
| `tokenizer_chat_v1` | Instruct/chat model: use its own tokenizer chat template with a user message and generation prompt, then assistant prefill `Answer:`. No extra tokenizer special tokens. Missing templates fail. |
| `plain_mc_v1` | Base model: question/options/instruction followed by `Answer:`, without chat wrappers. Tokenizer special tokens are enabled. |
| `legacy_mc_v1` | Preserve the previous TinyLlama-style format, including its letter-only instruction even with choice-text scoring. For compatibility, not a new scientific protocol. |

New policies align the instruction with letter versus complete-choice-text
continuations. Instructions remain English. Manifests record policy, scoring
mode, assistant prefill and the SHA-256 of the resolved chat template.
Exact-prefix token boundaries, padding masks and explicit position IDs are
shared by collection and scoring. Loss scores use preceding prediction positions;
activation summaries use consumed-token positions. Overflow/unstable boundaries
fail without truncation. MC decisions still use highest **mean** continuation
log-probability, including the leading separator; ties use choice order.

Native template/boundary checks passed locally with cached TinyLlama and
Qwen2.5-3B-Instruct tokenizers. Offline synthetic tests cover special-token
ownership and manual likelihood references. These checks do not establish
prompt quality or pretrained benchmark performance.

## Workflow

First produce a discovery activation run with the desired policy. Then:

```powershell
venv/Scripts/python.exe experiments/evaluate_component_masks.py --config configs/experiments/icml2027/component_causal.json --discovery-run results/icml2027_mmlu_component_activations_v1 --validate-only
venv/Scripts/python.exe experiments/evaluate_component_masks.py --config configs/experiments/icml2027/component_causal.json --discovery-run results/icml2027_mmlu_component_activations_v1 --granularity attention_head --token-region answer --top-k 1 --control-repeats 3
```

Use `validation` while developing the protocol and `test` only after freezing
choices. Evaluation on discovery is rejected. The source run must be complete,
exclusively discovery, with disjoint IDs, complete activation coverage and the
same dataset fingerprint, target, model, revision and prompt/answer policy.
Runtime checks compare the actual inventory, dtype, template and resolved model
revision. If the revision is unavailable, the manifest marks it unverified;
pin immutable revisions for research. `--validate-only` checks data/provenance,
but cannot check loaded templates, layouts or control-pool feasibility.

Selection uses mean-absolute-activation selectivity on the requested region and
granularity: `(target_mean - other_mean) / (target_mean + other_mean + 1e-12)`.
Examples have equal weight; non-target examples are pooled. Stable ID breaks
ties. This is a descriptive ranking, not an assumption of causal localization.

## Conditions and outputs

Baseline and selected top-k mask are evaluated on identical examples/candidates.
Default controls are repeated `random` and `layer_cost_matched` masks.
`--control-strategies` also accepts `layer_matched` and `cost_matched`.
All controls match component kind/count and exclude selected IDs. Matching cost
means channel count times downstream projection output width, a structural proxy.
Impossible matching fails explicitly. Controls are unique within each strategy;
small pools are exhausted rather than repeatedly evaluating the same masks.

Exclusive run artifacts include manifest/config/summary, source file fingerprints,
selection/ranking, inventory, all conditions and budgets, evaluation IDs/metadata,
candidate scores, decisions, per-example paired changes, per-value metrics and
selected-vs-control contrasts. Positive damage means baseline-minus-masked
accuracy/margin, or masked-minus-baseline correct loss. Random-mask mean/std is
descriptive spread, separate from the paired approximate intervals now saved in
`uncertainty.json`. Budget/precision flags and offline analysis: `docs/mask_uncertainty.md`.
No physical compression is performed.
Tiny random-init runs validate implementation only; source-level leakage audits,
prompt-quality checks and statistical uncertainty remain necessary for research.
