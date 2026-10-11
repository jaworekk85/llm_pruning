# Component IDs and reversible masks

These shared primitives support studies of localization, causal importance,
redundancy and later compression. The ICML modularity/compressibility hypothesis
is a starting research question; it does not determine component selection or
assume that knowledge is localized. Targets and scientific claims belong in
experiment configurations and project documentation.

## Usage

```python
from llm_pruning.components import ComponentRegistry
from llm_pruning.component_masks import ComponentMask

# model is an already-loaded HF LlamaForCausalLM or Qwen2ForCausalLM.
registry = ComponentRegistry(model, mlp_group_size=128)
selected = ["layer.0.attn.head.1", "layer.0.mlp.group.0"]
with ComponentMask(registry, selected):
    masked_output = model(**inputs)
# Hooks have been removed; weights and normal forward behavior are restored.
```

The registry enumerates every query head and every MLP intermediate-channel
group. IDs are canonical strings such as `layer.7.attn.head.3` and
`layer.12.mlp.group.4`. `registry.resolve(id)` returns the projection path and
half-open channel range. Unknown IDs, unsupported model families and unexpected
projection layouts fail explicitly before masking. Duplicate selections are
deduplicated; an empty selection is an identity intervention.

## Intervention semantics

- Attention heads: zero each selected head's output channels at the input to
  `self_attn.o_proj`, before heads are mixed into the residual stream. IDs refer
  to query heads, including when GQA shares key/value heads.
- MLP groups: zero channels at the input to `mlp.down_proj`, after the nonlinear
  gate and up-projection product. Groups partition intermediate channels into
  contiguous ranges of the requested size; the last group may be smaller.

Projection biases, residual connections and other channels are preserved.
Inputs are cloned before modification, and autograd remains supported. Hooks
are removed when exiting the context, including when an exception occurs.
An instance can be reused after close, but cannot be entered twice concurrently.
Separate instances can compose masks. Use one instance per active intervention
and do not concurrently evaluate masked and unmasked inputs on the same model.

Contiguous groups are an explicit structural partition, not discovered
functional modules. Changing group size changes the meaning of group IDs.
Head IDs and group IDs are architecture-relative, not cross-model alignments.

## Reproducibility and scope

Persist `registry.to_dict()` and the selected IDs together with the model
revision and the experiment's `manifest.json`. The JSON-ready inventory records
its schema version, grouping size, projection names and channel membership.
The core utility performs no model downloads or experiment artifact writes;
the calling runner must use the repository's run lifecycle and split policy.

Current support is eager PyTorch execution of standard, unquantized HF
Llama/Qwen2 causal LM layouts. Eager and SDPA attention paths are tested on tiny
random-init models, with GQA and a partial final MLP group. Pretrained 7–8B,
compiled/distributed execution, wrappers and quantized layouts are unvalidated.
Other architectures need an explicit adapter and offline tests.

Masks are reversible causal interventions. They do not shrink parameter tensors,
provide a hardware speedup, select useful components, or demonstrate scientific
modularity/compressibility. Future discovery/evaluation runners should consume
the same IDs and inventory; independent pruners can then measure compression.
Legacy module-name ablations remain a separate API with different MLP semantics.
