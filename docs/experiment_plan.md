# Experiment Plan

The first research question is:

> Are domain-relevant computations in a small language model more concentrated in some components than expected by chance?

This repo now supports three initial measurement levels.

## 1. Layer-wise

Layer-wise collection uses `output_hidden_states=True` and summarizes the hidden state after each transformer block.

This answers coarse questions:

- Do domains separate more strongly in early, middle, or late layers?
- Are there layers where agriculture prompts differ from unrelated domains more than expected?

## 2. MLP neuron-wise

Neuron-wise collection currently measures units from each MLP `gate_proj` module.

This is a practical first proxy for MLP neurons in TinyLlama/Llama-like models. It is not yet the final post-activation intermediate value inside the MLP. If the signal looks promising, the next step is to patch or wrap the model MLP forward pass so we can collect the exact post-activation intermediate.

Useful first command:

```powershell
python activations.py --granularity mlp_neuron --top-units-per-module 32 --output-csv results/mlp_neuron_top32.csv
```

Without `--top-units-per-module`, this can produce large CSV files because TinyLlama has thousands of MLP units per layer.

## 3. Attention-head-wise

Attention-head collection hooks the input to `self_attn.o_proj`, reshapes it into attention heads, and summarizes each head vector.

This gives a first head-level signal:

- Which heads activate more strongly for one domain than others?
- Are domain-selective heads concentrated in particular layers?

This is not yet a causal attention test. Later experiments should ablate heads and measure domain-specific performance drops.

## Recommended Order

1. Run layer-wise collection on all seed domains.
2. Run attention-head collection on all seed domains.
3. Run MLP-neuron collection with a top-k limit.
4. Expand the prompt sets before interpreting results.
5. Add causal ablation: mask top-ranked components and compare against random baselines.
6. Only then attempt pruning and agriculture-focused fine-tuning.
