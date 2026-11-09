# LLM Pruning Research Framework

Study how knowledge and computation are organized inside pretrained LLMs and how
that relates to selective compression. **Primary publication objective: ICML 2027.**
UCT multilingual and Waikato streaming are independent planned collaborations.

Start with [STATUS.md](STATUS.md). For a new GPU server, follow
[server handoff](docs/server_handoff.md): rebuild ignored data, pin/cache the model
and tokenizer, then evaluate the dense model before pruning.

The current candidate benchmark uses connected lookup→lookup and lookup→addition
with shared evidence and paired behavioral probes. Engineering checks passed;
pretrained competence, the scientific hypothesis and novelty remain unconfirmed.
Context tasks measure use of supplied facts, not factual storage in model weights.

## Find the relevant document

[Co można zbadać z tym repo?](docs/research_capabilities.md) — Polish guide to
available experiments, GPU-dependent work and planned extensions.

[Documentation index](docs/README.md) lists the owner of each topic and historical
material. [Code map](docs/code_map.md) locates modules without reading the source tree.
Agent rules: [AGENTS.md](AGENTS.md).

| Path | Contains |
|---|---|
| `src/llm_pruning/` | Reusable model, evaluation, intervention and pruning tools |
| `experiments/`, `configs/` | Project entrypoints and configuration |
| `tests/` | Offline checks with tiny random models |
| `data/` | Tracked seed/imported prompt sets; inspect selectively |
| `results/` | Ignored local datasets, manifests and experiment outputs |

## Offline verification

In an environment with the project's dependencies installed:

```bash
python -m pytest -q -x
```

Environment/model setup and the ordered server commands have one owner:
[server_handoff.md](docs/server_handoff.md). Earlier setup/legacy examples remain
[available in Git](https://github.com/jaworekk85/llm_pruning/blob/58e27a0c126cd31458c6ef73e6dc369e67535a4a/README.md).
