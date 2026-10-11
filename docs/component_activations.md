# Component activation collection

The shared collector measures the same query-head and gated-MLP-group channels
as `ComponentMask`. It accepts explicit tokenized batches, `ActivationSample`
metadata and boolean token masks, independently of the dataset or prompt format.
Current architecture support is standard eager-mode HF Llama/Qwen2.

## MC runner

```powershell
venv/Scripts/python.exe experiments/collect_component_activations.py --config configs/experiments/icml2027/component_activations.json --validate-only
venv/Scripts/python.exe experiments/collect_component_activations.py --config configs/experiments/icml2027/component_activations.json --mlp-group-size 128 --token-regions prompt answer
```

The first command checks settings and the entire dataset's split isolation,
without loading a model or creating results. The second loads cached weights;
it has not been run on a pretrained model. The example caps discovery data at
two examples per domain and is an engineering configuration, not scientific evidence.

The input uses the existing `BaselineConfig` JSON schema. Sampling is seeded
and ID-order-independent, and includes all values of the target field on the
configured split. `domain`, `language`, `capability` and `task` are supported
when labels exist. CLI collection settings are recorded in the resolved config.
Use `discovery` for component discovery; validation/test can be measured but
must not be used to choose components for subsequent test claims.

`scoring_mode` determines the teacher-forced **correct** answer continuation:
letter or choice text. No generated answers or incorrect candidates are measured.
Prompt includes the question, options, instruction, wrapper and prefix specials.
Answer positions are consumed input tokens, not preceding next-token prediction
positions. Standalone prompt IDs must be an exact prefix of full-sequence IDs;
cross-boundary merges or trailing prompt specials cause an explicit failure.
Padding is excluded, position IDs handle either padding side, and context
overflow fails without truncation. `prompt_policy` selects legacy, plain base-model
or native tokenizer chat formatting; see `docs/component_causal_evaluation.md`.

## Outputs and interpretation

A fresh `results/<experiment_name>/` contains `config.json`, `manifest.json`,
`summary.md` (at most 40 lines), `component_inventory.json`,
`collection_records.json` (IDs, source/target metadata and answer index), and
`activations.csv`. Failures are recorded; existing directories are never reused.

Each CSV row is one example/component/token region, with split, target field/value,
target membership, token/channel counts and mean, mean absolute value, RMS,
population standard deviation and maximum absolute value. Statistics pool the
selected token x channel values in float32, without pooling different examples.
The last MLP group uses its actual channel count. Raw tensors are not retained;
rows are streamed to disk per batch. Group definitions must accompany IDs.

Tests use random-init models and a local tokenizer with HF offline flags.
Activation magnitude is not causal importance, modularity or compressibility.
QA/generation datasets can use the core with explicit inputs/masks; the current
CLI imports MC records only. Pretrained, compiled/distributed and quantized
execution remain unvalidated. Component semantics: `docs/component_masks.md`.
