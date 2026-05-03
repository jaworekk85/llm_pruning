# Experiment Log

This file records the main experimental results in a compact, paper-oriented form.
Raw CSV outputs live under `results/`, which is git-ignored, so important metrics should be copied here after each run.

## Runtime Notes

Local machine:

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM
- PyTorch CUDA: `2.6.0+cu124`
- Usual inference flags: `--device cuda --dtype float16`

Observed wall-clock examples:

| Run | Approx Time | Notes |
| --- | ---: | --- |
| QA v1 agriculture head ablation, CPU, r3 | 616 s | `batch-size=1`, CPU |
| Same QA v1 run, GPU, r3 | 15.5 s | `batch-size=4`, ~40x faster |
| QA v1 large head discovery activations | 243 s | 750 discovery prompts |
| MMLU capped head discovery activations | 117 s | 399 discovery records |
| MMLU astronomy choice-text ablation, one k, r30 | ~25 min | 334 validation records across 4 domains, 31 conditions |

Runtime rough estimate for ablation:

```text
time ~= model_load_time + conditions * candidates * average_choice_length / GPU_throughput
conditions = 1 baseline + 1 top_target + random_control_repeats
candidates = records * choices
```

For the current MMLU astronomy curve, each `k` with `r30` evaluates:

```text
31 conditions * 334 records * 4 choices = 41,416 scored continuations
```

That has been taking about 25 minutes per `k` locally. A five-point curve is therefore about 2 hours.

## Synthetic QA v1 Findings

Dataset:

- `data/prompt_sets/qa_v1.jsonl`: 250 records
- `data/prompt_sets/qa_v1_large.jsonl`: 1,750 records
- Domains: agriculture, astronomy, math, medicine, politics
- Evaluation metric: target-answer cross-entropy loss

### Agriculture Attention Heads, QA v1 Pilot

Path:

- `results/qa_v1_ablation_curve_agriculture_heads_layer_matched_r30`

Layer-matched controls, 30 repeats:

| k heads | top-minus-layer-matched loss |
| ---: | ---: |
| 1 | +0.045 |
| 3 | +0.103 |
| 5 | +0.117 |
| 10 | +0.151 |
| 20 | +0.285 |

Interpretation: strong synthetic dose-response signal.

### Agriculture Attention Heads, QA v1 Large

Path:

- `results/qa_v1_large_ablation_curve_agriculture_heads_large_scores_layer_matched_r30`

Large discovery rankings, large validation set, layer-matched controls, 30 repeats:

| k heads | top-minus-layer-matched loss |
| ---: | ---: |
| 1 | +0.040 |
| 3 | +0.089 |
| 5 | +0.082 |
| 10 | +0.194 |
| 20 | +0.257 |

Interpretation: synthetic signal survives larger validation. Not perfectly monotonic at k=3 to k=5, but clearly positive.

### Agriculture MLP Neurons, QA v1 Pilot

Path:

- `results/qa_v1_ablation_curve_agriculture_neurons_r30`

| k neurons | top-minus-random loss |
| ---: | ---: |
| 5 | -0.002 |
| 20 | -0.001 |
| 50 | -0.008 |
| 100 | -0.244 |

Interpretation: MLP neuron proxies are domain-discriminative but do not show causal damage under this ablation setup.

### Agriculture MLP Neurons, QA v1 Large-k Ablation

Paths:

- `results/qa_v1_ablation_curve_agriculture_neurons_large_random_r30`
- `results/qa_v1_ablation_curve_agriculture_neurons_layer_matched_r30`

Same QA v1 validation split and same MLP neuron rankings as above. These runs test whether the earlier null/negative neuron result was caused by ablating too few neurons.

Random controls, 30 repeats:

| k neurons | top delta loss | random mean delta loss | top-minus-random loss |
| ---: | ---: | ---: | ---: |
| 200 | -0.014 | +0.473 | -0.487 |
| 400 | +0.013 | +1.834 | -1.821 |
| 500 | +0.533 | +3.441 | -2.909 |

Layer-matched controls, 30 repeats:

| k neurons | top delta loss | layer-matched mean delta loss | top-minus-layer-matched loss |
| ---: | ---: | ---: | ---: |
| 50 | +0.020 | +0.043 | -0.023 |
| 100 | +0.043 | +0.059 | -0.016 |
| 200 | -0.014 | +0.243 | -0.257 |
| 300 | -0.001 | +0.707 | -0.709 |

Notes:

- k=600 could not be run with random controls from the current top-32-per-prompt component score file: after selecting 600 agriculture neurons, fewer than 600 agriculture-scored components remain for each random control set.
- Layer-matched controls are limited to about k=300 because some layers do not have enough unselected same-layer alternatives after larger selections.

Interpretation: increasing k does not rescue the MLP neuron causal story. Even at hundreds of selected neurons, top agriculture-selective neurons are less damaging than random or layer-matched neuron controls. This strongly supports the decodability-versus-causality framing: MLP neuron activations contain domain-discriminative information, but the selected raw neurons are not compact causal levers for agriculture answer loss in this setup.

### Politics MLP Neurons, QA v1 Large-k Ablation

Paths:

- `results/qa_v1_ablation_curve_politics_neurons_large_random_r30`
- `results/qa_v1_ablation_curve_politics_neurons_layer_matched_r30`

This repeats the large-k neuron test for a second domain.

Random controls, 30 repeats:

| k neurons | top delta loss | random mean delta loss | top-minus-random loss |
| ---: | ---: | ---: | ---: |
| 200 | -0.054 | +0.470 | -0.524 |
| 400 | +0.926 | +0.336 | +0.590 |
| 500 | +0.696 | +1.004 | -0.307 |

Layer-matched controls, 30 repeats:

| k neurons | top delta loss | layer-matched mean delta loss | top-minus-layer-matched loss |
| ---: | ---: | ---: | ---: |
| 50 | -0.025 | +0.057 | -0.082 |
| 100 | -0.033 | +0.109 | -0.141 |
| 200 | -0.054 | +0.832 | -0.886 |

Interpretation: unrestricted random controls are noisy for politics, including one positive top-minus-random point at k=400. The cleaner layer-matched controls agree with agriculture: selected politics neurons are less damaging than same-layer neuron controls. This strengthens the claim that raw MLP neuron selectivity does not straightforwardly identify compact causal domain components.

### Agriculture Head-vs-Neuron Budget Comparison

Path:

- `results/qa_v1_agriculture_head_neuron_budget_comparison.md`
- `results/qa_v1_agriculture_head_neuron_budget_comparison_top128.md`

This table expresses k as a fraction of total model components: 704 attention heads and 123,904 MLP gate neurons.

| curve | granularity | k | model fraction | top-minus-control |
| --- | --- | ---: | ---: | ---: |
| agriculture heads, layer-matched | attention head | 1 | 0.142% | +0.045 |
| agriculture heads, layer-matched | attention head | 3 | 0.426% | +0.103 |
| agriculture heads, layer-matched | attention head | 5 | 0.710% | +0.117 |
| agriculture heads, layer-matched | attention head | 10 | 1.420% | +0.151 |
| agriculture heads, layer-matched | attention head | 20 | 2.841% | +0.285 |
| agriculture neurons, layer-matched | MLP neuron | 50 | 0.040% | -0.023 |
| agriculture neurons, layer-matched | MLP neuron | 100 | 0.081% | -0.016 |
| agriculture neurons, layer-matched | MLP neuron | 200 | 0.161% | -0.257 |
| agriculture neurons, layer-matched | MLP neuron | 300 | 0.242% | -0.709 |

The original top-32-per-prompt neuron score file could not support the 0.710% matched budget. A broader top-128-per-layer-per-prompt activation collection was therefore added:

- `results/qa_v1_discovery_mlp_neurons_top128.csv`
- `results/qa_v1_analysis_mlp_neurons_top128`

Top-128 neuron analysis:

| metric | value |
| --- | ---: |
| activation rows | 422,400 |
| unique scored components | 7,100 |
| agriculture positive-selectivity neurons | 2,480 |
| leave-one-out domain decodability | 0.920 |

Matching the 5-head budget of 0.710% requires about 880 MLP neurons. The top-128 score file supports this k with layer-matched controls.

Matched-budget agriculture neuron ablation, k=880, 30 repeats:

| control | top delta loss | control mean delta loss | top-minus-control loss |
| --- | ---: | ---: | ---: |
| random | +0.033 | +0.387 | -0.354 |
| layer-matched | +0.033 | +0.666 | -0.632 |

Interpretation: even at the same global component fraction as 5 attention heads, selected agriculture MLP neurons are less damaging than controls. This substantially strengthens the head-vs-neuron contrast: attention heads show positive causal loss effects at small budgets, while raw MLP neurons remain highly decodable but not compact causal levers under zero-ablation.

### QA v1 Large Matched-Budget Neuron Check

Paths:

- `results/qa_v1_large_discovery_mlp_neurons_top128.csv`
- `results/qa_v1_large_analysis_mlp_neurons_top128`
- `results/qa_v1_large_ablation_agriculture_neurons_top128_k880_layer_matched_r30`

Large top-128 neuron analysis:

| metric | value |
| --- | ---: |
| activation rows | 2,112,000 |
| unique scored components | 8,874 |
| agriculture positive-selectivity neurons | 2,971 |
| leave-one-out domain decodability | 0.968 |

Matched-budget agriculture neuron ablation on QA v1 large, k=880, layer-matched controls, 30 repeats:

| comparison | top delta loss | control mean delta loss | top-minus-control loss |
| --- | ---: | ---: | ---: |
| MLP neurons, k=880 | +0.079 | +0.181 | -0.101 |

For comparison, the QA v1 large agriculture head curve has:

| comparison | top-minus-control loss |
| --- | ---: |
| attention heads, k=5 | +0.082 |

Interpretation: the larger synthetic dataset preserves the same qualitative contrast. Broad MLP neuron activations are highly domain-decodable, but matched-budget selected neurons still do not beat layer-matched controls; selected attention heads do.

### Math Head-vs-Neuron Matched-Budget Check

Paths:

- `results/qa_v1_ablation_math_heads_k5_layer_matched_r30`
- `results/qa_v1_ablation_math_neurons_top128_k880_layer_matched_r30`

QA v1 validation, math target domain, layer-matched controls, 30 repeats:

| component | k | model fraction | top-minus-control loss |
| --- | ---: | ---: | ---: |
| attention heads | 5 | 0.710% | +0.012 |
| MLP neurons | 880 | 0.710% | -0.435 |

Interpretation: math does not show a strong positive attention-head causal effect at k=5 in this run, but it does replicate the negative neuron result. This weakens any broad claim that attention heads always show compact causal domain leverage, while strengthening the claim that raw MLP neuron selectivity is not sufficient for causal importance.

### QA v1 MLP Neuron Layer Heatmaps

Paths:

- `results/qa_v1_analysis_mlp_neurons_top32/mlp_neuron_layer_heatmap_top100.svg`
- `results/qa_v1_analysis_mlp_neurons_top32/mlp_neuron_layer_heatmap_top500.svg`

These heatmaps aggregate discriminative MLP neuron scores by layer for each domain. They are not ablation results; they show where the domain-selective neuron signal is located.

Top layers by selectivity mass, top 100 neurons per domain:

| domain | strongest layers |
| --- | --- |
| agriculture | L5 0.087, L17 0.083, L6 0.073 |
| astronomy | L0 0.133, L3 0.115, L4 0.104 |
| math | L4 0.154, L1 0.122, L6 0.109 |
| medicine | L4 0.126, L17 0.114, L3 0.113 |
| politics | L2 0.149, L3 0.123, L1 0.121 |

Interpretation: the neuron selectivity signal is structured by layer and domain. Astronomy, math, and politics concentrate strongly in early layers; agriculture has a more mixed pattern with notable mid/late-layer mass. This supports the idea that neuron activations are decodable, but it does not resolve whether they are causally necessary. A fair neuron ablation should test much larger neuron sets or percentage-matched intervention sizes.

## MMLU Multiple-Choice Findings

Dataset:

- `data/prompt_sets/mmlu_mc.jsonl`
- Imported from `cais/mmlu`
- Local cleaned records: 5,688
- Domains: astronomy, math, medicine, politics
- Splits: discovery 3,411 / validation 1,136 / test 1,141
- Dropped during import: 130 duplicate questions, 4 bad-choice records

MMLU activation discovery:

- Path: `results/mmlu_mc_discovery_heads_cap100.csv`
- Analysis: `results/mmlu_mc_analysis_heads_cap100`
- Capped discovery: up to 100 records per domain
- Domain decodability from attention-head activations: 0.8596

Interpretation: MMLU choice-prompt activations are strongly domain-discriminative.

### Letter Scoring Ablation

Target-domain politics top-5 heads:

- Path: `results/mmlu_mc_ablation_politics_heads_cap100_validation_r10`
- Result: no convincing causal harm; top target did not increase correct-choice loss over controls.

Target-domain astronomy top-5 heads:

- Path: `results/mmlu_mc_ablation_astronomy_heads_cap100_validation_r10`
- Result: no convincing causal harm under letter-only scoring.

Interpretation: scoring only `A/B/C/D` is likely too brittle for this chat model.

### Choice-Text Scoring Ablation

Target-domain astronomy top-5 heads:

- Path: `results/mmlu_mc_ablation_astronomy_heads_cap100_validation_r10_choice_text`

| Metric | Value |
| --- | ---: |
| top delta accuracy | -0.059 |
| top delta correct-choice loss | +0.044 |
| top-minus-random correct-choice loss | +0.053 |
| top delta margin | -0.018 |

Interpretation: choice-text scoring reveals a plausible astronomy harm signal.

Target-domain politics top-5 heads:

- Path: `results/mmlu_mc_ablation_politics_heads_cap100_validation_r10_choice_text`

| Metric | Value |
| --- | ---: |
| top delta accuracy | +0.020 |
| top delta correct-choice loss | -0.026 |
| top-minus-random correct-choice loss | +0.034 |
| top delta margin | +0.005 |

Interpretation: politics does not show a convincing causal harm signal.

### Astronomy Choice-Text k-Curve

Path:

- `results/mmlu_mc_curve_astronomy_heads_choice_text_r30`

MMLU-derived head rankings, validation split, choice-text scoring, 30 layer-matched controls.
Values are point estimates with 95% bootstrap intervals in brackets:

| k heads | accuracy vs random | correct-loss vs random | margin vs random |
| ---: | ---: | ---: | ---: |
| 1 | -0.014 [-0.088, +0.029] | +0.004 [-0.027, +0.053] | -0.006 [-0.023, +0.006] |
| 3 | -0.032 [-0.118, +0.029] | +0.029 [-0.053, +0.097] | -0.018 [-0.059, +0.013] |
| 5 | -0.028 [-0.118, +0.088] | +0.045 [-0.040, +0.124] | -0.018 [-0.061, +0.021] |
| 10 | -0.027 [-0.176, +0.088] | +0.036 [-0.047, +0.119] | -0.003 [-0.062, +0.050] |
| 20 | +0.007 [-0.147, +0.147] | +0.059 [-0.163, +0.283] | +0.041 [-0.036, +0.129] |

Interpretation:

- Correct-choice loss is consistently positive over matched controls.
- Bootstrap intervals cross zero for all metrics, so the result is suggestive rather than statistically clean.
- Accuracy is noisy because astronomy validation has only 34 records.
- Margin is mixed; the curve is not a clean dose-response.
- This is a real benchmark signal, but not yet strong enough for a final paper claim.

### MMLU Answer-Token Activation Smoke Test

Smoke paths:

- `results/mmlu_mc_answer_activations_heads_smoke.csv`
- `results/mmlu_mc_answer_activations_mlp_neurons_smoke.csv`

This is a small teacher-forced activation collection over correct answer text only: 5 discovery records per domain, choice-text continuations, CUDA fp16.

Attention-head smoke:

| domain | strongest answer-token components |
| --- | --- |
| astronomy | L8 H9, L21 H12, L19 H6 |
| math | L3 H5, L4 H8, L20 H4 |
| medicine | L3 H14, L4 H10, L1 H19 |
| politics | L4 H11, L4 H7, L4 H25 |

MLP-neuron smoke:

| domain | strongest answer-token components |
| --- | --- |
| astronomy | L12 N2679, L15 N3811, L14 N5021 |
| math | L0 N1645, L1 N910, L18 N2720 |
| medicine | L1 N5610, L2 N4151, L1 N2575 |
| politics | L4 N4028, L3 N4775, L15 N3933 |

Interpretation: the new answer-token measurement path works. These smoke results are too small for claims, but they show that domain-selective head and neuron signals can be measured during answer scoring, not only after prompt reading.

### MMLU Answer-Token Activation Cap-50

Paths:

- `results/mmlu_mc_answer_activations_heads_cap50.csv`
- `results/mmlu_mc_answer_analysis_heads_cap50`
- `results/mmlu_mc_answer_activations_mlp_neurons_top32_cap50.csv`
- `results/mmlu_mc_answer_analysis_mlp_neurons_top32_cap50`

Teacher-forced answer-token activations over correct choice text only, 50 MMLU discovery records per domain.

| granularity | records | components | leave-one-out domain decodability |
| --- | ---: | ---: | ---: |
| attention head | 140,800 | 704 | 0.735 |
| MLP neuron, top 32 per layer/prompt | 140,800 | 10,488 | 0.845 |

Top answer-token attention heads:

| domain | strongest components |
| --- | --- |
| astronomy | L4 H11, L7 H31, L20 H12 |
| math | L3 H5, L5 H29, L8 H7 |
| medicine | L7 H31, L1 H19, L5 H10 |
| politics | L7 H25, L9 H7, L7 H10 |

Top answer-token MLP neurons:

| domain | strongest components |
| --- | --- |
| astronomy | L1 N4416, L1 N2492, L1 N1697 |
| math | L4 N5015, L1 N262, L0 N2806 |
| medicine | L1 N2371, L4 N3333, L1 N2375 |
| politics | L2 N3437, L2 N792, L1 N1504 |

Interpretation: MLP neuron activations remain more domain-decodable than attention heads during answer scoring, not only after prompt reading. This makes the neuron-ablation result more interesting: neurons carry strong domain information at answer time, yet matched-budget neuron ablation still does not identify compact causal levers.

### MMLU Prompt-vs-Answer Component Overlap

Paths:

- `results/mmlu_mc_prompt_analysis_heads_cap50`
- `results/mmlu_mc_prompt_analysis_mlp_neurons_top32_cap50`
- `results/mmlu_mc_prompt_vs_answer_heads_overlap_cap50.csv`
- `results/mmlu_mc_prompt_vs_answer_mlp_neurons_overlap_top32_cap50.csv`

Prompt-token and answer-token rankings use the same cap-50 MMLU discovery records.

Top-k overlap, prompt-selected versus answer-selected:

| granularity | domain | overlap@10 | Jaccard@10 | overlap@50 | Jaccard@50 |
| --- | --- | ---: | ---: | ---: | ---: |
| attention head | astronomy | 6/10 | 0.429 | 26/50 | 0.351 |
| attention head | math | 1/10 | 0.053 | 19/50 | 0.235 |
| attention head | medicine | 7/10 | 0.538 | 35/50 | 0.538 |
| attention head | politics | 4/10 | 0.250 | 25/50 | 0.333 |
| MLP neuron | astronomy | 0/10 | 0.000 | 1/50 | 0.010 |
| MLP neuron | math | 0/10 | 0.000 | 4/50 | 0.042 |
| MLP neuron | medicine | 0/10 | 0.000 | 0/50 | 0.000 |
| MLP neuron | politics | 0/10 | 0.000 | 2/50 | 0.020 |

Interpretation: attention-head rankings have moderate prompt/answer overlap, while MLP-neuron rankings are almost disjoint. This suggests prompt-time and answer-time neuron selectivity may reflect different populations.

### MMLU Answer-Selected Ablation Pilot

Paths:

- `results/mmlu_mc_answer_selected_astronomy_heads_k5_choice_text_r3_cap25`
- `results/mmlu_mc_prompt_selected_astronomy_heads_k5_choice_text_r3_cap25`
- `results/mmlu_mc_answer_selected_astronomy_neurons_k500_choice_text_r3_cap25`
- `results/mmlu_mc_answer_selected_astronomy_neurons_k880_choice_text_r3_cap25`
- `results/mmlu_mc_prompt_selected_astronomy_neurons_k500_choice_text_r3_cap25`

MMLU validation, capped at 25 records per domain, astronomy target, choice-text scoring, layer-matched controls, 3 repeats. These are exploratory pilots, not final statistical claims.

Astronomy top-minus-control metrics:

| selection time | component | k | accuracy | correct-loss | margin |
| --- | --- | ---: | ---: | ---: | ---: |
| prompt | attention head | 5 | +0.013 | +0.017 | -0.030 |
| answer | attention head | 5 | -0.013 | -0.020 | -0.033 |
| prompt | MLP neuron | 500 | -0.053 | -2.101 | +0.205 |
| answer | MLP neuron | 500 | -0.107 | +1.471 | -0.426 |
| answer | MLP neuron | 880 | -0.027 | +1.496 | -0.319 |

Interpretation: answer-selected MLP neurons behave very differently from prompt-selected MLP neurons in this pilot. Prompt-selected neurons at k=500 are less damaging than controls, while answer-selected neurons at k=500 and k=880 strongly increase correct-choice loss. This is a major lead: the previous negative neuron result may be specific to prompt-selected neuron rankings, while answer-time neuron rankings can identify behaviorally relevant neurons. Because this is cap25/r3, it needs a larger repeat before becoming a paper claim.

### MMLU Answer-Selected Neuron Follow-Up, r10

Paths:

- `results/mmlu_mc_answer_selected_astronomy_neurons_k500_choice_text_r10_cap50`
- `results/mmlu_mc_answer_selected_astronomy_neurons_k880_choice_text_r10_cap50`

MMLU validation, astronomy target, all 34 astronomy validation records and up to 50 records for each non-target domain, choice-text scoring, layer-matched controls, 10 repeats.

Astronomy top-minus-control metrics:

| selection time | component | k | accuracy | correct-loss | margin |
| --- | --- | ---: | ---: | ---: | ---: |
| answer | MLP neuron | 500 | -0.171 | +1.971 | -0.596 |
| answer | MLP neuron | 880 | -0.062 | +1.796 | -0.531 |

Interpretation: the answer-selected neuron effect survives the stronger r10 follow-up. Answer-time MLP neurons selected by astronomy domain selectivity substantially increase correct-choice loss over layer-matched controls. This changes the story: prompt-selected raw MLP neurons are highly decodable but not compact causal levers in QA; answer-selected MLP neurons on MMLU can be behaviorally important. The timing of activation measurement appears central.

## Current Scientific Framing

Best current framing:

> Domain-selective activations can be highly decodable but do not always imply causal task relevance. Synthetic QA shows a clean attention-head ablation effect; MMLU shows a weaker, scoring-dependent astronomy effect and no politics effect.

This is more defensible than claiming that domain knowledge simply "resides" in specific heads.

## Next Recommended Runs

1. Add SciQ or another science multiple-choice dataset to increase astronomy/science validation size.
2. Import SciQ or another science benchmark and check whether the astronomy/science pattern repeats with more validation examples.
3. Repeat MMLU choice-text curves for math and medicine.
4. Test a stronger model, because TinyLlama is weak on MMLU and accuracy is near chance in several domains.
