# Paper Result Index

This file is the decision layer above `docs/experiment_log.md`. It tracks which results are ready for paper figures/tables, which are supporting evidence, and which remain exploratory.

## Current Thesis

Working claim:

> Domain-selective activations can be highly decodable without being compact causal levers for task performance.

Best-supported subclaim:

> In TinyLlama on QA v1, attention heads selected by domain selectivity causally increase target-domain loss under ablation, while raw MLP neurons remain more domain-decodable but are less damaging than matched controls, even at a matched component budget.

## Candidate Main Results

| ID | Result | Status | Use In Paper? | Why |
| --- | --- | --- | --- | --- |
| R1 | QA v1 agriculture attention-head ablation curve | strong | yes | Clean positive dose-response with layer-matched controls. |
| R2 | QA v1 large agriculture attention-head curve | strong | yes | Replicates R1 on larger synthetic validation. |
| R3 | QA v1 MLP neuron small/large-k ablations | strong negative | yes | Shows neuron selectivity does not translate into causal damage. |
| R4 | QA v1 matched-budget neuron ablation, k=880 | strong negative | yes | Addresses the fairness objection: 880 neurons matches 5 heads by global component fraction. |
| R5 | QA v1 MLP neuron decodability, top128 | strong support | yes | Shows neurons are highly domain-decodable, so the negative ablation is not due to absence of signal. |
| R6 | Politics MLP neuron large-k replication | moderate support | maybe | Layer-matched controls agree with agriculture, but random controls are noisy. |
| R7 | MMLU astronomy choice-text head ablation | suggestive | maybe | Real benchmark signal, but bootstrap intervals cross zero. |
| R8 | MMLU answer-token activations cap50 | support/exploratory | maybe | Shows answer-time neuron decodability, but no answer-token-based ablation yet. |
| R9 | MLP neuron layer heatmaps | descriptive | supporting figure | Useful visualization, not causal evidence. |
| R10 | QA v1 large matched-budget neuron check | strong support | yes | Replicates the matched-budget negative neuron result on larger synthetic data. |
| R11 | Math matched-budget check | mixed | maybe | Replicates negative neuron result, but attention-head effect is weak for math. |
| R12 | Prompt-vs-answer component overlap | strong descriptive | yes | Heads overlap moderately; MLP neuron rankings are nearly disjoint. |
| R13 | MMLU answer-selected neuron ablation | promising | maybe/yes after bootstrap | Answer-selected MLP neurons strongly affect MMLU astronomy and survive r10 follow-up. |

## Main Figure Candidates

### Figure 1: Probing-Ablation Gap

Use:

- QA v1 head curve, layer-matched controls.
- QA v1 neuron curve, layer-matched controls.
- Matched-budget point: 5 heads vs 880 MLP neurons.

Message: selected heads cause target-domain loss increases; selected raw neurons do not, despite strong domain decodability.

### Figure 2: Neurons Are Still Decodable

Use:

- QA v1 top128 MLP neuron decodability: 0.920.
- QA v1 large top128 MLP neuron decodability: 0.968.
- MMLU answer-token MLP neuron decodability: 0.845.
- MMLU answer-token head decodability: 0.735.

Message: the negative neuron ablation is not because neurons lack domain information.

### Figure 3: Benchmark Caution And Timing

Use:

- MMLU astronomy choice-text k-curve with bootstrap intervals.
- MMLU answer-selected neuron ablation r10 as the strongest benchmark lead.

Message: benchmark evidence suggests timing matters: prompt-selected head effects are weak/noisy, but answer-selected MLP neurons can be behaviorally important.

### Figure 4: Prompt-Time and Answer-Time Select Different Neurons

Use:

- Prompt-vs-answer overlap table for MMLU cap50.
- Heads: moderate overlap.
- MLP neurons: near-zero top-10 overlap and very low top-50 overlap.

Message: the timing of activation measurement matters, especially for raw MLP neurons.

## Core Tables

### Table A: QA v1 Head-vs-Neuron Matched Budget

| component | k | model fraction | top-minus-control loss |
| --- | ---: | ---: | ---: |
| attention heads | 5 | 0.710% | +0.117 |
| MLP neurons | 880 | 0.710% | -0.632 |

Source:

- `results/qa_v1_agriculture_head_neuron_budget_comparison_top128.md`

### Table B: Answer-Token Domain Decodability

| granularity | component count | decodability |
| --- | ---: | ---: |
| attention heads | 704 | 0.735 |
| MLP neurons | 10,488 | 0.845 |

Source:

- `results/mmlu_mc_answer_analysis_heads_cap50/decodability.csv`
- `results/mmlu_mc_answer_analysis_mlp_neurons_top32_cap50/decodability.csv`

## Important Caveats

- QA v1 is synthetic and should be framed as controlled evidence, not final benchmark evidence.
- MMLU astronomy validation has only 34 target-domain examples, so confidence intervals are wide.
- Raw neuron units may be the wrong mechanistic unit; sparse features or directions may be more causal than individual neurons.
- Zero-ablation is a blunt intervention and may create off-manifold behavior, especially for large random neuron sets.
- TinyLlama is small and weak; a second model is needed before making broad claims.

## Next Paper-Leverage Runs

1. Bootstrap the MMLU answer-selected neuron r10 results over records and controls.
2. Repeat answer-selected neurons on another MMLU/SciQ domain.
3. Add bootstrap confidence intervals for QA v1 ablation curves.
4. Add a second model, preferably Mistral 7B on a server GPU.
5. Run MMLU/SciQ with more target-domain validation examples.

## Do Not Overclaim

Avoid:

> We found where domain knowledge lives.

Prefer:

> Domain-selective information is present in component activations, but selectivity and causal task relevance can diverge sharply.
