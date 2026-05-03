# Multiple-Choice Evaluation Methodology

This note records the current benchmark-style protocol for the TinyLlama domain-localization experiments.

## Datasets

`data/prompt_sets/mmlu_mc.jsonl` is imported from `cais/mmlu` using selected subjects grouped into astronomy, math, medicine, and politics. Public MMLU rows are assigned into local `discovery`, `validation`, and `test` splits so that component selection and evaluation can stay separate.

`experiments/import_sciq_mc.py` adds an import path for `allenai/sciq`. SciQ is broader science rather than clean astronomy, so the default local domain is `science`. It is useful mainly for increasing benchmark-style science coverage and for testing whether the MMLU astronomy signal generalizes.

## Prompt And Scoring

Each record is formatted as a multiple-choice question. The model is not asked to generate a long free-form answer during scoring. Instead, we score each candidate answer continuation and choose the candidate with the best score.

Current scoring modes:

- `letter`: score the answer labels, such as `A`, `B`, `C`, and `D`.
- `choice_text`: score the full answer text.

For TinyLlama chat, `choice_text` has been more informative than letter-only scoring. Letter scoring can be brittle because the model may not strongly represent the answer as a single standalone label token.

## Activation Discovery

Activations are measured on the prompt forward pass: the question and choices are in context, and the model is about to score or predict the answer continuation. For attention-head discovery, each head's activation summary is used as a feature for domain discrimination.

This tests whether component activations contain domain information. It does not by itself prove that the component causally stores or uses domain knowledge.

The newer answer-token collector uses teacher forcing: it appends a candidate answer continuation to the prompt, runs one full forward pass, and summarizes only the answer-token positions. This keeps examples comparable while asking whether domain information appears during answer scoring rather than only after reading the prompt.

## Ablation Evaluation

After ranking components on the discovery split, we ablate the top `k` target-domain components on held-out validation examples. We compare this against random controls. For attention heads, the current preferred control is layer-matched random heads: if a selected head comes from layer 12, the control also samples from layer 12.

Metrics:

- `accuracy`: whether the highest-scoring choice is the correct one.
- `correct_loss`: cross-entropy loss of the correct choice text.
- `margin`: score gap between the correct choice and the strongest incorrect choice.

The most stable current metric is `correct_loss`. Accuracy is easy to understand but noisy when the target validation domain has few examples.

## Bootstrap Intervals

Bootstrap confidence intervals quantify how much the measured top-minus-control effect could move if we had sampled a different validation set from the same source. The script resamples validation records with replacement and resamples one random-control repeat, then recomputes the metric difference many times.

Bootstrapping does not create new evidence. It only measures uncertainty in the evidence we already have. With small target-domain validation sets, wide intervals are expected.

## Current Interpretation

The strongest defensible claim is:

> Domain-selective activations can be highly decodable, but decodability does not automatically imply causal task relevance.

Synthetic QA shows a clean attention-head ablation effect. MMLU currently shows a weaker astronomy effect under choice-text scoring, with wide bootstrap intervals. That is suggestive, not yet a final paper-level causal claim.
