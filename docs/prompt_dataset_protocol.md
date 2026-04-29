# Prompt Dataset Protocol

The prompt dataset is part of the experiment, not a disposable input file. Every prompt should be reproducible, traceable, and assigned to a split before any activation-based conclusions are drawn.

## Prompt Record Format

Use JSONL, one prompt per line. Required fields:

```json
{
  "id": "agriculture.seed.0001",
  "domain": "agriculture",
  "prompt": "What is agriculture?",
  "split": "discovery",
  "source_type": "manual_seed",
  "source_name": "initial_repo_seed",
  "prompt_type": "definition",
  "difficulty": "basic",
  "language": "en",
  "target": "Agriculture is the practice of growing crops and raising animals.",
  "license": "project",
  "notes": "Seed prompt for code-path testing."
}
```

Allowed splits:

- `discovery`: used to discover candidate localized components.
- `validation`: used to check whether discovered components generalize.
- `test`: held out until the end.

## Recommended Organization

```text
data/
  prompt_sets/
    seed.jsonl
    seed_qa.jsonl
    qa_v1.jsonl
    discovery_v1.jsonl
    validation_v1.jsonl
    test_v1.jsonl
  prompts/
    agriculture.txt  # legacy/simple format
```

The legacy `data/prompts/*.txt` files are fine for fast smoke tests. Paper experiments should use JSONL prompt sets.

For causal ablation and evaluation experiments, prefer answer-bearing JSONL records with a `target` field. This allows the experiment to score the answer text rather than the question prompt.

`qa_v1.jsonl` is a deterministic project-generated dataset built from manual concept blueprints:

```powershell
python experiments/build_qa_v1.py --output data/prompt_sets/qa_v1.jsonl
python experiments/validate_prompts.py --prompt-jsonl data/prompt_sets/qa_v1.jsonl --require-targets
```

Current `qa_v1` balance:

- 5 domains
- 50 records per domain
- 250 total records
- 30 discovery, 10 validation, 10 test records per domain
- 5 prompt types, 10 records per prompt type per domain

This is a stronger engineering dataset than the seed files, but it is still generated from local project blueprints. It is suitable for pipeline development and pilot analysis, not yet for final paper claims.

## Collection Strategy

Do not aim for "as many prompts as possible" first. Aim for balanced prompts that can support causal claims.

Recommended first serious target after `qa_v1`:

- 8-12 domains.
- 200-500 prompts per domain.
- Balanced prompt types per domain:
  - definitions
  - factual questions
  - explanations
  - comparisons
  - simple reasoning
  - procedural questions
- Balanced difficulty:
  - basic
  - intermediate
  - advanced

## Candidate Source Pools

Possible public source pools to investigate:

- [MMLU](https://arxiv.org/abs/2009.03300), especially subject-labeled multiple-choice questions.
- [BIG-bench](https://arxiv.org/abs/2206.04615), for diverse task families.
- [TruthfulQA](https://arxiv.org/abs/2109.07958), useful as an adversarial factuality-oriented source, not as a broad domain dataset.
- Open textbooks, government educational pages, and public-domain educational resources for domain-specific question generation.

Every imported source must be recorded with source name, source type, license, and any preprocessing rules.

## Bias Controls

The dataset should avoid trivial style shortcuts:

- Do not let one domain be mostly "Explain..." while another is mostly "What is...".
- Keep prompt length distributions similar across domains.
- Keep difficulty distributions similar across domains.
- Avoid domain labels inside prompts unless naturally necessary.
- Avoid source-specific formatting that appears only in one domain.
- Keep generated prompts separate from human-written or benchmark-derived prompts in metadata.

## Reproducible Generation Rule

If LLM-generated prompts are used, store:

- generator model name and version
- generation date
- generation prompt/template
- random seed, if available
- filtering rules
- deduplication rules
- manual review criteria

Generated prompts should not be mixed with benchmark-derived prompts without source metadata.

## Workflow

1. Define domain taxonomy.
2. Collect or generate candidate prompts with metadata.
3. Validate schema and duplicates:

   ```powershell
   python experiments/validate_prompts.py --prompt-jsonl data/prompt_sets/seed.jsonl
   ```

4. Split into discovery, validation, and test before analysis.
5. Use discovery prompts to rank components.
6. Use validation prompts to check ranking stability.
7. Use test prompts only for final claims.
