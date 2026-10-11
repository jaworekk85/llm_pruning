"""Reusable multiple-choice likelihood scoring and decisions (no model downloads)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from llm_pruning.choice_records import ChoiceRecord
from llm_pruning.mc_prompts import candidate_text, format_choice_prompt, prompt_policy_metadata
from llm_pruning.token_regions import tokenize_prompt_answers


MC_METRIC_DEFINITIONS = {
    "accuracy": "Fraction of examples whose highest mean candidate token log-probability is the correct choice; ties use choice order.",
    "correct_loss": "Mean per-example negative mean token log-probability of the correct continuation (nats).",
    "margin": "Mean per-example correct mean token log-probability minus best incorrect mean token log-probability (nats).",
    "aggregation": "Each example has equal weight within its target-field value; all candidates use the configured scoring_mode.",
}


@dataclass(frozen=True)
class CandidateScore:
    record_id: str
    condition: str
    domain: str
    subject: str
    split: str
    choice_index: int
    is_correct: bool
    mean_logprob: float
    sum_logprob: float
    token_count: int


@dataclass(frozen=True)
class RecordDecision:
    record_id: str
    condition: str
    domain: str
    subject: str
    split: str
    answer_index: int
    predicted_index: int
    is_correct: bool
    correct_mean_logprob: float
    correct_loss: float
    margin: float


def ensure_padding_token(loaded) -> None:
    if loaded.tokenizer.pad_token is None:
        loaded.tokenizer.pad_token = loaded.tokenizer.eos_token


def batched(items: list[tuple[ChoiceRecord, int]], batch_size: int) -> list[list[tuple[ChoiceRecord, int]]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def score_candidate_batch(
    loaded,
    candidates: list[tuple[ChoiceRecord, int]],
    condition: str,
    scoring_mode: str,
    *, prompt_policy: str = "legacy_mc_v1",
) -> list[CandidateScore]:
    if not candidates:
        return []
    protocol = prompt_policy_metadata(loaded.tokenizer, prompt_policy, scoring_mode)
    prompts = [format_choice_prompt(record, policy=prompt_policy, tokenizer=loaded.tokenizer, scoring_mode=scoring_mode) for record, _choice_index in candidates]
    continuations = [
        candidate_text(record, choice_index, scoring_mode)
        for record, choice_index in candidates
    ]
    inputs, masks = tokenize_prompt_answers(
        loaded.tokenizer, prompts, continuations, loaded.device,
        add_special_tokens=protocol["add_special_tokens"],
    )
    limit = getattr(loaded.model.config, "max_position_embeddings", None)
    if limit is not None and inputs["input_ids"].shape[1] > limit:
        raise ValueError("Sequence exceeds model context length; truncation is not allowed")
    labels = inputs["input_ids"].clone()
    labels[~masks["answer"]] = -100

    with torch.no_grad():
        logits = loaded.model(**inputs, use_cache=False).logits

    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)

    scores: list[CandidateScore] = []
    valid_tokens = shift_labels.ne(-100)
    for row_index, (record, choice_index) in enumerate(candidates):
        token_count = int(valid_tokens[row_index].sum().item())
        if token_count == 0:
            raise ValueError(f"No scored tokens for {record.id} choice {choice_index}")

        token_loss_sum = float(losses[row_index][valid_tokens[row_index]].sum().item())
        if not math.isfinite(token_loss_sum):
            raise ValueError(f"Non-finite candidate loss for {record.id} choice {choice_index}")
        mean_logprob = -token_loss_sum / token_count
        scores.append(
            CandidateScore(
                record_id=record.id,
                condition=condition,
                domain=record.domain,
                subject=record.subject,
                split=record.split,
                choice_index=choice_index,
                is_correct=choice_index == record.answer_index,
                mean_logprob=mean_logprob,
                sum_logprob=-token_loss_sum,
                token_count=token_count,
            )
        )
    return scores


def decisions_from_scores(scores: list[CandidateScore], records_by_id: dict[str, ChoiceRecord]) -> list[RecordDecision]:
    grouped: dict[tuple[str, str], list[CandidateScore]] = {}
    for score in scores:
        grouped.setdefault((score.condition, score.record_id), []).append(score)

    decisions: list[RecordDecision] = []
    for (condition, record_id), candidate_scores in grouped.items():
        record = records_by_id[record_id]
        ranked = sorted(candidate_scores, key=lambda score: score.mean_logprob, reverse=True)
        predicted = ranked[0]
        correct = [
            score
            for score in candidate_scores
            if score.choice_index == record.answer_index
        ][0]
        best_wrong = max(
            score.mean_logprob
            for score in candidate_scores
            if score.choice_index != record.answer_index
        )
        decisions.append(
            RecordDecision(
                record_id=record_id,
                condition=condition,
                domain=record.domain,
                subject=record.subject,
                split=record.split,
                answer_index=record.answer_index,
                predicted_index=predicted.choice_index,
                is_correct=predicted.choice_index == record.answer_index,
                correct_mean_logprob=correct.mean_logprob,
                correct_loss=-correct.mean_logprob,
                margin=correct.mean_logprob - best_wrong,
            )
        )
    return decisions
