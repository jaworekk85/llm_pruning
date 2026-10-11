from __future__ import annotations

import hashlib

from llm_pruning.choice_records import ChoiceRecord


PROMPT_POLICIES = {"legacy_mc_v1", "plain_mc_v1", "tokenizer_chat_v1"}


def prompt_policy_metadata(tokenizer, policy: str, scoring_mode: str) -> dict:
    if policy not in PROMPT_POLICIES:
        raise ValueError(f"Unsupported prompt policy: {policy}")
    if scoring_mode not in {"letter", "choice_text"}:
        raise ValueError(f"Unsupported scoring mode: {scoring_mode}")
    template = None
    if policy == "tokenizer_chat_v1":
        if not tokenizer.chat_template:
            raise ValueError("tokenizer_chat_v1 requires an explicit tokenizer chat template")
        template = tokenizer.get_chat_template()
    return {
        "policy": policy, "scoring_mode": scoring_mode, "instruction_language": "en",
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest() if template else None,
        "add_special_tokens": policy != "tokenizer_chat_v1",
        "assistant_prefill": "Answer:",
    }


def format_choice_prompt(record: ChoiceRecord, *, policy: str = "legacy_mc_v1", tokenizer=None, scoring_mode: str = "letter") -> str:
    if policy not in PROMPT_POLICIES:
        raise ValueError(f"Unsupported prompt policy: {policy}")
    if scoring_mode not in {"letter", "choice_text"}:
        raise ValueError(f"Unsupported scoring mode: {scoring_mode}")
    if len(record.choices) > 26:
        raise ValueError("MC prompts support at most 26 options")
    labels = [chr(ord("A") + index) for index in range(len(record.choices))]
    lines = [
        "<|user|>",
        record.question.strip(),
    ]
    for label, choice in zip(labels, record.choices, strict=True):
        lines.append(f"{label}. {choice}")
    lines.extend(
        [
            "Choose the single best answer. Reply with only the answer letter.",
            "<|assistant|>",
            "Answer:",
        ]
    )
    if policy == "legacy_mc_v1":
        return "\n".join(lines)
    instruction = (
        "Choose the single best answer. Reply with only the answer letter."
        if scoring_mode == "letter" else
        "Choose the single best answer. Reply with the complete text of the correct option."
    )
    content = "\n".join([record.question.strip(), *[f"{label}. {choice}" for label, choice in zip(labels, record.choices, strict=True)], instruction])
    if policy == "plain_mc_v1":
        return content + "\nAnswer:"
    if tokenizer is None or not tokenizer.chat_template:
        raise ValueError("tokenizer_chat_v1 requires an explicit tokenizer chat template")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
    ) + "Answer:"


def candidate_text(record: ChoiceRecord, choice_index: int, scoring_mode: str) -> str:
    if scoring_mode == "letter":
        return f" {chr(ord('A') + choice_index)}"
    if scoring_mode == "choice_text":
        return f" {record.choices[choice_index]}"
    raise ValueError(f"Unsupported scoring mode: {scoring_mode}")
