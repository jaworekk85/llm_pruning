from __future__ import annotations

from llm_pruning.choice_records import ChoiceRecord


def format_choice_prompt(record: ChoiceRecord) -> str:
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
    return "\n".join(lines)


def candidate_text(record: ChoiceRecord, choice_index: int, scoring_mode: str) -> str:
    if scoring_mode == "letter":
        return f" {chr(ord('A') + choice_index)}"
    if scoring_mode == "choice_text":
        return f" {record.choices[choice_index]}"
    raise ValueError(f"Unsupported scoring mode: {scoring_mode}")
