"""Checked prompt/continuation tokenization shared by measurement and scoring."""
from typing import Sequence

import torch


def tokenize_prompt_answers(tokenizer, prompts: Sequence[str], answers: Sequence[str], device, *, add_special_tokens: bool = True):
    """Full teacher-forced sequences, exact-prefix boundaries, no truncation.

    Chat-rendered strings must use add_special_tokens=False because the template
    already owns special tokens. Fail on cross-boundary merges or prompt suffixes.
    """
    if not prompts or len(prompts) != len(answers):
        raise ValueError("Provide matching nonempty prompt/answer batches")
    if any(not isinstance(text, str) or not text.strip() for text in (*prompts, *answers)):
        raise ValueError("Prompts and answers must be nonempty strings")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires a pad token or eos token")
        tokenizer.pad_token = tokenizer.eos_token
    prefixes = tokenizer(list(prompts), padding=False, truncation=False, add_special_tokens=add_special_tokens)["input_ids"]
    encoded = tokenizer(
        [prompt + answer for prompt, answer in zip(prompts, answers, strict=True)],
        return_tensors="pt", padding=True, truncation=False, add_special_tokens=add_special_tokens,
    ).to(device)
    ids, attention = encoded["input_ids"], encoded["attention_mask"]
    prompt_mask = torch.zeros_like(attention, dtype=torch.bool)
    for row, prefix in enumerate(prefixes):
        positions = attention[row].nonzero(as_tuple=True)[0]
        if not prefix or len(prefix) >= len(positions) or not torch.equal(
            ids[row, positions[:len(prefix)]], torch.tensor(prefix, device=ids.device, dtype=ids.dtype),
        ):
            raise ValueError(f"Unstable prompt/answer token boundary at batch row {row}")
        prompt_mask[row, positions[:len(prefix)]] = True
    position_ids = attention.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention == 0, 0)
    return {
        "input_ids": ids, "attention_mask": attention, "position_ids": position_ids,
    }, {"prompt": prompt_mask, "answer": attention.bool() & ~prompt_mask}
