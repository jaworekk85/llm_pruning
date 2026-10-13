"""Strict integer short answers and greedy generation; no answer extraction by guessing."""
from __future__ import annotations

import hashlib
import re
import unicodedata

import torch


def parse_integer(text: str) -> int | None:
    """Accept one signed decimal integer, whitespace and optional terminal period.

    Explanations, units, multiple numbers, decimals, separators and empty output
    are malformed. A wrong but well-formed integer remains a valid prediction.
    """
    if not isinstance(text, str):
        return None
    value = unicodedata.normalize("NFKC", text).strip()
    if not re.fullmatch(r"[+-]?[0-9]{1,18}\.?", value):
        return None
    return int(value.removesuffix("."))


def short_answer_protocol(tokenizer, policy: str) -> dict:
    if policy not in {"plain_short_v1", "tokenizer_chat_v1"}:
        raise ValueError("Use plain_short_v1 or tokenizer_chat_v1")
    template = None
    if policy == "tokenizer_chat_v1":
        if not tokenizer.chat_template:
            raise ValueError("Native chat requires an explicit tokenizer chat template")
        template = tokenizer.get_chat_template()
    return {"policy": policy, "add_special_tokens": template is None,
            "template_sha256": hashlib.sha256(template.encode()).hexdigest() if template else None,
            "instruction": "Reply with one integer only.", "parser": "strict_integer_v1",
            "decoding": "greedy", "instruction_language": "en"}


def format_short_prompt(prompt: str, tokenizer, policy: str) -> str:
    protocol = short_answer_protocol(tokenizer, policy)
    content = prompt.strip() + "\n" + protocol["instruction"]
    if policy == "plain_short_v1":
        return content + "\nAnswer:"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
    )


def encode_short_prompts(loaded, prompts: list[str], policy: str, *, reserve_tokens: int = 0):
    """Left-pad without truncation; use the same formatting for calibration/generation."""
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Nonempty prompts required")
    tokenizer = loaded.tokenizer
    protocol = short_answer_protocol(tokenizer, policy)
    original_side, original_pad = tokenizer.padding_side, tokenizer.pad_token
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Generation requires a pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        texts = [format_short_prompt(p, tokenizer, policy) for p in prompts]
        encoded = tokenizer(texts, padding=True, truncation=False,
                            add_special_tokens=protocol["add_special_tokens"], return_tensors="pt")
        limit = getattr(loaded.model.config, "max_position_embeddings", None)
        if limit is not None and encoded["input_ids"].shape[1] + reserve_tokens > limit:
            raise ValueError("Prompt plus generation exceeds model context; truncation is forbidden")
        if not encoded["attention_mask"].bool().any(dim=1).all():
            raise ValueError("Tokenized prompts must be nonempty")
        return {key: encoded[key] for key in ("input_ids", "attention_mask")}
    finally:
        tokenizer.padding_side, tokenizer.pad_token = original_side, original_pad


@torch.inference_mode()
def generate_short_answers(loaded, prompts, *, policy, batch_size=8, max_new_tokens=16):
    if loaded.model.training:
        raise ValueError("Generation requires an eval model")
    if type(batch_size) is not int or batch_size < 1 or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("Positive integer batch_size/max_new_tokens required")
    outputs = []
    for start in range(0, len(prompts), batch_size):
        encoded = encode_short_prompts(loaded, prompts[start:start + batch_size], policy,
                                       reserve_tokens=max_new_tokens)
        width = encoded["input_ids"].shape[1]
        pad = loaded.tokenizer.pad_token_id
        if pad is None:
            pad = loaded.tokenizer.eos_token_id
        generated = loaded.model.generate(
            **{k: v.to(loaded.device) for k, v in encoded.items()}, do_sample=False,
            num_beams=1, num_return_sequences=1, max_new_tokens=max_new_tokens,
            pad_token_id=pad, use_cache=True,
        )
        texts = loaded.tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
        if len(texts) != len(encoded["input_ids"]):
            raise ValueError("Generation returned an unexpected number of answers")
        outputs.extend(texts)
    return outputs
