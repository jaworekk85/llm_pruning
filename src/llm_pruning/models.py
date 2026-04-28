from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@dataclass(frozen=True)
class LoadedModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    try:
        return dtype_map[dtype]
    except KeyError as exc:
        valid = ", ".join(sorted([*dtype_map, "auto"]))
        raise ValueError(f"Unsupported dtype '{dtype}'. Use one of: {valid}.") from exc


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    dtype: str = "auto",
) -> LoadedModel:
    resolved_device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, resolved_device)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
    )
    model.to(resolved_device)
    model.eval()

    return LoadedModel(model=model, tokenizer=tokenizer, device=resolved_device)


def format_chat_prompt(question: str) -> str:
    return f"<|user|>\n{question.strip()}\n<|assistant|>\n"
