from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.models import DEFAULT_MODEL_NAME, format_chat_prompt, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small generation sanity check.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--prompt",
        default="Explain agriculture in simple terms.",
        help="Question to send to the chat model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_model(args.model_name, device=args.device, dtype=args.dtype)

    prompt = format_chat_prompt(args.prompt)
    inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)

    with torch.no_grad():
        outputs = loaded.model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    text = loaded.tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("\nRESULT:\n")
    print(text)


if __name__ == "__main__":
    main()
