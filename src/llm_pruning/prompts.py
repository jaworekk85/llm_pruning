from __future__ import annotations

from pathlib import Path

from llm_pruning.prompt_records import group_prompts_by_domain, load_prompt_records


def load_prompt_file(path: Path) -> list[str]:
    prompts: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)

    return prompts


def load_domain_prompts(prompt_dir: Path) -> dict[str, list[str]]:
    if not prompt_dir.exists():
        raise FileNotFoundError(f"Prompt directory does not exist: {prompt_dir}")

    domains: dict[str, list[str]] = {}
    for prompt_file in sorted(prompt_dir.glob("*.txt")):
        prompts = load_prompt_file(prompt_file)
        if prompts:
            domains[prompt_file.stem] = prompts

    if not domains:
        raise ValueError(f"No prompts found in: {prompt_dir}")

    return domains


def load_domain_prompts_jsonl(path: Path, split: str | None = None) -> dict[str, list[str]]:
    records = load_prompt_records(path)
    domains = group_prompts_by_domain(records, split=split)

    if not domains:
        split_text = f" for split '{split}'" if split is not None else ""
        raise ValueError(f"No prompts found in {path}{split_text}")

    return domains


def filter_domains(
    prompts_by_domain: dict[str, list[str]],
    requested_domains: list[str] | None,
) -> dict[str, list[str]]:
    if not requested_domains:
        return prompts_by_domain

    missing = sorted(set(requested_domains) - set(prompts_by_domain))
    if missing:
        available = ", ".join(sorted(prompts_by_domain))
        raise ValueError(
            f"Unknown domain(s): {', '.join(missing)}. Available domains: {available}."
        )

    return {domain: prompts_by_domain[domain] for domain in requested_domains}
