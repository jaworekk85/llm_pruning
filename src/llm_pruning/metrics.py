from __future__ import annotations

from collections import defaultdict

from llm_pruning.hooks import ActivationRecord


def mean_by_domain_and_module(
    records: list[ActivationRecord],
) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for record in records:
        totals[record.domain][record.module_name] += record.mean_abs
        counts[record.domain][record.module_name] += 1

    means: dict[str, dict[str, float]] = {}
    for domain, module_totals in totals.items():
        means[domain] = {}
        for module_name, total in module_totals.items():
            means[domain][module_name] = total / counts[domain][module_name]

    return means


def selectivity_scores(
    records: list[ActivationRecord],
    target_domain: str,
    eps: float = 1e-8,
) -> list[tuple[str, float, float, float]]:
    means = mean_by_domain_and_module(records)
    if target_domain not in means:
        raise ValueError(f"Unknown target domain: {target_domain}")

    scores: list[tuple[str, float, float, float]] = []
    target_modules = means[target_domain]

    for module_name, target_mean in target_modules.items():
        other_values = [
            module_means[module_name]
            for domain, module_means in means.items()
            if domain != target_domain and module_name in module_means
        ]
        if not other_values:
            continue

        other_mean = sum(other_values) / len(other_values)
        score = (target_mean - other_mean) / (abs(target_mean) + abs(other_mean) + eps)
        scores.append((module_name, score, target_mean, other_mean))

    return sorted(scores, key=lambda item: item[1], reverse=True)
