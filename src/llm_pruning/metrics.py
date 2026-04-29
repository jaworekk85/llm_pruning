from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from llm_pruning.hooks import ActivationRecord


@dataclass(frozen=True)
class ComponentScore:
    domain: str
    component: str
    selectivity: float
    effect_size: float
    target_mean: float
    other_mean: float
    target_std: float
    other_std: float
    target_count: int
    other_count: int


@dataclass(frozen=True)
class ConcentrationScore:
    domain: str
    score_kind: str
    component_count: int
    positive_signal: float
    entropy: float
    normalized_entropy: float
    effective_components: float
    gini: float
    top_1_share: float
    top_5_share: float
    top_10_share: float


@dataclass(frozen=True)
class DecodabilityScore:
    accuracy: float
    correct: int
    total: int
    domains: tuple[str, ...]
    component_count: int


def component_key(record: ActivationRecord) -> str:
    if record.unit_index is None:
        return record.module_name
    return f"{record.module_name}[{record.unit_index}]"


def mean_by_domain_and_module(
    records: list[ActivationRecord],
) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for record in records:
        key = component_key(record)
        totals[record.domain][key] += record.mean_abs
        counts[record.domain][key] += 1

    means: dict[str, dict[str, float]] = {}
    for domain, module_totals in totals.items():
        means[domain] = {}
        for key, total in module_totals.items():
            means[domain][key] = total / counts[domain][key]

    return means


def values_by_domain_and_component(
    records: list[ActivationRecord],
) -> dict[str, dict[str, list[float]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        values[record.domain][component_key(record)].append(record.mean_abs)

    return values


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def population_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def pooled_std(target_values: list[float], other_values: list[float]) -> float:
    target_var = population_std(target_values) ** 2
    other_var = population_std(other_values) ** 2
    target_n = len(target_values)
    other_n = len(other_values)
    total_n = target_n + other_n

    if total_n == 0:
        return 0.0

    return math.sqrt(((target_n * target_var) + (other_n * other_var)) / total_n)


def component_scores(
    records: list[ActivationRecord],
    target_domain: str,
    eps: float = 1e-8,
) -> list[ComponentScore]:
    values = values_by_domain_and_component(records)
    if target_domain not in values:
        raise ValueError(f"Unknown target domain: {target_domain}")

    scores: list[ComponentScore] = []
    target_components = values[target_domain]

    for component, target_values in target_components.items():
        other_values = [
            value
            for domain, component_values in values.items()
            if domain != target_domain
            for value in component_values.get(component, [])
        ]
        if not other_values:
            continue

        target_mean = mean(target_values)
        other_mean = mean(other_values)
        target_std = population_std(target_values)
        other_std = population_std(other_values)
        std = pooled_std(target_values, other_values)
        selectivity = (target_mean - other_mean) / (
            abs(target_mean) + abs(other_mean) + eps
        )
        effect_size = 0.0 if std <= eps else (target_mean - other_mean) / std

        scores.append(
            ComponentScore(
                domain=target_domain,
                component=component,
                selectivity=selectivity,
                effect_size=effect_size,
                target_mean=target_mean,
                other_mean=other_mean,
                target_std=target_std,
                other_std=other_std,
                target_count=len(target_values),
                other_count=len(other_values),
            )
        )

    return sorted(scores, key=lambda score: score.selectivity, reverse=True)


def selectivity_scores(
    records: list[ActivationRecord],
    target_domain: str,
    eps: float = 1e-8,
) -> list[tuple[str, float, float, float]]:
    del eps
    return [
        (score.component, score.selectivity, score.target_mean, score.other_mean)
        for score in component_scores(records, target_domain)
    ]


def gini(values: list[float]) -> float:
    positive_values = sorted(value for value in values if value > 0)
    n = len(positive_values)
    if n == 0:
        return 0.0

    total = sum(positive_values)
    if total == 0:
        return 0.0

    weighted_sum = sum((index + 1) * value for index, value in enumerate(positive_values))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def entropy(values: list[float], eps: float = 1e-12) -> float:
    positive_values = [max(value, 0.0) for value in values]
    total = sum(positive_values)
    if total <= eps:
        return 0.0

    probabilities = [value / total for value in positive_values if value > eps]
    return -sum(probability * math.log(probability) for probability in probabilities)


def top_fraction_share(values: list[float], fraction: float) -> float:
    positive_values = sorted((value for value in values if value > 0), reverse=True)
    if not positive_values:
        return 0.0

    total = sum(positive_values)
    if total == 0:
        return 0.0

    k = max(1, math.ceil(len(positive_values) * fraction))
    return sum(positive_values[:k]) / total


def concentration_scores(
    records: list[ActivationRecord],
    score_kind: str = "selectivity",
) -> list[ConcentrationScore]:
    if score_kind not in {"selectivity", "effect_size"}:
        raise ValueError("score_kind must be either 'selectivity' or 'effect_size'.")

    domains = sorted({record.domain for record in records})
    concentrations: list[ConcentrationScore] = []

    for domain in domains:
        scores = component_scores(records, domain)
        values = [
            max(getattr(score, score_kind), 0.0)
            for score in scores
        ]
        positive_signal = sum(values)
        raw_entropy = entropy(values)
        component_count = len(values)
        normalized_entropy = (
            raw_entropy / math.log(component_count) if component_count > 1 else 0.0
        )

        concentrations.append(
            ConcentrationScore(
                domain=domain,
                score_kind=score_kind,
                component_count=component_count,
                positive_signal=positive_signal,
                entropy=raw_entropy,
                normalized_entropy=normalized_entropy,
                effective_components=math.exp(raw_entropy),
                gini=gini(values),
                top_1_share=top_fraction_share(values, 0.01),
                top_5_share=top_fraction_share(values, 0.05),
                top_10_share=top_fraction_share(values, 0.10),
            )
        )

    return concentrations


def prompt_vectors(
    records: list[ActivationRecord],
) -> tuple[list[tuple[str, int]], list[str], list[list[float]]]:
    samples = sorted({(record.domain, record.prompt_index) for record in records})
    components = sorted({component_key(record) for record in records})
    component_index = {component: index for index, component in enumerate(components)}
    sample_index = {sample: index for index, sample in enumerate(samples)}
    vectors = [[0.0 for _ in components] for _ in samples]

    for record in records:
        row = sample_index[(record.domain, record.prompt_index)]
        col = component_index[component_key(record)]
        vectors[row][col] = record.mean_abs

    return samples, components, vectors


def squared_distance(left: list[float], right: list[float]) -> float:
    return sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right))


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []

    width = len(vectors[0])
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(width)
    ]


def leave_one_out_domain_decodability(records: list[ActivationRecord]) -> DecodabilityScore:
    samples, components, vectors = prompt_vectors(records)
    domains = tuple(sorted({domain for domain, _prompt_index in samples}))

    if len(samples) < 2 or len(domains) < 2:
        return DecodabilityScore(
            accuracy=0.0,
            correct=0,
            total=0,
            domains=domains,
            component_count=len(components),
        )

    correct = 0
    total = 0

    for test_index, (true_domain, _prompt_index) in enumerate(samples):
        train_by_domain: dict[str, list[list[float]]] = defaultdict(list)
        for train_index, (domain, _train_prompt_index) in enumerate(samples):
            if train_index == test_index:
                continue
            train_by_domain[domain].append(vectors[train_index])

        centroids = {
            domain: centroid(domain_vectors)
            for domain, domain_vectors in train_by_domain.items()
            if domain_vectors
        }
        if len(centroids) < 2:
            continue

        prediction = min(
            centroids,
            key=lambda domain: squared_distance(vectors[test_index], centroids[domain]),
        )
        correct += int(prediction == true_domain)
        total += 1

    accuracy = correct / total if total else 0.0
    return DecodabilityScore(
        accuracy=accuracy,
        correct=correct,
        total=total,
        domains=domains,
        component_count=len(components),
    )
