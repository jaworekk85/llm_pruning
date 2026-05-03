from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap multiple-choice ablation results from decision CSVs."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def load_decisions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def k_from_result_dir(path: Path) -> int:
    name = path.name
    marker = "_k"
    if marker not in name:
        raise ValueError(f"Cannot infer k from result directory name: {name}")
    return int(name.rsplit(marker, maxsplit=1)[1])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def metric_delta(
    rows: list[dict[str, str]],
    condition: str,
    baseline_by_record: dict[str, dict[str, str]],
    metric: str,
) -> float:
    deltas: list[float] = []
    for row in rows:
        baseline = baseline_by_record[row["record_id"]]
        if metric == "accuracy":
            value = 1.0 if row["is_correct"] == "True" else 0.0
            baseline_value = 1.0 if baseline["is_correct"] == "True" else 0.0
        else:
            value = float(row[metric])
            baseline_value = float(baseline[metric])
        deltas.append(value - baseline_value)
    return mean(deltas)


def bootstrap_one_result(
    result_dir: Path,
    target_domain: str,
    samples: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    decisions = [
        row
        for row in load_decisions(result_dir / "decisions.csv")
        if row["domain"] == target_domain
    ]
    baseline_rows = [row for row in decisions if row["condition"] == "baseline"]
    top_rows = [row for row in decisions if row["condition"] == "top_target"]
    random_rows = [
        row for row in decisions if row["condition"].startswith("random_control_")
    ]

    baseline_by_record = {row["record_id"]: row for row in baseline_rows}
    top_by_record = {row["record_id"]: row for row in top_rows}
    random_by_condition_record: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in random_rows:
        random_by_condition_record[row["condition"]][row["record_id"]] = row

    record_ids = sorted(baseline_by_record)
    random_conditions = sorted(random_by_condition_record)
    metrics = ["accuracy", "correct_loss", "margin"]
    sampled_values: dict[str, list[float]] = {metric: [] for metric in metrics}

    for _sample_index in range(samples):
        sampled_record_ids = [rng.choice(record_ids) for _ in record_ids]
        sampled_random_condition = rng.choice(random_conditions) if random_conditions else None

        sampled_top_rows = [top_by_record[record_id] for record_id in sampled_record_ids]
        if sampled_random_condition is None:
            sampled_random_rows = []
        else:
            random_by_record = random_by_condition_record[sampled_random_condition]
            sampled_random_rows = [
                random_by_record[record_id]
                for record_id in sampled_record_ids
            ]

        sampled_baseline_by_record = {
            record_id: baseline_by_record[record_id]
            for record_id in sampled_record_ids
        }

        for metric in metrics:
            top_delta = metric_delta(
                sampled_top_rows,
                condition="top_target",
                baseline_by_record=sampled_baseline_by_record,
                metric=metric,
            )
            random_delta = metric_delta(
                sampled_random_rows,
                condition=sampled_random_condition or "",
                baseline_by_record=sampled_baseline_by_record,
                metric=metric,
            )
            sampled_values[metric].append(top_delta - random_delta)

    rows: list[dict[str, object]] = []
    for metric, values in sampled_values.items():
        rows.append(
            {
                "k": k_from_result_dir(result_dir),
                "target_domain": target_domain,
                "metric": metric,
                "estimate_mean": mean(values),
                "ci_lower_95": percentile(values, 0.025),
                "ci_upper_95": percentile(values, 0.975),
                "bootstrap_samples": samples,
                "record_count": len(record_ids),
                "random_control_repeats": len(random_conditions),
                "result_dir": str(result_dir),
            }
        )
    return rows


def write_rows(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.random_seed)
    result_dirs = sorted(
        path
        for path in args.result_root.iterdir()
        if path.is_dir() and (path / "decisions.csv").exists()
    )
    rows: list[dict[str, object]] = []
    for result_dir in result_dirs:
        rows.extend(
            bootstrap_one_result(
                result_dir,
                target_domain=args.target_domain,
                samples=args.samples,
                rng=rng,
            )
        )

    output_csv = args.output_csv or args.result_root / "bootstrap_ci.csv"
    write_rows(rows, output_csv)
    print(f"Wrote bootstrap intervals to {output_csv}")
    for row in rows:
        print(
            f"k={row['k']} {row['metric']}: "
            f"mean={row['estimate_mean']:.4f}, "
            f"95% CI=[{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}]"
        )


if __name__ == "__main__":
    main()
