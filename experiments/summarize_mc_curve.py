from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_LABELS = {
    "accuracy": "accuracy vs random",
    "correct_loss": "correct-loss vs random",
    "margin": "margin vs random",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact Markdown/CSV table for an MC ablation k-curve."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def k_from_result_dir(path: Path) -> int:
    return int(path.name.rsplit("_k", maxsplit=1)[1])


def load_contrasts(result_root: Path, target_domain: str) -> dict[tuple[int, str], float]:
    values: dict[tuple[int, str], float] = {}
    metric_columns = {
        "accuracy": "top_minus_random_delta_accuracy",
        "correct_loss": "top_minus_random_delta_correct_loss",
        "margin": "top_minus_random_delta_margin",
    }
    for result_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        contrast_path = result_dir / "contrasts.csv"
        if not contrast_path.exists():
            continue
        k = k_from_result_dir(result_dir)
        for row in read_csv(contrast_path):
            if row["domain"] != target_domain:
                continue
            for metric, column in metric_columns.items():
                values[(k, metric)] = float(row[column])
    return values


def load_bootstraps(result_root: Path, target_domain: str) -> dict[tuple[int, str], dict[str, float]]:
    bootstrap_path = result_root / "bootstrap_ci.csv"
    if not bootstrap_path.exists():
        return {}

    values: dict[tuple[int, str], dict[str, float]] = {}
    for row in read_csv(bootstrap_path):
        if row["target_domain"] != target_domain:
            continue
        values[(int(row["k"]), row["metric"])] = {
            "mean": float(row["estimate_mean"]),
            "lower": float(row["ci_lower_95"]),
            "upper": float(row["ci_upper_95"]),
        }
    return values


def format_number(value: float) -> str:
    return f"{value:+.3f}"


def format_metric(
    contrasts: dict[tuple[int, str], float],
    bootstraps: dict[tuple[int, str], dict[str, float]],
    k: int,
    metric: str,
) -> str:
    point = contrasts[(k, metric)]
    interval = bootstraps.get((k, metric))
    if interval is None:
        return format_number(point)
    return (
        f"{format_number(point)} "
        f"[{format_number(interval['lower'])}, {format_number(interval['upper'])}]"
    )


def build_rows(
    contrasts: dict[tuple[int, str], float],
    bootstraps: dict[tuple[int, str], dict[str, float]],
) -> list[dict[str, str]]:
    ks = sorted({k for k, _metric in contrasts})
    rows: list[dict[str, str]] = []
    for k in ks:
        row = {"k heads": str(k)}
        for metric, label in METRIC_LABELS.items():
            row[label] = format_metric(contrasts, bootstraps, k, metric)
        rows.append(row)
    return rows


def write_markdown(rows: list[dict[str, str]], output_path: Path) -> None:
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _header in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    contrasts = load_contrasts(args.result_root, args.target_domain)
    if not contrasts:
        raise SystemExit(f"No contrast rows found for domain '{args.target_domain}'.")

    bootstraps = load_bootstraps(args.result_root, args.target_domain)
    rows = build_rows(contrasts, bootstraps)

    output_md = args.output_md or args.result_root / "summary_table.md"
    output_csv = args.output_csv or args.result_root / "summary_table.csv"
    write_markdown(rows, output_md)
    write_csv(rows, output_csv)

    print(f"Wrote Markdown table to {output_md}")
    print(f"Wrote CSV table to {output_csv}")


if __name__ == "__main__":
    main()
