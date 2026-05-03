from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


K_PATTERN = re.compile(r"_k(\d+)$")


DEFAULT_TOTAL_COMPONENTS = {
    "attention_head": 22 * 32,
    "mlp_neuron": 22 * 5632,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ablation curves with comparable component-budget fractions."
    )
    parser.add_argument("--target-domain", required=True)
    parser.add_argument(
        "--curve",
        nargs=3,
        action="append",
        metavar=("LABEL", "GRANULARITY", "RESULT_ROOT"),
        required=True,
        help="Curve to include, for example: heads attention_head results/path",
    )
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def k_from_dir(path: Path) -> int:
    match = K_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot infer k from directory name: {path.name}")
    return int(match.group(1))


def collect_rows(
    label: str,
    granularity: str,
    result_root: Path,
    target_domain: str,
) -> list[dict[str, object]]:
    total_components = DEFAULT_TOTAL_COMPONENTS[granularity]
    rows: list[dict[str, object]] = []
    for result_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        contrast_path = result_dir / "contrasts.csv"
        if not contrast_path.exists():
            continue
        domain_rows = [
            row for row in read_csv(contrast_path) if row["domain"] == target_domain
        ]
        if not domain_rows:
            continue
        row = domain_rows[0]
        k = k_from_dir(result_dir)
        rows.append(
            {
                "label": label,
                "granularity": granularity,
                "k": k,
                "global_fraction": k / total_components,
                "top_delta_loss": float(row["top_target_delta_loss"]),
                "control_mean_delta_loss": float(row["random_control_mean_delta_loss"]),
                "top_minus_control_delta_loss": float(row["top_minus_random_delta_loss"]),
                "control_repeats": int(row["random_control_repeats"]),
                "result_dir": str(result_dir),
            }
        )
    return sorted(rows, key=lambda row: (str(row["label"]), int(row["k"])))


def fmt_float(value: object) -> str:
    return f"{float(value):+.3f}"


def fmt_pct(value: object) -> str:
    return f"{100.0 * float(value):.3f}%"


def write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "| curve | granularity | k | model fraction | top delta loss | control delta loss | top-minus-control |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["granularity"]),
                    str(row["k"]),
                    fmt_pct(row["global_fraction"]),
                    fmt_float(row["top_delta_loss"]),
                    fmt_float(row["control_mean_delta_loss"]),
                    fmt_float(row["top_minus_control_delta_loss"]),
                ]
            )
            + " |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for label, granularity, result_root in args.curve:
        rows.extend(
            collect_rows(
                label=label,
                granularity=granularity,
                result_root=Path(result_root),
                target_domain=args.target_domain,
            )
        )
    if not rows:
        raise SystemExit("No contrast rows found.")

    write_markdown(rows, args.output_md)
    if args.output_csv is not None:
        write_csv(rows, args.output_csv)
    print(f"Wrote budget comparison to {args.output_md}")
    if args.output_csv is not None:
        print(f"Wrote budget comparison CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
