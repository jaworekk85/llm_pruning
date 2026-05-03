from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare top-k component overlap between two component-score CSVs."
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--ks", nargs="*", type=int, default=[5, 10, 20, 50, 100])
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_scores(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def top_components(rows: list[dict[str, str]], domain: str, k: int) -> list[str]:
    domain_rows = [
        row
        for row in rows
        if row["domain"] == domain and float(row["selectivity"]) > 0.0
    ]
    ranked = sorted(domain_rows, key=lambda row: float(row["selectivity"]), reverse=True)
    return [row["component"] for row in ranked[:k]]


def compare(left_rows: list[dict[str, str]], right_rows: list[dict[str, str]], ks: list[int]):
    domains = sorted({row["domain"] for row in left_rows} & {row["domain"] for row in right_rows})
    output_rows: list[dict[str, object]] = []
    for domain in domains:
        for k in ks:
            left_top = top_components(left_rows, domain, k)
            right_top = top_components(right_rows, domain, k)
            left_set = set(left_top)
            right_set = set(right_top)
            intersection = left_set & right_set
            union = left_set | right_set
            output_rows.append(
                {
                    "domain": domain,
                    "k": k,
                    "left_count": len(left_set),
                    "right_count": len(right_set),
                    "overlap_count": len(intersection),
                    "left_overlap_fraction": len(intersection) / len(left_set) if left_set else 0.0,
                    "right_overlap_fraction": len(intersection) / len(right_set) if right_set else 0.0,
                    "jaccard": len(intersection) / len(union) if union else 0.0,
                    "overlap_components": "|".join(sorted(intersection)),
                }
            )
    return output_rows


def write_rows(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = compare(read_scores(args.left), read_scores(args.right), args.ks)
    write_rows(rows, args.output_csv)
    print(
        f"Wrote {args.left_label} vs {args.right_label} overlap to {args.output_csv}"
    )
    for row in rows:
        if int(row["k"]) in {10, 50}:
            print(
                f"{row['domain']} k={row['k']}: overlap={row['overlap_count']}, "
                f"jaccard={float(row['jaccard']):.3f}"
            )


if __name__ == "__main__":
    main()
