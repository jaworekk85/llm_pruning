from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.hooks import ActivationRecord
from llm_pruning.metrics import (
    component_scores,
    concentration_scores,
    leave_one_out_domain_decodability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze activation CSVs with localization metrics."
    )
    parser.add_argument("--input-csv", type=Path, default=Path("results/activations.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def parse_optional_int(value: str | None) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(value)


def load_records(input_csv: Path) -> list[ActivationRecord]:
    records: list[ActivationRecord] = []

    with input_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(
                ActivationRecord(
                    domain=row["domain"],
                    prompt_index=int(row["prompt_index"]),
                    granularity=row.get("granularity") or "unknown",
                    module_name=row["module_name"],
                    unit_index=parse_optional_int(row.get("unit_index")),
                    mean_abs=float(row["mean_abs"]),
                    std=float(row["std"]),
                    max_abs=float(row["max_abs"]),
                    numel=int(row["numel"]),
                )
            )

    if not records:
        raise ValueError(f"No activation records found in {input_csv}")

    return records


def write_component_scores(records: list[ActivationRecord], output_path: Path) -> None:
    domains = sorted({record.domain for record in records})

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "domain",
                "component",
                "selectivity",
                "effect_size",
                "target_mean",
                "other_mean",
                "target_std",
                "other_std",
                "target_count",
                "other_count",
            ],
        )
        writer.writeheader()

        for domain in domains:
            for score in component_scores(records, domain):
                writer.writerow(score.__dict__)


def write_concentration(records: list[ActivationRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "domain",
                "score_kind",
                "component_count",
                "positive_signal",
                "entropy",
                "normalized_entropy",
                "effective_components",
                "gini",
                "top_1_share",
                "top_5_share",
                "top_10_share",
            ],
        )
        writer.writeheader()

        for score_kind in ["selectivity", "effect_size"]:
            for score in concentration_scores(records, score_kind=score_kind):
                writer.writerow(score.__dict__)


def write_decodability(records: list[ActivationRecord], output_path: Path) -> None:
    score = leave_one_out_domain_decodability(records)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "accuracy",
                "correct",
                "total",
                "domains",
                "component_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "accuracy": score.accuracy,
                "correct": score.correct,
                "total": score.total,
                "domains": "|".join(score.domains),
                "component_count": score.component_count,
            }
        )


def print_summary(records: list[ActivationRecord], top_k: int) -> None:
    domains = sorted({record.domain for record in records})
    granularities = sorted({record.granularity for record in records})
    components = {
        f"{record.module_name}[{record.unit_index}]"
        if record.unit_index is not None
        else record.module_name
        for record in records
    }

    print(f"Loaded {len(records)} activation records")
    print(f"Granularity: {', '.join(granularities)}")
    print(f"Domains: {', '.join(domains)}")
    print(f"Components: {len(components)}")

    print("\nTop selectivity scores:")
    for domain in domains:
        print(f"\n{domain}")
        for score in component_scores(records, domain)[:top_k]:
            print(
                f"  {score.component}: selectivity={score.selectivity:.4f}, "
                f"effect_size={score.effect_size:.4f}"
            )

    print("\nConcentration by selectivity:")
    for score in concentration_scores(records, score_kind="selectivity"):
        print(
            f"  {score.domain}: normalized_entropy={score.normalized_entropy:.4f}, "
            f"effective_components={score.effective_components:.2f}, "
            f"top_5_share={score.top_5_share:.4f}, gini={score.gini:.4f}"
        )

    decodability = leave_one_out_domain_decodability(records)
    print(
        "\nDomain decodability: "
        f"accuracy={decodability.accuracy:.4f} "
        f"({decodability.correct}/{decodability.total})"
    )


def main() -> None:
    args = parse_args()
    records = load_records(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_component_scores(records, args.output_dir / "component_scores.csv")
    write_concentration(records, args.output_dir / "concentration.csv")
    write_decodability(records, args.output_dir / "decodability.csv")
    print_summary(records, args.top_k)

    print(f"\nWrote analysis CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
