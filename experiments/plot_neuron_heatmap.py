from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


LAYER_PATTERN = re.compile(r"model\.layers\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a layer-by-domain heatmap from MLP neuron component scores."
    )
    parser.add_argument("--component-scores", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--score-column", default="selectivity")
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        help="Top scored neurons per domain to include before aggregating by layer.",
    )
    parser.add_argument(
        "--normalize",
        choices=["domain_sum", "global_max"],
        default="domain_sum",
        help="domain_sum shows each domain's distribution across layers.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def layer_from_component(component: str) -> int:
    match = LAYER_PATTERN.search(component)
    if match is None:
        raise ValueError(f"Cannot parse layer from component: {component}")
    return int(match.group(1))


def selected_rows_by_domain(
    rows: list[dict[str, str]],
    score_column: str,
    top_k: int,
) -> dict[str, list[dict[str, str]]]:
    by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if score_column not in row:
            raise ValueError(f"Missing score column '{score_column}'")
        score = float(row[score_column])
        if score <= 0.0:
            continue
        by_domain[row["domain"]].append(row)

    selected: dict[str, list[dict[str, str]]] = {}
    for domain, domain_rows in by_domain.items():
        selected[domain] = sorted(
            domain_rows,
            key=lambda row: float(row[score_column]),
            reverse=True,
        )[:top_k]
    return selected


def aggregate_layers(
    rows_by_domain: dict[str, list[dict[str, str]]],
    score_column: str,
) -> list[dict[str, object]]:
    domains = sorted(rows_by_domain)
    layers = sorted(
        {
            layer_from_component(row["component"])
            for rows in rows_by_domain.values()
            for row in rows
        }
    )
    score_by_domain_layer: dict[tuple[str, int], float] = defaultdict(float)
    count_by_domain_layer: dict[tuple[str, int], int] = defaultdict(int)

    for domain, rows in rows_by_domain.items():
        for row in rows:
            layer = layer_from_component(row["component"])
            score_by_domain_layer[(domain, layer)] += float(row[score_column])
            count_by_domain_layer[(domain, layer)] += 1

    summary_rows: list[dict[str, object]] = []
    for domain in domains:
        domain_total = sum(score_by_domain_layer[(domain, layer)] for layer in layers)
        for layer in layers:
            score_sum = score_by_domain_layer[(domain, layer)]
            summary_rows.append(
                {
                    "domain": domain,
                    "layer": layer,
                    "selected_neuron_count": count_by_domain_layer[(domain, layer)],
                    "score_sum": score_sum,
                    "domain_score_share": score_sum / domain_total if domain_total else 0.0,
                }
            )
    return summary_rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def color_for_value(value: float) -> str:
    value = max(0.0, min(1.0, value))
    low = (247, 251, 255)
    high = (8, 81, 156)
    rgb = tuple(round(low[index] + (high[index] - low[index]) * value) for index in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def scaled_values(
    rows: list[dict[str, object]],
    normalize: str,
) -> dict[tuple[str, int], float]:
    raw = {
        (str(row["domain"]), int(row["layer"])): float(row["domain_score_share"])
        for row in rows
    }
    if normalize == "domain_sum":
        row_max_by_domain: dict[str, float] = defaultdict(float)
        for (domain, _layer), value in raw.items():
            row_max_by_domain[domain] = max(row_max_by_domain[domain], value)
        return {
            key: value / row_max_by_domain[key[0]] if row_max_by_domain[key[0]] else 0.0
            for key, value in raw.items()
        }

    global_max = max(raw.values()) if raw else 0.0
    return {key: value / global_max if global_max else 0.0 for key, value in raw.items()}


def write_svg(
    rows: list[dict[str, object]],
    path: Path,
    *,
    title: str,
    normalize: str,
) -> None:
    domains = sorted({str(row["domain"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    value_by_domain_layer = {
        (str(row["domain"]), int(row["layer"])): float(row["domain_score_share"])
        for row in rows
    }
    scaled_by_domain_layer = scaled_values(rows, normalize)

    cell = 26
    left = 120
    top = 70
    right = 30
    bottom = 45
    width = left + len(layers) * cell + right
    height = top + len(domains) * cell + bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="16" '
        f'font-weight="700">{title}</text>',
        f'<text x="{left}" y="48" font-family="Arial, sans-serif" font-size="11" '
        f'fill="#555">Cell value is share of selected neuron selectivity mass in that layer.</text>',
    ]

    for layer_index, layer in enumerate(layers):
        x = left + layer_index * cell + cell / 2
        parts.append(
            f'<text x="{x}" y="{top - 10}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="10" fill="#333">{layer}</text>'
        )

    for domain_index, domain in enumerate(domains):
        y = top + domain_index * cell
        parts.append(
            f'<text x="{left - 10}" y="{y + 17}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#222">{domain}</text>'
        )
        for layer_index, layer in enumerate(layers):
            x = left + layer_index * cell
            scaled_value = scaled_by_domain_layer[(domain, layer)]
            raw_value = value_by_domain_layer[(domain, layer)]
            color = color_for_value(scaled_value)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" '
                f'fill="{color}"><title>{domain}, layer {layer}: {raw_value:.3f}</title></rect>'
            )

    parts.append(
        f'<text x="{left}" y="{height - 15}" font-family="Arial, sans-serif" '
        f'font-size="10" fill="#555">Darker cells mark layers with more of that domain&apos;s '
        f'top-neuron selectivity mass.</text>'
    )
    parts.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def print_top_layers(rows: list[dict[str, object]], top_n: int = 3) -> None:
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)

    for domain, domain_rows in sorted(by_domain.items()):
        top_rows = sorted(
            domain_rows,
            key=lambda row: float(row["domain_score_share"]),
            reverse=True,
        )[:top_n]
        summary = ", ".join(
            f"L{row['layer']}={float(row['domain_score_share']):.3f}"
            for row in top_rows
        )
        print(f"{domain}: {summary}")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.component_scores)
    selected = selected_rows_by_domain(rows, args.score_column, args.top_k)
    summary_rows = aggregate_layers(selected, args.score_column)

    stem = args.component_scores.parent / f"mlp_neuron_layer_heatmap_top{args.top_k}"
    output_csv = args.output_csv or stem.with_suffix(".csv")
    output_svg = args.output_svg or stem.with_suffix(".svg")
    write_csv(summary_rows, output_csv)
    write_svg(
        summary_rows,
        output_svg,
        title=f"MLP neuron layer heatmap, top {args.top_k} per domain",
        normalize=args.normalize,
    )

    print(f"Wrote layer summary to {output_csv}")
    print(f"Wrote SVG heatmap to {output_svg}")
    print_top_layers(summary_rows)


if __name__ == "__main__":
    main()
