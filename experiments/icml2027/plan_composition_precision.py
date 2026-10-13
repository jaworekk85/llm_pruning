"""Offline sample-size scenarios for independent one-episode source groups; no models."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import NormalDist

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_pruning.run_artifacts import RunArtifacts


def precision_plan(*, groups=(32, 128, 512, 1024, 2048), confidence=.95,
                   comparisons=3, half_width=.05, expected_extra_loss=.20,
                   meaningful_extra_loss=.10, power=.80):
    """Known-range bounds and assumed-SD normal approximations, not empirical power.

    Damage D = dense correctness - pruned correctness is in [-1, 1].
    Extra composition loss Z = D_composed - D_staged is in [-2, 2].
    Bonferroni covers the specified, frozen family of comparisons. The default
    family is extra loss plus A and B damage for ONE model/mask condition.
    Bounds assume independent groups with one episode each and a fixed protocol.
    """
    if not groups or any(type(n) is not int or n < 2 for n in groups):
        raise ValueError("Each independent-group count must be an integer >= 2")
    if type(comparisons) is not int or comparisons < 1:
        raise ValueError("Positive integer comparisons required")
    for name, value in (("confidence", confidence), ("power", power), ("half_width", half_width)):
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be finite and in (0, 1)")
    if (not math.isfinite(expected_extra_loss) or not math.isfinite(meaningful_extra_loss)
            or not 0 <= meaningful_extra_loss < expected_extra_loss <= 2):
        raise ValueError("Need 0 <= meaningful_extra_loss < expected_extra_loss <= 2")
    alpha = 1 - confidence
    log_factor = math.log(2 * comparisons / alpha)
    z_two = NormalDist().inv_cdf(1 - alpha / (2 * comparisons))
    z_one = NormalDist().inv_cdf(1 - alpha / comparisons)
    gap = expected_extra_loss - meaningful_extra_loss
    scenarios = []
    for n in groups:
        scenarios.append({"independent_groups": n,
            "extra_loss_half_width_hoeffding": 4 * math.sqrt(log_factor / (2 * n)),
            "atomic_damage_half_width_hoeffding": 2 * math.sqrt(log_factor / (2 * n)),
            "zero_worsened_probability_upper": -math.expm1(math.log(alpha / comparisons) / n),
            "extra_loss_half_width_normal_sd_0_5": z_two * .5 / math.sqrt(n),
            "extra_loss_half_width_normal_sd_1": z_two / math.sqrt(n)})
    return {"kind": "design_scenarios_no_observed_model_results",
        "confidence": confidence, "comparisons": comparisons, "target_half_width": half_width,
        "expected_extra_loss": expected_extra_loss, "meaningful_extra_loss": meaningful_extra_loss,
        "target_power_normal_approximation": power, "scenarios": scenarios,
        "groups_for_half_width_hoeffding": {
            "extra_loss": math.ceil(16 * log_factor / (2 * half_width ** 2)),
            "atomic_damage": math.ceil(4 * log_factor / (2 * half_width ** 2))},
        "groups_for_power_normal_approximation": {
            f"assumed_sd_{sd:g}": math.ceil((sd * (z_one + NormalDist().inv_cdf(power)) / gap) ** 2)
            for sd in (.5, 1.)},
        "assumptions": ["One episode per independent source group; fixed model/mask/protocol.",
            "Normal SDs are scenarios, not estimates; they do not guarantee power.",
            "Zero-worsened upper bound applies only if NO dense-correct item becomes wrong.",
            "For repeated or unequal source groups, use a separately specified cluster design.",
            "Calibration seeds/masks are repeated conditions, not extra independent cases.",
            "Bounds address case sampling, not transfer to other models/tasks/calibration seeds."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, nargs="+", default=[32, 128, 512, 1024, 2048])
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--comparisons", type=int, default=3)
    parser.add_argument("--half-width", type=float, default=.05)
    parser.add_argument("--expected-extra-loss", type=float, default=.20)
    parser.add_argument("--meaningful-extra-loss", type=float, default=.10)
    parser.add_argument("--power", type=float, default=.80)
    args = parser.parse_args()
    config = {key: value for key, value in vars(args).items() if key != "output_dir"}
    plan = precision_plan(**config)
    summary = ["# Composition precision scenarios", "", "Status: complete; no model inference.",
        f"Confidence: {args.confidence}; frozen comparisons: {args.comparisons}.",
        "One episode per independent source group; fixed model/mask condition.",
        "n | normal extra-loss half-width (SD=.5 / 1) | zero-worsened upper bound"]
    for row in plan["scenarios"]:
        summary.append(f"{row['independent_groups']} | {row['extra_loss_half_width_normal_sd_0_5']:.3f} / "
                       f"{row['extra_loss_half_width_normal_sd_1']:.3f} | "
                       f"{row['zero_worsened_probability_upper']:.3f}")
    summary.extend(["Normal approximations assume SD; no empirical power guarantee.",
        "See plan.json for conservative known-range bounds and assumptions.",
        "Choose sample size on validation; freeze before opening test outputs."])
    with RunArtifacts(args.output_dir, REPO_ROOT, {**config, "seed": 0},
                      run_kind="composition_precision_planning") as run:
        run.write_json("plan.json", plan)
        run.complete(summary)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
