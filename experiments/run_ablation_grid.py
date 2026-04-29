from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCORE_FILES = {
    "mlp_module": Path("results/qa_v1_analysis_mlp_modules/component_scores.csv"),
    "mlp_neuron": Path("results/qa_v1_analysis_mlp_neurons_top32/component_scores.csv"),
    "attention_head": Path("results/qa_v1_analysis_heads/component_scores.csv"),
}


@dataclass(frozen=True)
class AblationJob:
    granularity: str
    target_domain: str
    device: str | None
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a grid of ablation jobs, optionally one process per GPU."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prompt-jsonl", type=Path, default=Path("data/prompt_sets/qa_v1.jsonl"))
    parser.add_argument("--split", choices=["discovery", "validation", "test"], default="validation")
    parser.add_argument(
        "--domains",
        nargs="*",
        default=["agriculture", "astronomy", "math", "medicine", "politics"],
        help="Target domains to test.",
    )
    parser.add_argument(
        "--granularities",
        nargs="*",
        choices=["mlp_module", "mlp_neuron", "attention_head"],
        default=["mlp_module", "mlp_neuron", "attention_head"],
    )
    parser.add_argument("--component-count", type=int, default=5)
    parser.add_argument("--random-control-repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("results/qa_v1_ablation_grid"))
    parser.add_argument(
        "--devices",
        nargs="*",
        default=[],
        help="Optional device strings, for example cuda:0 cuda:1. Jobs are assigned round-robin.",
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=1,
        help="Number of ablation subprocesses to run at once. Use 1 per GPU.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def score_file_for(granularity: str) -> Path:
    try:
        return DEFAULT_SCORE_FILES[granularity]
    except KeyError as exc:
        raise ValueError(f"Unsupported granularity: {granularity}") from exc


def make_job(args: argparse.Namespace, granularity: str, target_domain: str, index: int) -> AblationJob:
    device = args.devices[index % len(args.devices)] if args.devices else None
    output_dir = args.output_root / f"{args.split}_{target_domain}_{granularity}_k{args.component_count}"

    command = [
        args.python,
        str(REPO_ROOT / "experiments" / "ablate_components.py"),
        "--prompt-jsonl",
        str(args.prompt_jsonl),
        "--split",
        args.split,
        "--component-scores",
        str(score_file_for(granularity)),
        "--granularity",
        granularity,
        "--target-domain",
        target_domain,
        "--component-count",
        str(args.component_count),
        "--random-control-repeats",
        str(args.random_control_repeats),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(output_dir),
    ]
    if device is not None:
        command.extend(["--device", device])

    return AblationJob(
        granularity=granularity,
        target_domain=target_domain,
        device=device,
        command=command,
    )


def run_jobs(jobs: list[AblationJob], parallel_jobs: int, dry_run: bool) -> None:
    if parallel_jobs < 1:
        raise ValueError("--parallel-jobs must be at least 1")

    for job in jobs:
        print(" ".join(job.command))
    if dry_run:
        return

    running: list[tuple[AblationJob, subprocess.Popen[bytes]]] = []
    remaining = list(jobs)
    failures: list[AblationJob] = []

    while remaining or running:
        while remaining and len(running) < parallel_jobs:
            job = remaining.pop(0)
            label = f"{job.target_domain}/{job.granularity}"
            device = f" on {job.device}" if job.device else ""
            print(f"\nStarting {label}{device}", flush=True)
            running.append((job, subprocess.Popen(job.command, cwd=REPO_ROOT)))

        next_running: list[tuple[AblationJob, subprocess.Popen[bytes]]] = []
        for job, process in running:
            return_code = process.poll()
            if return_code is None:
                next_running.append((job, process))
                continue

            label = f"{job.target_domain}/{job.granularity}"
            if return_code == 0:
                print(f"Finished {label}", flush=True)
            else:
                print(f"FAILED {label} with exit code {return_code}", flush=True)
                failures.append(job)
        running = next_running

        if running and len(running) >= parallel_jobs:
            running[0][1].wait(timeout=None)

    if failures:
        failed = ", ".join(f"{job.target_domain}/{job.granularity}" for job in failures)
        raise SystemExit(f"Ablation jobs failed: {failed}")


def main() -> None:
    args = parse_args()
    jobs = [
        make_job(args, granularity, target_domain, index)
        for index, (target_domain, granularity) in enumerate(
            (target_domain, granularity)
            for target_domain in args.domains
            for granularity in args.granularities
        )
    ]
    run_jobs(jobs, args.parallel_jobs, args.dry_run)


if __name__ == "__main__":
    main()
