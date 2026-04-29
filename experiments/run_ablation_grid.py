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
    component_count: int
    device: str | None
    command: list[str]
    log_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a grid of ablation jobs, optionally one process per GPU."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prompt-jsonl", type=Path, default=Path("data/prompt_sets/qa_v1.jsonl"))
    parser.add_argument("--split", choices=["discovery", "validation", "test"], default="validation")
    parser.add_argument(
        "--component-scores",
        type=Path,
        default=None,
        help=(
            "Optional score CSV override. Use only when running one granularity; "
            "otherwise built-in QA v1 score paths are used."
        ),
    )
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
    parser.add_argument(
        "--component-counts",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of k values for ablation curves. Overrides --component-count.",
    )
    parser.add_argument("--random-control-repeats", type=int, default=10)
    parser.add_argument(
        "--control-strategy",
        choices=["random", "layer_matched"],
        default="random",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("results/qa_v1_ablation_grid"))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional directory for per-job stdout/stderr logs.",
    )
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


def score_file_for(args: argparse.Namespace, granularity: str) -> Path:
    if args.component_scores is not None:
        if len(args.granularities) != 1:
            raise ValueError("--component-scores can only be used with one granularity.")
        return args.component_scores

    try:
        return DEFAULT_SCORE_FILES[granularity]
    except KeyError as exc:
        raise ValueError(f"Unsupported granularity: {granularity}") from exc


def make_job(
    args: argparse.Namespace,
    granularity: str,
    target_domain: str,
    component_count: int,
    index: int,
) -> AblationJob:
    device = args.devices[index % len(args.devices)] if args.devices else None
    output_dir = args.output_root / f"{args.split}_{target_domain}_{granularity}_k{component_count}"
    log_path = None
    if args.log_dir is not None:
        log_path = args.log_dir / f"{args.split}_{target_domain}_{granularity}_k{component_count}.log"

    command = [
        args.python,
        str(REPO_ROOT / "experiments" / "ablate_components.py"),
        "--prompt-jsonl",
        str(args.prompt_jsonl),
        "--split",
        args.split,
        "--component-scores",
        str(score_file_for(args, granularity)),
        "--granularity",
        granularity,
        "--target-domain",
        target_domain,
        "--component-count",
        str(component_count),
        "--random-control-repeats",
        str(args.random_control_repeats),
        "--control-strategy",
        args.control_strategy,
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
        component_count=component_count,
        device=device,
        command=command,
        log_path=log_path,
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
            label = f"{job.target_domain}/{job.granularity}/k{job.component_count}"
            device = f" on {job.device}" if job.device else ""
            log_target = f", log={job.log_path}" if job.log_path else ""
            print(f"\nStarting {label}{device}{log_target}", flush=True)
            if job.log_path is None:
                running.append((job, subprocess.Popen(job.command, cwd=REPO_ROOT)))
            else:
                job.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = job.log_path.open("wb")
                running.append(
                    (
                        job,
                        subprocess.Popen(
                            job.command,
                            cwd=REPO_ROOT,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                        ),
                    )
                )

        next_running: list[tuple[AblationJob, subprocess.Popen[bytes]]] = []
        for job, process in running:
            return_code = process.poll()
            if return_code is None:
                next_running.append((job, process))
                continue

            label = f"{job.target_domain}/{job.granularity}/k{job.component_count}"
            if return_code == 0:
                print(f"Finished {label}", flush=True)
            else:
                print(f"FAILED {label} with exit code {return_code}", flush=True)
                failures.append(job)
        running = next_running

        if running and len(running) >= parallel_jobs:
            running[0][1].wait(timeout=None)

    if failures:
        failed = ", ".join(
            f"{job.target_domain}/{job.granularity}/k{job.component_count}"
            for job in failures
        )
        raise SystemExit(f"Ablation jobs failed: {failed}")


def main() -> None:
    args = parse_args()
    component_counts = args.component_counts or [args.component_count]
    jobs = [
        make_job(args, granularity, target_domain, component_count, index)
        for index, (target_domain, granularity, component_count) in enumerate(
            (target_domain, granularity, component_count)
            for target_domain in args.domains
            for granularity in args.granularities
            for component_count in component_counts
        )
    ]
    run_jobs(jobs, args.parallel_jobs, args.dry_run)


if __name__ == "__main__":
    main()
