"""Linked short-answer baseline/Wanda screen; validate-only never loads a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    sys.path.insert(0, str(path))

import torch

from experiments.icml2027.composition_config import CompositionConfig
from experiments.icml2027.composition_controls import control_summary, reference_rows
from experiments.icml2027.controlled_composition import BENCHMARK_ID, validate_controlled_benchmark
from llm_pruning.composition_analysis import paired_composition_summary
from llm_pruning.linked_tasks import STATIC_VARIANTS, decision, staged_record, validate_linked_records
from llm_pruning.models import load_model
from llm_pruning.prompt_records import load_prompt_records
from llm_pruning.run_artifacts import RunArtifacts, file_fingerprint
from llm_pruning.short_answers import encode_short_prompts, generate_short_answers, short_answer_protocol
from llm_pruning.wanda import WandaPruning, model_cost


def preflight(config):
    episodes = validate_linked_records(load_prompt_records(config.dataset_path))
    benchmark_ids = {e['a'].metadata.get('benchmark_id') for e in episodes.values()}
    if BENCHMARK_ID in benchmark_ids:
        if benchmark_ids != {BENCHMARK_ID}:
            raise ValueError("Do not mix controlled benchmark and legacy records")
        validate_controlled_benchmark([r for e in episodes.values() for r in e.values()])
        if len({e['a'].metadata['task_family'] for e in episodes.values()}) != 1:
            raise ValueError("Evaluate controlled task families separately")
    config.validate_confirmation()
    selected = {key: e for key, e in episodes.items()
                if e["a"].split == config.split and e["a"].metadata["knowledge_scope"] == config.knowledge_scope}
    discovery = {key: e for key, e in episodes.items()
                 if e["a"].split == "discovery" and e["a"].metadata["knowledge_scope"] == config.knowledge_scope}
    if not selected or not discovery:
        raise ValueError("Evaluation and discovery episodes required in the configured knowledge_scope")
    if not config.fixture_run and any(e["a"].metadata["dataset_role"] == "engineering_fixture"
                                     for e in [*selected.values(), *discovery.values()]):
        raise ValueError("Engineering fixture data requires fixture_run=true")
    if config.max_eval_episodes is not None:
        ids = sorted(selected)
        random.Random(config.seed).shuffle(ids)
        selected = {key: selected[key] for key in sorted(ids[:config.max_eval_episodes])}
        all_by_group = {}
        for key, episode in episodes.items():
            if episode['a'].split == config.split and episode['a'].metadata['knowledge_scope'] == config.knowledge_scope:
                all_by_group.setdefault(episode['a'].metadata['source_group_id'], set()).add(key)
        for episode in selected.values():
            if not all_by_group[episode['a'].metadata['source_group_id']] <= set(selected):
                raise ValueError("Episode cap cuts a source/probe group; use max_eval_source_groups")
    if config.max_eval_source_groups is not None:
        groups = sorted({e['a'].metadata['source_group_id'] for e in selected.values()})
        random.Random(config.seed).shuffle(groups)
        keep = set(groups[:config.max_eval_source_groups])
        selected = {key: e for key, e in selected.items() if e['a'].metadata['source_group_id'] in keep}
    return selected, discovery


def make_calibration(loaded, discovery, config, mode, seed):
    """Exact non-padding token budget; final prompt prefix is explicitly logged.

    No oracle variants or output targets enter calibration. B's provided numeric
    intermediate is part of the task input in mixed mode, not a generated label.
    """
    variants = ("composed",) if mode == "composed" else ("a", "b", "composed")
    pool = [e[v] for _, e in sorted(discovery.items())
            if e['a'].metadata.get('case_role', 'reference') == 'reference' for v in variants]
    random.Random(seed).shuffle(pool)
    batches, ledger = [], []
    remaining = config.calibration_token_budget
    for record in pool:
        encoded = encode_short_prompts(loaded, [record.prompt], config.prompt_policy)
        length = encoded["input_ids"].shape[1]
        used = min(length, remaining)
        batches.append({key: value[:, :used].cpu() for key, value in encoded.items()})
        ledger.append({"record_id": record.id, "episode_id": record.metadata["episode_id"],
                       "source_group_id": record.metadata["source_group_id"], "split": record.split,
                       "variant": record.metadata["variant"], "prompt_tokens": length, "used_tokens": used,
                       "selection": "full_prompt" if used == length else "explicit_prefix_for_token_budget",
                       "input_ids_sha256": hashlib.sha256(encoded["input_ids"][:, :used].numpy().tobytes()).hexdigest()})
        remaining -= used
        if remaining == 0:
            return batches, ledger
    raise ValueError(f"Insufficient unique discovery tokens for {mode}: missing {remaining}; lower budget or add data")


def evaluate_episodes(loaded, episodes, config):
    records = [episode[v] for _, episode in sorted(episodes.items()) for v in STATIC_VARIANTS]
    kwargs = {"policy": config.prompt_policy, "batch_size": config.batch_size, "max_new_tokens": config.max_new_tokens}
    outputs = generate_short_answers(loaded, [r.prompt for r in records], **kwargs)
    rows = [decision(record, text) for record, text in zip(records, outputs, strict=True)]
    a_rows = {row["episode_id"]: row for row in rows if row["variant"] == "a"}
    staged, invalid = [], []
    for episode_id, episode in sorted(episodes.items()):
        a = a_rows[episode_id]
        record = staged_record(episode, a["prediction"] if a["valid"] else 0)
        if a["valid"]:
            staged.append(record)
        else:
            # Placeholder prompt is never submitted; do not feed malformed text to B.
            row = decision(record, "", malformed_stage=True)
            row["prompt"] = ""
            invalid.append(row)
    staged_outputs = generate_short_answers(loaded, [r.prompt for r in staged], **kwargs) if staged else []
    rows.extend(decision(record, text) for record, text in zip(staged, staged_outputs, strict=True))
    rows.extend(invalid)
    for row in rows:
        metadata = episodes[row['episode_id']]['a'].metadata
        row.update(case_role=metadata.get('case_role', 'reference'),
                   task_family=metadata.get('task_family', metadata.get('template_id')),
                   template_id=metadata.get('template_id'))
    return rows


def _sync(loaded):
    if loaded.device.type == "cuda":
        torch.cuda.synchronize(loaded.device)


def run_screen(config, results_root=REPO_ROOT / "results"):
    output_dir = results_root / config.experiment_name
    if output_dir.exists():
        raise FileExistsError(f"Results already exist: {output_dir}; use a new experiment_name")
    episodes, discovery = preflight(config)
    if not config.fixture_run and config.revision is None:
        raise ValueError("Pin an immutable model revision before a pretrained research run")
    with RunArtifacts(output_dir, REPO_ROOT, config.to_dict(), run_kind="composition_screen",
                      dataset=file_fingerprint(config.dataset_path), pruner="wanda_sequential_v1",
                      model={"identifier": config.model_name, "requested_revision": config.revision},
                      fixture_run=config.fixture_run, knowledge_scope=config.knowledge_scope) as run:
        run.write_json("evaluation_records.json", [
            {"episode_id": key, "source_group_id": e["a"].metadata["source_group_id"],
             "split": e["a"].split, "record_ids": {v: e[v].id for v in STATIC_VARIANTS}}
            for key, e in sorted(episodes.items())])
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        loaded = load_model(config.model_name, device=config.device, dtype=config.dtype,
                            revision=config.revision, local_files_only=config.local_files_only)
        run.manifest["model"].update(resolved_revision=getattr(loaded.model.config, "_commit_hash", None),
                                     device=str(loaded.device), dtype=str(loaded.model.dtype))
        run.manifest["prompt_protocol"] = short_answer_protocol(loaded.tokenizer, config.prompt_policy)
        run.manifest["frozen_protocol"] = file_fingerprint(config.test_protocol_path) if config.test_protocol_path else None
        run.manifest["cost_limits"] = "Dense weights; CPU restoration backups; no sparse speed/VRAM claim."
        run.write_json("manifest.json", run.manifest)
        # Verify token availability before the first evaluation or pruning.
        calibration = {(mode, seed): make_calibration(loaded, discovery, config, mode, seed)
                       for mode in config.calibration_modes for seed in config.calibration_seeds}
        dense_cost = model_cost(loaded.model)
        _sync(loaded)
        started = time.perf_counter()
        baseline = evaluate_episodes(loaded, episodes, config)
        _sync(loaded)
        baseline_seconds = time.perf_counter() - started
        summaries = []
        with (output_dir / "decisions.csv").open("x", encoding="utf-8", newline="") as handle:
            fields = ["condition", "calibration_mode", "calibration_seed", "retained_fraction", *baseline[0].keys()]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for mode in config.calibration_modes:
                for seed in config.calibration_seeds:
                    batches, ledger = calibration[(mode, seed)]
                    run.write_json(f"calibration_{mode}_{seed}.json", ledger)
                    for budget in sorted(config.retained_fractions, reverse=True):
                        condition = f"{mode}_seed{seed}_retained{budget:.17g}"
                        pruning_seconds = 0.
                        backup_bytes = 0
                        peak = None
                        if budget == 1:
                            current, cost, seconds = baseline, dense_cost, baseline_seconds
                            report = {"method": "none", "requested_sparsity": 0}
                        else:
                            if loaded.device.type == "cuda":
                                torch.cuda.reset_peak_memory_stats(loaded.device)
                            with WandaPruning(loaded.model) as pruning:
                                _sync(loaded)
                                started = time.perf_counter()
                                report = pruning.apply(batches, 1 - budget)
                                _sync(loaded)
                                pruning_seconds = time.perf_counter() - started
                                cost = model_cost(loaded.model, pruning.groups)
                                backup_bytes = sum(t.numel() * t.element_size() for _, t in pruning.backups)
                                started = time.perf_counter()
                                current = evaluate_episodes(loaded, episodes, config)
                                _sync(loaded)
                                seconds = time.perf_counter() - started
                                if loaded.device.type == "cuda":
                                    peak = torch.cuda.max_memory_allocated(loaded.device)
                        summary = paired_composition_summary(reference_rows(baseline), reference_rows(current), seed=config.seed,
                            bootstrap_samples=config.bootstrap_samples, confidence=config.confidence)
                        summary["cohort"] = "reference_cases_only"
                        summary["behavioral_controls"] = control_summary(baseline, current, seed=config.seed,
                            bootstrap_samples=config.bootstrap_samples, confidence=config.confidence)
                        summary.update(condition=condition, calibration_mode=mode, calibration_seed=seed,
                            retained_fraction=budget, calibration_tokens=config.calibration_token_budget,
                            cost=cost, pruning_seconds=pruning_seconds, evaluation_seconds=seconds,
                            peak_cuda_allocated_bytes=peak, cpu_restoration_backup_bytes=backup_bytes,
                            baseline_reused=budget == 1)
                        summary["baseline_feasible_point_estimate"] = all(
                            summary["baseline_accuracy"][v] >= config.min_baseline_accuracy for v in ("a", "b", "composed"))
                        summary["exploratory_pattern"] = (
                            summary["baseline_feasible_point_estimate"]
                            and max(summary["paired"][f"{v}_accuracy_damage"]["estimate"] for v in ("a", "b")) <= config.max_atomic_loss
                            and summary["paired"]["composed_minus_staged_damage"]["estimate"] >= config.minimum_extra_loss)
                        summaries.append(summary)
                        run.write_json(f"pruning_{condition}.json", report)
                        writer.writerows({"condition": condition, "calibration_mode": mode,
                            "calibration_seed": seed, "retained_fraction": budget, **row} for row in current)
        run.write_json("metrics.json", summaries)
        run.write_json("protocol_candidate.json", config.protocol())
        run.complete(["# Composition pruning screen", "", "Status: complete.",
                      f"Split: {config.split}; episodes: {len(episodes)}; knowledge: {config.knowledge_scope}",
                      f"Conditions: {len(summaries)}; baseline computed once; fixture_run={config.fixture_run}",
                      "Read metrics.json for A/B/composed/oracle/staged accuracy, malformed rates and paired intervals.",
                      "Primary metrics use reference cases; paired behavioral probes are reported separately.",
                      "Source-group bootstrap is exploratory/pointwise; no power, mechanism or novelty guarantee.",
                      "Feasibility/pattern flags use point estimates only; not a statistical GO decision.",
                      "Dense storage/nonzeros are separate; pruning does not imply sparse VRAM/latency savings.",
                      "Context-supplied data do not measure factual storage; staged calls are separate from direct composition.",
                      "No test-driven selection. Pin/freeze the validation protocol before a confirmation run."],
                     evaluation_episode_count=len(episodes), condition_count=len(summaries))
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate dense competence/probes once; no pruning.")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--validate-only", action="store_true")
    actions.add_argument("--freeze-protocol", type=Path, help="Write chosen validation settings exclusively; does not run models.")
    args = parser.parse_args()
    config = CompositionConfig.load(args.config)
    if args.baseline_only:
        config = replace(config, experiment_name=config.experiment_name + '_dense', retained_fractions=(1.,),
                         calibration_modes=(config.calibration_modes[0],), calibration_seeds=(config.calibration_seeds[0],))
    if args.validate_only or args.freeze_protocol:
        episodes, discovery = preflight(config)
        if args.freeze_protocol:
            if config.split != "validation" or (not config.fixture_run and config.revision is None):
                raise ValueError("Freeze from validation only, with a pinned research model revision")
            with args.freeze_protocol.open("x", encoding="utf-8") as handle:
                json.dump(config.protocol(), handle, indent=2, allow_nan=False)
                handle.write("\n")
        groups = {e['a'].metadata['source_group_id'] for e in episodes.values()}
        refs = sum(e['a'].metadata.get('case_role', 'reference') == 'reference' for e in episodes.values())
        print(f"Valid: {len(episodes)} {config.split} episodes ({refs} reference; {len(groups)} source groups); "
              f"{len(discovery)} discovery episodes; no model loaded.")
        if config.revision is None and not config.fixture_run:
            print("Research execution pending: pin revision and validate pretrained task/prompt feasibility on GPU.")
        print("Token-budget feasibility is checked with the tokenizer at execution; no generation or results created.")
        return
    output_dir = run_screen(config, args.results_root)
    print((output_dir / "summary.md").read_text(encoding="utf-8"), end="")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
