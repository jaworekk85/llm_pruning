"""Select components from a completed discovery run, enforcing held-out provenance."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .choice_records import load_choice_records
from .components import ComponentId
from .run_artifacts import file_fingerprint


def load_discovery_selection(path: Path, config, evaluation_records, *, granularity="attention_head", token_region="answer", top_k=1):
    if config.split not in {"validation", "test"}:
        raise ValueError("Causal evaluation requires validation or test, never discovery")
    if granularity not in {"attention_head", "mlp_group"} or token_region not in {"prompt", "answer"}:
        raise ValueError("Unsupported discovery granularity or token region")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    path = path.resolve()
    files = {name: path / name for name in ("manifest.json", "component_inventory.json", "collection_records.json", "activations.csv")}
    manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
    inventory = json.loads(files["component_inventory.json"].read_text(encoding="utf-8"))
    records = json.loads(files["collection_records.json"].read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("run_kind") != "component_activations":
        raise ValueError("Require a completed component activation run")
    source_config = manifest["config"]
    expected_target = {"field": config.target.field, "value": config.target.value}
    if source_config.get("split") != "discovery" or source_config.get("target") != expected_target:
        raise ValueError("Discovery split and target must match the evaluation protocol")
    if (source_config.get("model_name"), source_config.get("revision")) != (config.model_name, config.revision):
        raise ValueError("Discovery model/revision differs from evaluation")
    if (source_config.get("scoring_mode"), source_config.get("prompt_policy", "legacy_mc_v1")) != (config.scoring_mode, config.prompt_policy):
        raise ValueError("Discovery prompt/answer policy differs from evaluation")
    protocol = manifest.get("prompt_protocol")
    if not protocol or protocol.get("policy") != config.prompt_policy:
        raise ValueError("Discovery requires recorded prompt_protocol; recollect with the current runner")
    if token_region not in source_config.get("token_regions", []):
        raise ValueError("Requested token region was not collected")
    if manifest["dataset"]["sha256"] != file_fingerprint(config.dataset_path)["sha256"]:
        raise ValueError("Discovery dataset fingerprint differs from evaluation")
    by_id = {record["id"]: record for record in records}
    if not by_id or len(by_id) != len(records) or any(record["split"] != "discovery" for record in records):
        raise ValueError("Discovery record IDs must be unique and exclusively discovery")
    if set(by_id) & {record.id for record in evaluation_records}:
        raise ValueError("Discovery and evaluation record IDs overlap")
    actual = {record.id: record for record in load_choice_records(config.dataset_path)}
    for record_id, record in by_id.items():
        if record_id not in actual or actual[record_id].split != "discovery" or getattr(actual[record_id], config.target.field) != record[config.target.field]:
            raise ValueError("Discovery metadata does not match the source dataset")
    component_map = {item["id"]: item for item in inventory["components"]}
    if len(component_map) != len(inventory["components"]):
        raise ValueError("Discovery inventory contains duplicate component IDs")
    components = [value for value in component_map if ComponentId.parse(value).kind == granularity]
    component_set = set(components)
    if top_k > len(components):
        raise ValueError("top_k exceeds the discovery component count")
    target_ids = {record_id for record_id, record in by_id.items() if record[config.target.field] == config.target.value}
    if not target_ids or target_ids == set(by_id):
        raise ValueError("Selectivity ranking needs target and non-target discovery examples")
    totals = defaultdict(lambda: [0.0, 0.0])
    seen = set()
    row_count = 0
    with files["activations.csv"].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            record_id, component = row["record_id"], row["component_id"]
            if record_id not in by_id or component not in component_map or row["split"] != "discovery":
                raise ValueError("Activation rows are not part of the discovery inventory/dataset")
            if row["target_field"] != config.target.field or row["target_value"] != by_id[record_id][config.target.field]:
                raise ValueError("Activation target metadata differs from discovery records")
            if row["token_region"] != token_region or component not in component_set:
                continue
            if (record_id, component) in seen:
                raise ValueError("Duplicate discovery activation row")
            seen.add((record_id, component))
            value = float(row["mean_abs"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Discovery mean_abs must be finite and nonnegative")
            totals[component][0 if record_id in target_ids else 1] += value
    if row_count != manifest.get("activation_row_count") or len(seen) != len(by_id) * len(components):
        raise ValueError("Incomplete discovery activation coverage or row count")
    ranking = []
    for component in components:
        target_mean = totals[component][0] / len(target_ids)
        other_mean = totals[component][1] / (len(by_id) - len(target_ids))
        ranking.append({"component_id": component, "score": (target_mean - other_mean) / (target_mean + other_mean + 1e-12),
                        "target_mean_abs": target_mean, "other_mean_abs": other_mean})
    ranking.sort(key=lambda row: (-row["score"], row["component_id"]))
    return {
        "selected": [row["component_id"] for row in ranking[:top_k]], "ranking": ranking,
        "method": "mean_abs_selectivity_v1", "granularity": granularity, "token_region": token_region,
        "aggregation": "Equal example weights; non-target pools all other target-field values.",
        "inventory": inventory, "source_model": manifest["model"], "prompt_protocol": protocol,
        "source_files": {name: file_fingerprint(file) for name, file in files.items()},
        "discovery_ids": sorted(by_id),
    }
