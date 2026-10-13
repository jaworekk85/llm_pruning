"""Controlled composition benchmark: shared evidence, executable truth, paired probes.

These context tasks test use of information, not storage of parametric knowledge.
The existing lookup/affine builder remains an engineering exercise.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict

from llm_pruning.linked_tasks import validate_linked_records
from llm_pruning.prompt_records import PromptRecord

BENCHMARK_ID = "controlled_composition_v1"
FAMILIES = ("lookup_lookup", "lookup_add")
ROLES = ("reference", "changed_target", "changed_distractor", "reordered", "renamed")


def solve(case, family):
    """Independent symbolic semantics; the model is never used to create labels."""
    entities = dict(case["entities"])
    intermediate = entities[case["target_entity"]]
    final = dict(case["values"])[intermediate] if family == "lookup_lookup" else intermediate + case["addend"]
    return intermediate, final


def render_context(case, family, layout):
    if layout == "sentences":
        context = "\n".join(f"{name} has code {value}." if family == "lookup_lookup"
                            else f"{name} holds {value} units." for name, value in case["entities"])
        if family == "lookup_lookup":
            context += "\n" + "\n".join(f"Code {key} has value {value}." for key, value in case["values"])
    else:
        context = "Entity | " + ("Code" if family == "lookup_lookup" else "Units") + "\n"
        context += "\n".join(f"{name} | {value}" for name, value in case["entities"])
        if family == "lookup_lookup":
            context += "\nCode | Value\n" + "\n".join(f"{key} | {value}" for key, value in case["values"])
    return "Use only the following fictional records.\n" + context


def _records(case, family, layout, group, role, split, seed, fixture):
    intermediate, final = solve(case, family)
    context = render_context(case, family, layout)
    entity = case["target_entity"]
    if family == "lookup_lookup":
        questions = {"a": f"What code does {entity} have?",
                     "composed": f"What is the value of the code that {entity} has?"}
        b_template = context + "\nWhat is the value of code {intermediate}?"
    else:
        questions = {"a": f"How many units does {entity} hold?",
                     "composed": f"What is {case['addend']} plus the number of units held by {entity}?"}
        b_template = context + f"\nGiven an input of {{intermediate}} units, what is {case['addend']} plus that input?"
    composed = context + "\n" + questions["composed"]
    prompts = {"a": context + "\n" + questions["a"],
               "b": b_template.format(intermediate=intermediate), "composed": composed,
               "oracle": composed + f"\nThe correct intermediate is {intermediate}."}
    episode_id = f"{group}.{role}"
    for variant, prompt in prompts.items():
        yield PromptRecord(id=f"{episode_id}.{variant}", domain="controlled_composition",
            prompt=prompt, split=split, source_type="synthetic", source_name=BENCHMARK_ID,
            prompt_type="short_answer", difficulty=f"{len(case['entities'])}_entities",
            language="en", target=str(intermediate if variant == "a" else final), license="CC0-1.0",
            capability="context_lookup" if variant == "a" or (variant == "b" and family == "lookup_lookup")
                       else "addition" if variant == "b" else "composition", task=variant,
            metadata={"episode_id": episode_id, "source_group_id": group, "fact_group_id": group,
                "variant": variant, "knowledge_scope": "context_supplied",
                "dataset_role": "engineering_fixture" if fixture else "controlled_benchmark",
                "gold_intermediate": str(intermediate), "staged_b_template": b_template,
                "benchmark_id": BENCHMARK_ID, "task_family": family, "case_role": role,
                "template_id": layout, "shared_context": context, "symbolic_case": case,
                "builder_seed": seed, "carry": family == "lookup_add" and intermediate % 10 + case["addend"] >= 10,
                "research_limit": "Context use only; no parametric-storage or neural-mechanism claim."})


def build_controlled_benchmark(*, family="lookup_lookup", groups_per_split=128,
                               seed=2027, namespace="composition_v1", items=4, fixture=False):
    """Each split/group has its own RNG: expansion preserves old cases and splits.

    Five role episodes share a source group; all four static variants use the same
    evidence within an episode. Final and intermediate labels are two-digit integers.
    """
    if family not in FAMILIES or type(groups_per_split) is not int or groups_per_split < 1:
        raise ValueError("Known family and positive groups_per_split required")
    if type(seed) is not int or seed < 0 or type(items) is not int or not 4 <= items <= 8:
        raise ValueError("Nonnegative integer seed and 4..8 items required")
    if not isinstance(namespace, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", namespace):
        raise ValueError("Simple explicit namespace required")
    records = []
    for split in ("discovery", "validation", "test"):
        for i in range(groups_per_split):
            group = f"{namespace}.seed{seed}.{family}.items{items}.{split}.{i:06d}"
            digest = hashlib.sha256(group.encode()).hexdigest()
            rng = random.Random(int(digest, 16))
            names = [f"Item-{digest[:10]}-{j}" for j in range(items)]
            rng.shuffle(names)
            codes = rng.sample(range(10, 80), items)
            values = rng.sample(range(10, 100), items)
            while any(code == value for code, value in zip(codes, values)):
                values = rng.sample(range(10, 100), items)
            target_index = i % items
            case = {"entities": list(zip(names, codes)), "target_entity": names[target_index],
                    "values": list(zip(codes, values)), "addend": rng.randrange(2, 10)}
            rng.shuffle(case["values"])
            layout = ("sentences", "rows")[(i // items) % 2]
            for role in ROLES:
                probe = {**case, "entities": list(case["entities"]), "values": list(case["values"])}
                if role in {"changed_target", "changed_distractor"}:
                    alternatives = [j for j in range(items) if j != target_index]
                    j, k = (target_index, alternatives[0]) if role == "changed_target" else alternatives[:2]
                    left, right = probe["entities"][j], probe["entities"][k]
                    probe["entities"][j], probe["entities"][k] = (left[0], right[1]), (right[0], left[1])
                elif role == "reordered":
                    probe["entities"] = probe["entities"][1:] + probe["entities"][:1]
                    probe["values"] = probe["values"][1:] + probe["values"][:1]
                elif role == "renamed":
                    mapping = {name: f"Item-{digest[10:20]}-{j}" for j, (name, _) in enumerate(case["entities"])}
                    probe["entities"] = [(mapping[name], number) for name, number in probe["entities"]]
                    probe["target_entity"] = mapping[case["target_entity"]]
                records.extend(_records(probe, family, layout, group, role, split, seed, fixture))
    validate_controlled_benchmark(records)
    return records


def validate_controlled_benchmark(records):
    """Check executable gold, shared inputs and complete behavioral probe blocks."""
    episodes = validate_linked_records(records)
    groups = defaultdict(dict)
    for episode in episodes.values():
        a = episode["a"]
        m = a.metadata
        if m.get("benchmark_id") != BENCHMARK_ID or m.get("task_family") not in FAMILIES:
            raise ValueError("Invalid controlled benchmark identity/family")
        if m.get("template_id") not in {"sentences", "rows"}:
            raise ValueError("Unknown context layout")
        role = m.get("case_role")
        if role not in ROLES or role in groups[m["source_group_id"]]:
            raise ValueError("Invalid or duplicate probe role")
        case = m["symbolic_case"]
        names, numbers = zip(*case["entities"])
        keys, values = zip(*case["values"])
        if (not 4 <= len(names) <= 8 or len(set(names)) != len(names)
                or len(set(numbers)) != len(numbers) or len(set(keys)) != len(keys)
                or len(set(values)) != len(values) or set(numbers) != set(keys)
                or case["target_entity"] not in names
                or any(type(n) is not int or not 10 <= n < 100 for n in (*numbers, *keys, *values))
                or type(case["addend"]) is not int or not 2 <= case["addend"] <= 9):
            raise ValueError("Invalid symbolic case")
        expected_context = render_context(case, m["task_family"], m["template_id"])
        intermediate, final = solve(case, m["task_family"])
        canonical = {r.task: r for r in _records(case, m["task_family"], m["template_id"],
            m["source_group_id"], role, a.split, m["builder_seed"], m["dataset_role"] == "engineering_fixture")}
        for variant, record in episode.items():
            if any(record.metadata[key] != m[key] for key in
                   ("benchmark_id", "task_family", "case_role", "symbolic_case", "shared_context", "template_id")):
                raise ValueError("Probe metadata inconsistent within episode")
            if not record.prompt.startswith(expected_context + "\n") or m["shared_context"] != expected_context:
                raise ValueError("Variants must share exact evidence")
            if record.prompt != canonical[variant].prompt:
                raise ValueError("Prompt disagrees with symbolic question contract")
            if record.target != str(intermediate if variant == "a" else final):
                raise ValueError("Gold disagrees with independent symbolic execution")
        groups[m["source_group_id"]][role] = (m, intermediate, final)
    for probes in groups.values():
        if set(probes) != set(ROLES):
            raise ValueError("Each source group needs all five probe roles")
        ref, a, final = probes["reference"]
        for role, (m, next_a, next_final) in probes.items():
            if m["task_family"] != ref["task_family"] or m["template_id"] != ref["template_id"]:
                raise ValueError("Family/layout must stay fixed within a probe block")
            original, probe = ref['symbolic_case'], m['symbolic_case']
            original_entities, probe_entities = dict(original['entities']), dict(probe['entities'])
            if dict(original['values']) != dict(probe['values']) or original['addend'] != probe['addend']:
                raise ValueError("Probe must preserve the second-stage rule")
            if role in {'changed_target', 'changed_distractor'}:
                if (list(original_entities) != list(probe_entities)
                        or original['target_entity'] != probe['target_entity']
                        or sorted(original_entities.values()) != sorted(probe_entities.values())
                        or sum(original_entities[k] != v for k, v in probe_entities.items()) != 2):
                    raise ValueError("Information probes must swap exactly two entity values")
            elif role == 'reordered':
                if (original_entities != probe_entities or original['target_entity'] != probe['target_entity']
                        or original['entities'] == probe['entities']):
                    raise ValueError("Order probe must reorder unchanged facts")
            elif role == 'renamed':
                if (set(original_entities) & set(probe_entities)
                        or list(original_entities.values()) != list(probe_entities.values())
                        or list(original_entities).index(original['target_entity'])
                        != list(probe_entities).index(probe['target_entity'])):
                    raise ValueError("Name probe must bijectively rename unchanged facts and query")
            if role == "changed_target":
                if a == next_a or final == next_final:
                    raise ValueError("Target change must change both gold answers")
            elif (a, final) != (next_a, next_final):
                raise ValueError("Irrelevant/order/name changes must preserve gold answers")
    return episodes
