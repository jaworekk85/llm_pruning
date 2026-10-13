"""Build controlled-context cases or import numeric facts; split sources before variants."""
from __future__ import annotations

import random
from llm_pruning.linked_tasks import validate_linked_records
from llm_pruning.prompt_records import PromptRecord
from llm_pruning.short_answers import parse_integer

BUILDER_ID = "count_affine_v2"


def build_linked_data(facts: list[dict] | None = None, *, episodes_per_split=128, seed=0, fixture=False):
    """Generated contexts test use of supplied information, not parametric factual storage.

    Imported facts require id, source/fact groups, question, answer, knowledge_scope,
    source_name, license. context_supplied imports additionally need context.
    """
    if type(seed) is not int or seed < 0 or type(episodes_per_split) is not int or episodes_per_split < 1:
        raise ValueError("Nonnegative seed and positive episodes_per_split required")
    rng = random.Random(seed)
    imported = facts is not None
    if facts is None:
        facts = []
        for i in range(3 * episodes_per_split):
            entity, distractor = [f"Station-{j:05d}" for j in (2 * i, 2 * i + 1)]
            if rng.randrange(2):
                entity, distractor = distractor, entity
            answer = rng.randrange(11, 900)
            distractor_answer = rng.randrange(11, 900)
            while distractor_answer == answer:
                distractor_answer = rng.randrange(11, 900)
            sentences = [f"{entity} holds {answer} supply units.",
                         f"{distractor} holds {distractor_answer} supply units."]
            rng.shuffle(sentences)
            facts.append({"id": f"context.{i}", "source_group_id": f"station.{i}", "fact_group_id": f"fact.{i}",
                          "question": f"How many supply units does {entity} hold?", "answer": str(answer),
                          "context": " ".join(sentences),
                          "context_target_position": next(j for j, s in enumerate(sentences) if s.startswith(entity + " ")),
                          "knowledge_scope": "context_supplied", "source_name": "controlled_station_counts_v2",
                          "license": "CC0-1.0"})
    if len(facts) < 3:
        raise ValueError("At least three independent source groups are needed")
    seen_ids, fact_owners, inputs = set(), {}, set()
    groups = {}
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("Each imported fact must be a JSON object")
        for key in ("id", "source_group_id", "fact_group_id", "question", "answer", "knowledge_scope", "source_name", "license"):
            if not isinstance(fact.get(key), str) or not fact[key].strip():
                raise ValueError(f"Fact needs nonempty string {key}")
        if fact["id"] in seen_ids or parse_integer(fact["answer"]) is None:
            raise ValueError("Duplicate fact ID or malformed integer answer")
        seen_ids.add(fact["id"])
        if fact["knowledge_scope"] not in {"context_supplied", "parametric_recall"}:
            raise ValueError("Invalid fact knowledge_scope")
        if "context" in fact and not isinstance(fact["context"], str):
            raise ValueError("Fact context must be a string")
        if fact["knowledge_scope"] == "context_supplied" and not fact.get("context", "").strip():
            raise ValueError("Context-supplied facts require context")
        if fact["knowledge_scope"] == "parametric_recall" and fact.get("context"):
            raise ValueError("Parametric recall facts must not supply context")
        normalized = " ".join((fact.get("context", "") + fact["question"]).casefold().split())
        if normalized in inputs:
            raise ValueError("Duplicate fact input")
        inputs.add(normalized)
        owner = fact_owners.setdefault(fact["fact_group_id"], fact["source_group_id"])
        if owner != fact["source_group_id"]:
            raise ValueError("Put all occurrences of a fact in the same source group")
        groups.setdefault(fact["source_group_id"], []).append(fact)
    group_ids = sorted(groups)
    if len(group_ids) < 3:
        raise ValueError("At least three independent source groups are needed")
    rng.shuffle(group_ids)
    splits = {group: ("discovery", "validation", "test")[i % 3] for i, group in enumerate(group_ids)}
    records = []
    for group in sorted(groups):
        for fact in sorted(groups[group], key=lambda f: f["id"]):
            intermediate = str(parse_integer(fact["answer"]))
            multiplier, offset = rng.randrange(2, 6), rng.randrange(1, 20)
            target = str(int(intermediate) * multiplier + offset)
            if parse_integer(target) is None:
                raise ValueError("Final integer exceeds supported answer range")
            context = fact.get("context", "")
            a_prompt = (context + "\n" if context else "") + fact["question"]
            b_template = f"A count is {{intermediate}} units. What is {multiplier} times that count plus {offset}?"
            composed = ((context + "\n" if context else "")
                        + "Find the integer answer to this retrieval question without outputting it: "
                        + fact["question"]
                        + f"\nReturn only the final result of {multiplier} times that answer plus {offset}.")
            prompts = {"a": a_prompt, "b": b_template.format(intermediate=intermediate),
                       "composed": composed, "oracle": composed + f"\nThe correct intermediate count is {intermediate} units."}
            # B is identical to the staged second step given a correct A output.
            # A context identifier disambiguates equal arithmetic prompts across splits.
            b_template = f"Case {fact['id']}. " + b_template
            prompts["b"] = b_template.format(intermediate=intermediate)
            for variant, prompt in prompts.items():
                records.append(PromptRecord(
                    id=f"{fact['id']}.{variant}", domain="controlled_counts" if not imported else "numeric_facts",
                    prompt=prompt, split=splits[group], source_type="synthetic" if not imported else "imported",
                    source_name=fact["source_name"], prompt_type="short_answer",
                    difficulty="lookup" if variant == "a" else "affine" if variant == "b" else "lookup_affine",
                    language="en", target=intermediate if variant == "a" else target, license=fact["license"],
                    capability="retrieval" if variant == "a" else "arithmetic" if variant == "b" else "composition",
                    task=variant, metadata={"episode_id": fact["id"], "source_group_id": group,
                        "fact_group_id": fact["fact_group_id"], "variant": variant, "knowledge_scope": fact["knowledge_scope"],
                        "dataset_role": "engineering_fixture" if fixture else "imported_benchmark" if imported else "controlled_benchmark",
                        "gold_intermediate": intermediate, "staged_b_template": b_template,
                        "template_id": BUILDER_ID, "multiplier": multiplier, "offset": offset,
                        "context_target_position": fact.get("context_target_position"),
                        "builder_seed": seed, "research_limit": "No stored-knowledge claim for context-supplied facts."},
                ))
    validate_linked_records(records)
    return records
