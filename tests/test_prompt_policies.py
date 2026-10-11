"""Offline template ownership, instruction alignment and likelihood references."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tokenizers import processors

from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.choice_records import ChoiceRecord
from llm_pruning.mc_evaluation import score_candidate_batch
from llm_pruning.mc_prompts import candidate_text, format_choice_prompt, prompt_policy_metadata
from llm_pruning.targets import TargetSpec
from llm_pruning.token_regions import tokenize_prompt_answers


def choice_record():
    return ChoiceRecord("fixture", "science", "offline", "fixture", "fixture", "validation", "Which answer?", ["Short", "A longer answer"], 0)


def chat_template():
    return "{{ eos_token }}{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"


def test_legacy_is_explicit_and_new_instructions_match_answer_mode(tiny_loaded):
    record = choice_record()
    assert format_choice_prompt(record) == format_choice_prompt(record, policy="legacy_mc_v1", scoring_mode="choice_text")
    for policy in ["plain_mc_v1", "tokenizer_chat_v1"]:
        tiny_loaded.tokenizer.chat_template = chat_template()
        letter = format_choice_prompt(record, policy=policy, tokenizer=tiny_loaded.tokenizer, scoring_mode="letter")
        text = format_choice_prompt(record, policy=policy, tokenizer=tiny_loaded.tokenizer, scoring_mode="choice_text")
        assert "only the answer letter" in letter
        assert "complete text of the correct option" in text
        assert "only the answer letter" not in text
        assert letter.endswith("Answer:") and text.endswith("Answer:")
    with pytest.raises(ValueError, match="Unsupported prompt policy"):
        format_choice_prompt(record, policy="automatic_guess")
    config = BaselineConfig("fixture", "random-init", Path("data.jsonl"), TargetSpec("domain", "science"))
    assert config.prompt_policy == "legacy_mc_v1"
    with pytest.raises(ValueError, match="Unsupported prompt_policy"):
        replace(config, prompt_policy="automatic_guess")


def test_chat_owns_special_tokens_and_template_fingerprint(tiny_loaded):
    tokenizer = tiny_loaded.tokenizer
    tokenizer.chat_template = chat_template()
    tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<eos> $A", special_tokens=[("<eos>", tokenizer.eos_token_id)],
    )
    protocol = prompt_policy_metadata(tokenizer, "tokenizer_chat_v1", "letter")
    prompt = format_choice_prompt(choice_record(), policy="tokenizer_chat_v1", tokenizer=tokenizer)
    inputs, masks = tokenize_prompt_answers(tokenizer, [prompt], [" A"], "cpu", add_special_tokens=protocol["add_special_tokens"])
    assert (inputs["input_ids"] == tokenizer.eos_token_id).sum().item() == 1
    assert masks["answer"].sum().item() == 2
    assert protocol["add_special_tokens"] is False
    assert len(protocol["template_sha256"]) == 64
    tokenizer.chat_template += " "
    assert prompt_policy_metadata(tokenizer, "tokenizer_chat_v1", "letter")["template_sha256"] != protocol["template_sha256"]


def test_chat_without_template_fails_instead_of_guessing(tiny_loaded):
    tiny_loaded.tokenizer.chat_template = None
    with pytest.raises(ValueError, match="requires an explicit"):
        prompt_policy_metadata(tiny_loaded.tokenizer, "tokenizer_chat_v1", "letter")
    with pytest.raises(ValueError, match="requires an explicit"):
        format_choice_prompt(choice_record(), policy="tokenizer_chat_v1", tokenizer=tiny_loaded.tokenizer)


@pytest.mark.parametrize("policy", ["legacy_mc_v1", "plain_mc_v1", "tokenizer_chat_v1"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_candidate_likelihoods_match_manual_shifted_token_reference(tiny_loaded, policy, padding_side):
    tokenizer = tiny_loaded.tokenizer
    tokenizer.chat_template = chat_template()
    tokenizer.padding_side = padding_side
    record = choice_record()
    scores = score_candidate_batch(tiny_loaded, [(record, 0), (record, 1)], "baseline", "choice_text", prompt_policy=policy)
    prompt = format_choice_prompt(record, policy=policy, tokenizer=tokenizer, scoring_mode="choice_text")
    for index, score in enumerate(scores):
        inputs, masks = tokenize_prompt_answers(tokenizer, [prompt], [candidate_text(record, index, "choice_text")], "cpu", add_special_tokens=policy != "tokenizer_chat_v1")
        with torch.no_grad():
            logits = tiny_loaded.model(**inputs, use_cache=False).logits
        answer_positions = masks["answer"][0].nonzero(as_tuple=True)[0]
        target_ids = inputs["input_ids"][0, answer_positions]
        log_probabilities = logits[0, answer_positions - 1].float().log_softmax(dim=-1)
        expected_sum = log_probabilities.gather(1, target_ids[:, None]).sum().item()
        assert score.token_count == len(answer_positions)
        assert score.sum_logprob == pytest.approx(expected_sum, abs=1e-5)
        assert score.mean_logprob == pytest.approx(expected_sum / len(answer_positions), abs=1e-6)
