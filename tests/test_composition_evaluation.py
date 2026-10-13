"""Greedy generation, actual predicted intermediates, paired statistics and restoration."""
from dataclasses import replace
import csv
import json
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.icml2027.run_composition_screen import evaluate_episodes, make_calibration, main, preflight, run_screen
from llm_pruning.composition_analysis import paired_composition_summary
from llm_pruning.linked_tasks import decision, staged_record
from llm_pruning.models import LoadedModel
from llm_pruning.short_answers import encode_short_prompts, generate_short_answers, parse_integer
from test_composition_data import dataset_config


@pytest.mark.parametrize("text,expected", [("  +0007. \n", 7), ("−7", None), ("-7", -7), ("１２", 12),
    ("7 and 8", None), ("answer: 7", None), ("7.0", None), ("", None), ("7 units", None), ("1,234", None)])
def test_strict_answer_parser(text, expected):
    assert parse_integer(text) == expected


def test_staged_uses_wrong_valid_prediction_and_skips_malformed(tmp_path, tiny_loaded):
    config = dataset_config(tmp_path)
    episodes, _ = preflight(config)
    ids = sorted(episodes)
    gold = {r.prompt: r.target for e in episodes.values() for r in e.values()}
    wrong = staged_record(episodes[ids[0]], 999)
    prompts_seen = []
    def generate(loaded, prompts, **kwargs):
        prompts_seen.extend(prompts)
        return ["999" if p == episodes[ids[0]]["a"].prompt else "explanation 12"
                if p == episodes[ids[1]]["a"].prompt else "12345" if p == wrong.prompt else gold[p] for p in prompts]
    with patch("experiments.icml2027.run_composition_screen.generate_short_answers", side_effect=generate):
        rows = evaluate_episodes(tiny_loaded, episodes, config)
    stage = {r["episode_id"]: r for r in rows if r["variant"] == "staged"}
    assert wrong.prompt in prompts_seen
    assert stage[ids[0]]["prediction"] == 12345 and not stage[ids[0]]["correct"]
    assert stage[ids[1]]["status"] == "invalid_intermediate"
    assert stage[ids[1]]["prompt"] == "" and not stage[ids[1]]["valid"]
    assert len(rows) == 10


def test_generation_matches_single_greedy_decode_and_context_guard(tiny_loaded):
    tokenizer = tiny_loaded.tokenizer
    side = tokenizer.padding_side
    prompts = ["A?", "Longer question about a count?"]
    expected = []
    for p in prompts:
        encoded = encode_short_prompts(tiny_loaded, [p], "plain_short_v1", reserve_tokens=2)
        with torch.inference_mode():
            result = tiny_loaded.model.generate(**encoded, do_sample=False, max_new_tokens=2,
                num_beams=1, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        expected.append(tokenizer.decode(result[0, encoded['input_ids'].shape[1]:], skip_special_tokens=True))
    assert generate_short_answers(tiny_loaded, prompts, policy="plain_short_v1", batch_size=2, max_new_tokens=2) == expected
    assert tokenizer.padding_side == side
    with pytest.raises(ValueError, match="context"):
        generate_short_answers(tiny_loaded, ["x" * 1100], policy="plain_short_v1", max_new_tokens=2)
    with pytest.raises(ValueError, match="chat template"):
        generate_short_answers(tiny_loaded, prompts, policy="tokenizer_chat_v1", max_new_tokens=2)
    assert tokenizer.padding_side == side


def test_native_short_prompt_uses_template_without_duplicate_special_tokens(tiny_loaded):
    tiny_loaded.tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}{{ eos_token }}")
    encoded = encode_short_prompts(tiny_loaded, ['Count?'], 'tokenizer_chat_v1', reserve_tokens=2)
    assert int((encoded['input_ids'] == tiny_loaded.tokenizer.eos_token_id).sum()) == 1
    outputs = generate_short_answers(tiny_loaded, ['Count?'], policy='tokenizer_chat_v1', max_new_tokens=2)
    assert len(outputs) == 1 and isinstance(outputs[0], str)


def test_unpinned_research_rejected_without_model_or_results(tmp_path):
    config = dataset_config(tmp_path, fixture=False)
    with patch('experiments.icml2027.run_composition_screen.load_model') as loader:
        with pytest.raises(ValueError, match='immutable model revision'):
            run_screen(config, tmp_path / 'results')
        loader.assert_not_called()
    assert not (tmp_path / 'results').exists()


def test_calibration_exact_tokens_uses_discovery_no_oracle_or_targets(tmp_path, tiny_loaded):
    config = dataset_config(tmp_path)
    _, discovery = preflight(config)
    for mode in ("composed", "mixed"):
        batches, ledger = make_calibration(tiny_loaded, discovery, config, mode, 7)
        assert sum(int(b['attention_mask'].sum()) for b in batches) == 64
        assert sum(r['used_tokens'] for r in ledger) == 64
        assert all(r['split'] == 'discovery' and r['variant'] != 'oracle' for r in ledger)
        assert ledger[-1]['selection'] == 'explicit_prefix_for_token_budget'
        changed = {k: {v: replace(r, target="999999999") for v, r in e.items()} for k, e in discovery.items()}
        after, after_ledger = make_calibration(tiny_loaded, changed, config, mode, 7)
        assert ledger == after_ledger
        assert all(torch.equal(a['input_ids'], b['input_ids']) for a, b in zip(batches, after))
    with pytest.raises(ValueError, match="Insufficient"):
        make_calibration(tiny_loaded, discovery, replace(config, calibration_token_budget=100000), "mixed", 7)


def test_paired_summary_full_cohort_not_product_null_and_group_resampling(tmp_path):
    config = dataset_config(tmp_path)
    episodes, _ = preflight(config)
    base = [decision(r, r.target) for e in episodes.values() for r in [*e.values(), staged_record(e, int(e['a'].target))]]
    current = [{**r, 'correct': r['variant'] != 'composed'} for r in base]
    result = paired_composition_summary(base, current, seed=5, bootstrap_samples=30)
    assert result['paired']['composed_minus_staged_damage'] == {'estimate': 1., 'ci_low': 1., 'ci_high': 1.}
    assert result['atomic_ok_composed_wrong_rate'] == 1.
    assert result == paired_composition_summary(base, current, seed=5, bootstrap_samples=30)
    # Two episodes from one source do not constitute two independent bootstrap units.
    one_base = [{**r, 'source_group_id': 'one'} for r in base]
    one_now = [{**r, 'source_group_id': 'one'} for r in current]
    one = paired_composition_summary(one_base, one_now)
    assert one['source_group_count'] == 1 and one['paired']['composed_minus_staged_damage']['ci_low'] is None
    with pytest.raises(ValueError, match="same episodes"):
        paired_composition_summary(base, [r for r in current if r['episode_id'] == current[0]['episode_id']])


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_complete_random_model_screen_artifacts_and_restoration(tmp_path, tiny_loaded, family):
    config = dataset_config(tmp_path)
    if family == 'qwen2':
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=tiny_loaded.model.config.vocab_size,
            hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=1024, eos_token_id=tiny_loaded.tokenizer.eos_token_id)).eval()
        tiny_loaded = LoadedModel(model, tiny_loaded.tokenizer, torch.device('cpu'))
    original = {name: p.clone() for name, p in tiny_loaded.model.named_parameters()}
    with patch('experiments.icml2027.run_composition_screen.load_model', return_value=tiny_loaded) as loader:
        output = run_screen(config, tmp_path / 'results')
        with pytest.raises(FileExistsError):
            run_screen(config, tmp_path / 'results')
        assert loader.call_count == 1
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['condition_count'] == 4
    metrics = json.loads((output / 'metrics.json').read_text())
    assert len({m['cost']['resident_tensor_storage_bytes'] for m in metrics}) == 1
    assert len({m['cost']['prunable_nonzero_count'] for m in metrics}) == 2
    assert all(m['calibration_tokens'] == 64 for m in metrics)
    assert all(m['cpu_restoration_backup_bytes'] > 0 for m in metrics if m['retained_fraction'] < 1)
    assert all(m['paired']['composed_accuracy_damage']['estimate'] == 0 for m in metrics if m['retained_fraction'] == 1)
    with (output / 'decisions.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 * 2 * 5
    assert all(torch.equal(p, original[name]) for name, p in tiny_loaded.model.named_parameters())
    assert all(not m._forward_pre_hooks for m in tiny_loaded.model.modules())
    assert len((output / 'summary.md').read_text().splitlines()) <= 40


def test_generation_failure_restores_pruning_and_records_failure(tmp_path, tiny_loaded):
    config = replace(dataset_config(tmp_path), calibration_modes=("composed",))
    original = {name: p.clone() for name, p in tiny_loaded.model.named_parameters()}
    actual = evaluate_episodes
    calls = 0
    def fail_pruned(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected generation error")
        return actual(*args)
    with patch('experiments.icml2027.run_composition_screen.load_model', return_value=tiny_loaded), \
         patch('experiments.icml2027.run_composition_screen.evaluate_episodes', side_effect=fail_pruned):
        with pytest.raises(RuntimeError, match="injected"):
            run_screen(config, tmp_path / 'results')
    assert all(torch.equal(p, original[name]) for name, p in tiny_loaded.model.named_parameters())
    assert json.loads((tmp_path / 'results/screen/manifest.json').read_text())['status'] == 'failed'


def test_preflight_and_freeze_no_model_or_results(tmp_path, monkeypatch, capsys):
    config = dataset_config(tmp_path)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config.to_dict()), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['screen', '--config', str(path), '--validate-only'])
    with patch('experiments.icml2027.run_composition_screen.load_model') as loader:
        main()
        loader.assert_not_called()
    assert 'no model loaded' in capsys.readouterr().out
    protocol = tmp_path / 'frozen.json'
    monkeypatch.setattr('sys.argv', ['screen', '--config', str(path), '--freeze-protocol', str(protocol)])
    with patch('experiments.icml2027.run_composition_screen.load_model') as loader:
        main()
        loader.assert_not_called()
    assert json.loads(protocol.read_text()) == config.protocol()
    with pytest.raises(FileExistsError):
        main()
    assert not (tmp_path / 'results').exists()
