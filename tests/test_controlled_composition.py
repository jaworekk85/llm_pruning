"""Executable semantics, genuine dependency, complete paired blocks and runner integration."""
import json
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from experiments.icml2027.build_controlled_composition import build_dataset
from experiments.icml2027.composition_config import CompositionConfig
from experiments.icml2027.composition_controls import control_summary, reference_rows
from experiments.icml2027.controlled_composition import build_controlled_benchmark, solve, validate_controlled_benchmark
from experiments.icml2027.run_composition_screen import evaluate_episodes, make_calibration, preflight, run_screen, main


def config_for(tmp_path, family='lookup_lookup'):
    path = build_dataset(tmp_path / 'data', family=family, groups_per_split=2, fixture=True)
    assert b'\r\n' not in path.read_bytes()  # Cross-server fingerprints do not depend on OS.
    return CompositionConfig('controlled', path, 'offline-random', fixture_run=True,
        prompt_policy='plain_short_v1', max_new_tokens=2, max_eval_source_groups=1,
        retained_fractions=(1., .5), calibration_modes=('composed',), calibration_seeds=(7,),
        calibration_token_budget=64, bootstrap_samples=10)


@pytest.mark.parametrize('family', ['lookup_lookup', 'lookup_add'])
def test_dependency_invariance_shared_context_and_balanced_positions(family):
    episodes = validate_controlled_benchmark(build_controlled_benchmark(family=family, groups_per_split=8))
    blocks = {}
    for e in episodes.values():
        a = e['a']
        m = a.metadata
        intermediate, final = solve(m['symbolic_case'], family)
        assert a.target == str(intermediate) and e['composed'].target == str(final)
        assert all(r.prompt.startswith(m['shared_context'] + '\n') for r in e.values())
        assert all(10 <= int(r.target) < 100 for r in e.values())
        blocks.setdefault(m['source_group_id'], {})[m['case_role']] = e
    for block in blocks.values():
        ref = block['reference']['a'].metadata['symbolic_case']
        for role, e in block.items():
            case = e['a'].metadata['symbolic_case']
            if role == 'changed_target':
                assert dict(ref['entities'])[ref['target_entity']] != dict(case['entities'])[case['target_entity']]
                assert dict(ref['values']) == dict(case['values'])
                assert sum(dict(ref['entities'])[k] != v for k, v in case['entities']) == 2
                assert block['reference']['composed'].target != e['composed'].target
            elif role == 'changed_distractor':
                assert dict(ref['entities'])[ref['target_entity']] == dict(case['entities'])[case['target_entity']]
                assert sum(dict(ref['entities'])[k] != v for k, v in case['entities']) == 2
            elif role == 'reordered':
                assert dict(ref['entities']) == dict(case['entities']) and ref['entities'] != case['entities']
            elif role == 'renamed':
                assert not set(dict(ref['entities'])) & set(dict(case['entities']))
        assert len(block) == 5
    val = [e['a'] for e in episodes.values() if e['a'].split == 'validation' and e['a'].metadata['case_role'] == 'reference']
    assert {r.metadata['template_id'] for r in val} == {'rows', 'sentences'}
    assert {next(i for i, (name, _) in enumerate(r.metadata['symbolic_case']['entities'])
                if name == r.metadata['symbolic_case']['target_entity']) for r in val} == set(range(4))


def test_expanding_population_preserves_exact_existing_cases_and_splits():
    small = build_controlled_benchmark(groups_per_split=2, seed=11, namespace='reserved')
    large = {r.id: r for r in build_controlled_benchmark(groups_per_split=5, seed=11, namespace='reserved')}
    assert all(large[r.id] == r for r in small)
    new = build_controlled_benchmark(groups_per_split=2, seed=11, namespace='fresh')
    assert not {r.metadata['source_group_id'] for r in small} & {r.metadata['source_group_id'] for r in new}


@pytest.mark.parametrize('problem', ['missing_probe', 'wrong_question', 'wrong_context', 'wrong_label'])
def test_benchmark_audit_rejects_corrupted_semantics(problem):
    rows = build_controlled_benchmark(groups_per_split=1)
    if problem == 'missing_probe':
        rows = [r for r in rows if not (r.split == 'validation' and r.metadata['case_role'] == 'reordered')]
    else:
        i = next(i for i, r in enumerate(rows) if r.task == 'b')
        r = rows[i]
        if problem == 'wrong_question':
            # Keep B/template linked consistency while changing the task itself.
            old = r.metadata['staged_b_template']
            new = old.replace('What is the value of code', 'What is the code after')
            episode = r.metadata['episode_id']
            rows = [replace(x, prompt=new.format(intermediate=x.metadata['gold_intermediate']) if x.task == 'b' else x.prompt,
                            metadata={**x.metadata, 'staged_b_template': new})
                    if x.metadata['episode_id'] == episode else x for x in rows]
        elif problem == 'wrong_context':
            rows[i] = replace(r, prompt='No evidence.\n' + r.prompt)
        else:
            rows[i] = replace(r, target='99')
    with pytest.raises(ValueError):
        validate_controlled_benchmark(rows)


def test_group_cap_keeps_every_probe_and_calibration_uses_reference_only(tmp_path, tiny_loaded):
    config = config_for(tmp_path)
    episodes, discovery = preflight(config)
    assert len(episodes) == 5
    assert {e['a'].metadata['case_role'] for e in episodes.values()} == {
        'reference', 'changed_target', 'changed_distractor', 'reordered', 'renamed'}
    with pytest.raises(ValueError, match='cuts a source'):
        preflight(replace(config, max_eval_source_groups=None, max_eval_episodes=1))
    with pytest.raises(ValueError, match='not both'):
        replace(config, max_eval_episodes=1)
    _, ledger = make_calibration(tiny_loaded, discovery, config, 'mixed', 7)
    assert all(discovery[row['episode_id']]['a'].metadata['case_role'] == 'reference' for row in ledger)


def test_wrong_but_invariant_predictions_do_not_pass_controls(tmp_path, tiny_loaded):
    config = config_for(tmp_path)
    episodes, _ = preflight(config)
    gold = {r.prompt: r.target for e in episodes.values() for r in e.values()}
    with patch('experiments.icml2027.run_composition_screen.generate_short_answers',
               side_effect=lambda loaded, prompts, **kwargs: [gold[p] for p in prompts]):
        rows = evaluate_episodes(tiny_loaded, episodes, config)
    wrong = [{**row, 'correct': False, 'prediction': 999} for row in rows]
    report = control_summary(rows, wrong, seed=0, bootstrap_samples=10, confidence=.95)
    assert len(reference_rows(rows)) == 5
    assert report['paired_probe_groups'] == 1
    assert all(r['composed_pair_both_correct_rate'] == 0 for r in report['roles'].values())
    assert all(r['prediction_changed_rate_given_valid_pair'] == 0 for r in report['roles'].values())
    correct = control_summary(rows, rows, seed=0, bootstrap_samples=10, confidence=.95)
    assert correct['roles']['changed_target']['prediction_changed_rate_given_valid_pair'] == 1
    assert correct['roles']['changed_distractor']['prediction_changed_rate_given_valid_pair'] == 0


def test_random_model_screen_separates_reference_and_probes_and_restores(tmp_path, tiny_loaded):
    config = config_for(tmp_path)
    original = {name: p.clone() for name, p in tiny_loaded.model.named_parameters()}
    with patch('experiments.icml2027.run_composition_screen.load_model', return_value=tiny_loaded):
        output = run_screen(config, tmp_path / 'results')
    metrics = json.loads((output / 'metrics.json').read_text())
    assert all(m['episode_count'] == 1 and m['source_group_count'] == 1 for m in metrics)
    assert all(m['behavioral_controls']['paired_probe_groups'] == 1 for m in metrics)
    assert all(len(m['behavioral_controls']['roles']) == 4 for m in metrics)
    assert all(m['cohort'] == 'reference_cases_only' for m in metrics)
    assert all(torch.equal(p, original[name]) for name, p in tiny_loaded.model.named_parameters())
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'


def test_baseline_only_cli_never_constructs_pruner(tmp_path, tiny_loaded, monkeypatch, capsys):
    config = config_for(tmp_path)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config.to_dict()), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['screen', '--config', str(path), '--results-root', str(tmp_path / 'results'), '--baseline-only'])
    with patch('experiments.icml2027.run_composition_screen.load_model', return_value=tiny_loaded), \
         patch('experiments.icml2027.run_composition_screen.WandaPruning') as pruner:
        main()
        pruner.assert_not_called()
    output = tmp_path / 'results/controlled_dense'
    metrics = json.loads((output / 'metrics.json').read_text())
    assert len(metrics) == 1 and metrics[0]['retained_fraction'] == 1
    assert len(capsys.readouterr().out.splitlines()) <= 40
