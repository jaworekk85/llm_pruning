"""Sample-size scenarios respect pairing ranges, multiplicity and assumptions."""
import math
import json

import pytest

from experiments.icml2027.plan_composition_precision import precision_plan
from experiments.icml2027.plan_composition_precision import main


def test_zero_worsened_bound_and_contrast_range():
    plan = precision_plan(groups=(32, 128))
    a, b = plan['scenarios']
    # With no observed losses, 32 independent trials cannot exclude 5% loss.
    assert a['zero_worsened_probability_upper'] > .05
    assert b['zero_worsened_probability_upper'] < .05
    upper = a['zero_worsened_probability_upper']
    assert math.isclose((1 - upper) ** 32, .05 / 3)
    assert a['extra_loss_half_width_hoeffding'] == 2 * a['atomic_damage_half_width_hoeffding']
    assert math.isclose(a['extra_loss_half_width_normal_sd_1'],
                        2 * b['extra_loss_half_width_normal_sd_1'])


def test_more_comparisons_and_smaller_effect_gap_need_more_groups():
    default = precision_plan()
    multiple = precision_plan(comparisons=18)
    smaller = precision_plan(expected_extra_loss=.15)
    for key in default['groups_for_power_normal_approximation']:
        assert multiple['groups_for_power_normal_approximation'][key] > default['groups_for_power_normal_approximation'][key]
        assert smaller['groups_for_power_normal_approximation'][key] > default['groups_for_power_normal_approximation'][key]
    assert multiple['scenarios'][0]['zero_worsened_probability_upper'] > default['scenarios'][0]['zero_worsened_probability_upper']
    half = precision_plan(half_width=.025)
    assert half['groups_for_half_width_hoeffding']['extra_loss'] >= 4 * default['groups_for_half_width_hoeffding']['extra_loss'] - 3


@pytest.mark.parametrize('kwargs', [dict(groups=()), dict(groups=(True,)), dict(groups=(1,)),
    dict(confidence=float('nan')), dict(power=1), dict(comparisons=0), dict(comparisons=True),
    dict(half_width=0), dict(expected_extra_loss=.10), dict(expected_extra_loss=float('inf'))])
def test_invalid_design_rejected(kwargs):
    with pytest.raises(ValueError):
        precision_plan(**kwargs)


def test_planner_cli_writes_compact_exclusive_manifest(tmp_path, monkeypatch, capsys):
    output = tmp_path / 'plan'
    monkeypatch.setattr('sys.argv', ['plan', '--output-dir', str(output)])
    main()
    assert len(capsys.readouterr().out.splitlines()) <= 40
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'
    assert json.loads((output / 'plan.json').read_text())['kind'] == 'design_scenarios_no_observed_model_results'
    with pytest.raises(FileExistsError):
        main()
