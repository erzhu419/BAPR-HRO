import pytest

from experiments.reconstruction_v28.summarize import METHODS, summarize


def row(root_id, values, cap=100, *, no_menu=False):
    outcomes = {}
    for method, loss in zip(METHODS, values, strict=True):
        outcomes[method] = dict(action=None if no_menu or loss is None else method,
            loss=loss, arrived=loss is not None and not no_menu,
            status='no_menu' if no_menu else 'unresolved_arrival' if loss is None else 'arrived')
    return dict(root_id=root_id, day='day', root_kind='origin', start=1000, deadline=1000+cap,
                menu_size=0 if no_menu else 5, outcomes=outcomes)


def test_known_means_common_subset_and_pairwise_populations_remain_distinct():
    rows = [row('a', (50, 60, 55, 40, 30)), row('b', (20, None, None, 10, None)),
            row('c', (100, 100, 100, 100, 100), no_menu=True),
            row('d', (None, 100, None, None, 70), cap=200)]
    result = summarize(rows)
    assert result['methods']['single_target_nominal']['mean_known_loss'] == 50
    assert result['methods']['one_step_nominal']['mean_known_loss'] == pytest.approx(260/3)
    assert result['all_method_common']['roots'] == 2
    pair = result['comparisons']['single_target_nominal_vs_one_step_nominal']
    assert pair['complete_pairs'] == 2 and pair['unknown_pairs'] == 2
    assert pair['mean_gain'] == 10 and pair['mean_loss'] == {
        'one_step_nominal': 80, 'single_target_nominal': 70}
    assert (pair['wins'], pair['harms'], pair['ties'], pair['changed']) == (1, 0, 1, 1)
    assert pair['arrivals'] == {'one_step_nominal': 1, 'single_target_nominal': 1}
    assert pair['win_root_ids'] == ['a'] and pair['harm_root_ids'] == []
    assert result['methods']['full_first_offer']['statuses']['no_menu'] == 1


def test_missing_loss_bounds_use_each_roots_horizon_and_known_counterpart():
    rows = [row('known', (0, 80, 0, 30, 0), cap=100),
            row('candidate_missing', (0, 60, 0, None, 0), cap=100),
            row('baseline_missing', (0, None, 0, 40, 0), cap=200),
            row('both_missing', (0, None, 0, None, 0), cap=50)]
    pair = summarize(rows)['comparisons']['single_target_nominal_vs_one_step_nominal']
    # Gains:50, [-40,60], [-40,160], [-50,50], averaged over all four roots.
    bounds = pair['missingness_bounds']
    assert bounds['mean_gain_lower'] == -20
    assert bounds['mean_gain_upper'] == 80
    assert pair['one_unknown'] == 2 and pair['both_unknown'] == 1
    assert 'not confidence intervals' in bounds['interpretation']


def test_no_complete_pairs_are_not_reported_as_zero_gain_and_empty_input_stays_empty():
    result = summarize([row('missing', (None, None, None, None, None), cap=2400)])
    pair = result['comparisons']['combined_wait_nominal_vs_single_target_nominal']
    assert pair['mean_gain'] is None and pair['complete_pairs'] == 0
    assert pair['missingness_bounds']['mean_gain_lower'] == -2400
    assert pair['missingness_bounds']['mean_gain_upper'] == 2400
    empty = summarize([])
    assert empty['roots'] == 0 and empty['all_method_common']['roots'] == 0
    assert all(stats['mean_known_loss'] is None for stats in empty['methods'].values())
