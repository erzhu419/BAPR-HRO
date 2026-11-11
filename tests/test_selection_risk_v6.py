import math

from experiments.reconstruction_v6.calibration_selection import (score_bounds, state_score,
    select, thresholds, evaluate_menu, MENUS, METHODS)


def row(incoming, outgoing, actual=1., status='REAL', probability=.8):
    return {'incoming': incoming, 'outgoing': outgoing, 'actual': actual, 'status': status,
            'incoming_delay': 0, 'scheduled': 0, 'caught': actual is not None and 0 <= actual <= 2,
            'methods': {method: {'lower': 0., 'upper': 2., 'mean': 1.} for method in METHODS},
            'probabilities': {method: probability for method in METHODS}}


def test_service_category_and_unknown_clock_keep_different_meanings():
    assert score_bounds(row(0, 0, None, 'CANCELED'), 'student') == (-math.inf, -math.inf)
    assert score_bounds(row(0, 0, None, 'PROGNOSE'), 'student') == (-math.inf, math.inf)
    menu = [row(0, 0), row(0, 1, None, 'PROGNOSE', .5)]
    result = evaluate_menu(menu, 'student', 'menu_max', 0)
    assert result['n_incoming_states'] == 1
    assert result['coverage_lower'] == 0 and result['coverage_upper'] == 1
    assert result['unresolved_targets'] == 1
    selected = evaluate_menu(menu, 'student', 'fixed_selector', 0)
    assert selected['coverage_lower'] == selected['coverage_upper'] == 1


def test_menu_max_protects_every_selector_but_single_selector_does_not():
    menu = [row(0, 0, 1, probability=.9), row(0, 1, 5, probability=.2)]
    assert state_score(menu, 'student', 'menu_max') == (3., 3.)
    assert state_score(menu, 'student', 'fixed_selector') == (-1., -1.)
    changed_truth = [dict(value, actual=1000) for value in menu]
    assert select(menu, 'student')['outgoing'] == select(changed_truth, 'student')['outgoing']


def test_missingness_is_not_dropped_and_can_make_calibration_vacuous():
    def calibration(n_unknown):
        values = [row(i, 0, None, 'PROGNOSE') if i < n_unknown else row(i, 0) for i in range(10)]
        return {str(day): {menu: {'rows': values} for menu in MENUS} for day in range(9)}
    finite = thresholds(calibration(1), list(map(str, range(9))))
    assert finite['fixed_schedule']['student']['menu_max']['factor_upper'] == 0
    vacuous = thresholds(calibration(2), list(map(str, range(9))))
    assert vacuous['fixed_schedule']['student']['menu_max']['vacuous']
    detail = vacuous['fixed_schedule']['student']['menu_max']['date_scores']['0']['groups']['ordinary']
    assert detail['n_incoming_states'] == 10
