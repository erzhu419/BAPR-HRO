from copy import deepcopy

import pytest

from experiments.reconstruction_v47.audit import (
    certified, curve, exact_legacy_arms, matching_values,
    overlapping_legacy_bounds, regret_upper, values,
)


def point(a, c, count=0):
    def arm(bounds):
        lo, hi = bounds
        return dict(lower=lo, upper=hi, mean=lo if lo == hi else None,
                    unresolved_probability=0. if lo == hi else (hi-lo)/90)
    gap = [a[0]-c[1], a[1]-c[0]]
    return dict(actions=dict(alight=arm(a), **{'continue': arm(c)}),
        difference_interval=gap, certified_action=certified(gap),
        counters=dict(charged_compiles=count),
        cost=dict(process_cpu=count/1000, wall=count/900))


def test_curve_certification_and_root_regret_bound():
    rows = [point((10, 100), (10, 100)), point((20, 35), (40, 70), 100),
            point((30, 30), (50, 50), 150)]
    result = curve(rows, clock=10, deadline=100)
    assert result['certified_action'] == 'alight'
    assert regret_upper(rows[1], 'alight') == 0
    assert regret_upper(rows[1], 'continue') == 50
    assert matching_values(rows[-1], deepcopy(rows[-1]))


def test_curve_rejects_lost_refinement_and_budget_overrun():
    rows = [point((20, 50), (40, 70), 100), point((15, 40), (40, 70), 200)]
    with pytest.raises(ValueError, match='widened'):
        curve(rows, clock=10, deadline=100)
    with pytest.raises(ValueError, match='quota exceeded'):
        curve([point((30, 30), (50, 50), 2001)], clock=10, deadline=100)


def test_false_population_certificate_and_exactness_are_rejected():
    row = point((10, 80), (30, 70))
    row['certified_action'] = 'continue'
    with pytest.raises(ValueError, match='certification'):
        values(row, clock=10, deadline=100)
    row['certified_action'] = None
    row['actions']['alight']['mean'] = 40
    with pytest.raises(ValueError, match='advertised as exact'):
        values(row, clock=10, deadline=100)


def test_legacy_bounds_may_cross_but_cannot_disagree_on_truth():
    old = point((20, 70), (40, 40))
    new = point((30, 80), (35, 50))
    assert overlapping_legacy_bounds(new, old)
    assert exact_legacy_arms(new, old) == ['continue']
    bad = point((71, 90), (35, 50))
    with pytest.raises(ValueError, match='Disjoint'):
        overlapping_legacy_bounds(bad, old)
    bad = point((30, 80), (41, 50))
    with pytest.raises(ValueError, match='excludes established exact'):
        exact_legacy_arms(bad, old)
