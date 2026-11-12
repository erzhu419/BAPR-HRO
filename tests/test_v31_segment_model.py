from collections import Counter, defaultdict
import copy

import numpy as np
import pytest

from experiments.reconstruction_v30.forecast import forecast_board
from experiments.reconstruction_v31.model import SegmentDurationBelief, fit_segment_durations
from src.empirical_duration_belief import EmpiricalDurationBelief, fit_empirical_durations
from src.transfer_opportunity import RouteOption, TransferOpportunityScores
from src.transit_graph import Connection, Stop, TransitGraph


def measured(cid, trip, dep_stop, arr_stop, elapsed, *, dep_sequence=0, dep=600):
    endpoint = lambda time: dict(status='REAL', time_seconds=time, official_journey=trip)
    return dict(id=cid, route_id='R', trip_id=trip, dep_sequence=dep_sequence, arr_sequence=dep_sequence+1,
        dep_stop=dep_stop, arr_stop=arr_stop, scheduled_departure_seconds=dep,
        scheduled_arrival_seconds=dep+20, departure=endpoint(dep), arrival=endpoint(dep+elapsed),
        dropoff_allowed=True, pickup_allowed=True, outcome='MEASURED')


def data():
    rows = [measured(i, f'a{i}', 'A', 'B', 20) for i in range(20)]
    rows += [measured(i+20, f'b{i}', 'C', 'D', 40) for i in range(20)]
    return dict(train_dates=['2023-07-25'], dates={
        '2023-07-25': dict(connections=rows),
        '2023-07-16': dict(connections=[measured(99, 'eval', 'A', 'B', 9999)])})


def graph():
    result = TransitGraph()
    for sid in range(4):
        result.add_stop(Stop(sid, str(sid), min_transfer_time=0))
    result.add_connection(Connection(0, 'R', 'a', 0, 1, 600, 620))
    result.add_connection(Connection(1, 'R', 'b', 2, 3, 600, 620))
    return result


def candidate(segment_fit=None):
    source = data()
    fit = fit_empirical_durations(source)
    return SegmentDurationBelief(graph(), duration_fit=fit,
        segment_fit=fit_segment_durations(source) if segment_fit is None else segment_fit,
        stop_identities=dict(enumerate('ABCD')), grid_step_seconds=20, public_occurrences=False)


def test_training_only_fit_ignores_evaluation_values_and_rejects_evaluation_training_dates():
    source = data()
    fitted = fit_segment_durations(source)
    changed = copy.deepcopy(source)
    changed['dates']['2023-07-16']['connections'][0]['arrival']['time_seconds'] = 1000000
    assert fit_segment_durations(changed) == fitted
    changed['train_dates'].append('2023-07-16')
    with pytest.raises(ValueError, match='2023-07-25 only'):
        fit_segment_durations(changed)


def test_segment_atom_partition_exactly_reproduces_parent_fit_eligibility_including_dwell_and_gap():
    source = data()
    rows = source['dates']['2023-07-25']['connections']
    rows += [measured(50, 'chain', 'A', 'B', 20),
             measured(51, 'chain', 'B', 'C', 20, dep_sequence=1, dep=620),
             measured(52, 'chain', 'D', 'A', 20, dep_sequence=2, dep=660)]
    rows[0]['arrival']['official_journey'] = 'wrong'
    rows[1]['arrival']['time_seconds'] = 599
    fitted, parent = fit_segment_durations(source), fit_empirical_durations(source)
    counts = defaultdict(Counter)
    for group in fitted['groups']:
        counts[group['route'], group['kind'], group['scheduled']].update(
            dict(zip(group['values'], group['counts'])))
    expected = {(route, kind, int(scheduled)): Counter(dict(zip(group['values'], group['counts'])))
        for route, kinds in parent['route_groups'].items()
        for kind, schedules in kinds.items() for scheduled, group in schedules.items()}
    assert dict(counts) == expected
    assert fitted['excluded'] == parent['excluded']
    assert any(g['kind'] == 'dwell' and (g['from_stop'], g['to_stop']) == ('B', 'B') for g in fitted['groups'])
    assert any(g['kind'] == 'gap' and (g['from_stop'], g['to_stop']) == ('C', 'D') for g in fitted['groups'])


def test_fixed_shrinkage_has_expected_mean_and_unseen_segment_uses_exact_parent_atoms():
    belief = candidate()
    values, weights = belief.segment_atoms('travel', 20, route='R', segment=('A', 'B'))
    assert values.tolist() == [20, 40]
    np.testing.assert_allclose(weights, [.75, .25])
    assert values @ weights == pytest.approx(25.)  # .5*20 + .5*parent_mean30.
    values, weights = belief.segment_atoms('travel', 20, route='R', segment=('X', 'Y'))
    parent_values, parent_weights = belief.duration_atoms('travel', 20, route='R')
    np.testing.assert_array_equal(values, parent_values)
    np.testing.assert_array_equal(weights, parent_weights)


def test_unseen_segments_reproduce_parent_chain_and_current_board_value():
    source = data()
    fit = fit_segment_durations(source)
    fit['groups'] = []
    belief = candidate(fit)
    parent = EmpiricalDurationBelief(belief.graph, duration_fit=belief.duration_fit,
        grid_step_seconds=20, public_occurrences=False)
    params = {'R': dict(mean=0., std=.5, cancel_rate=.2)}
    a, b = belief._law('a', params), parent._law('a', params)
    np.testing.assert_array_equal(a.initial, b.initial)
    np.testing.assert_array_equal(a._transition(0), b._transition(0))
    options = {0: RouteOption((0,), {})}
    new = TransferOpportunityScores(belief.graph, belief, params, 800, options).board_score(0, 610)
    old = TransferOpportunityScores(parent.graph, parent, params, 800, options).board_score(0, 610)
    assert new == old


def test_same_route_and_schedule_do_not_share_different_segment_kernels():
    belief = candidate()
    params = {'R': dict(mean=0., std=.5, cancel_rate=0.)}
    first, second = belief._law('a', params), belief._law('b', params)
    # Evaluate the first shared-cache candidate before requesting the second.
    a = np.exp(first._transition(0))
    b = np.exp(second._transition(0))
    first_index = int(np.flatnonzero(first.times[0] == 600)[0])
    second_index = int(np.flatnonzero(second.times[0] == 600)[0])
    assert a[first_index] @ first.times[1] == pytest.approx(625.)
    assert b[second_index] @ second.times[1] == pytest.approx(635.)
    assert len(belief._segment_transition_cache) == 2


def test_segment_chain_preserves_service_mass_and_full_distribution_score():
    belief = candidate()
    params = {'R': dict(mean=0., std=.5, cancel_rate=.2)}
    model = belief._law('a', params)
    transition = np.exp(model._transition(0))
    np.testing.assert_allclose(transition.sum(axis=1), 1.)
    assert not transition[model.canceled[0][:, None] != model.canceled[1][None, :]].any()
    compiler = TransferOpportunityScores(belief.graph, belief, params, 800,
        {0: RouteOption((0,), {})})
    forecast = forecast_board(compiler, 0, 600)
    final = forecast['distributions']['final_arrival']
    assert dict(zip(final['values'], final['probabilities'])) == pytest.approx({620: .75, 640: .25})
    assert forecast['cost'] == compiler.board_score(0, 600).exact == pytest.approx(625.)
