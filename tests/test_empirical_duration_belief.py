import copy

import numpy as np
import pytest

from src.empirical_duration_belief import EmpiricalDurationBelief, empirical_clock_matrix, fit_empirical_durations
from src.evolving_trip_belief import EvolvingTripBelief
from src.transit_graph import Connection, Stop, TransitGraph


def measured(cid, trip, dep_seq, dep_stop, arr_stop, dep, arr, actual_dep, actual_arr):
    endpoint = lambda clock: dict(status='REAL', time_seconds=clock, official_journey=trip)
    return dict(id=cid, route_id='R', trip_id=trip, dep_sequence=dep_seq, arr_sequence=dep_seq+1,
                dep_stop=dep_stop, arr_stop=arr_stop, scheduled_departure_seconds=dep,
                scheduled_arrival_seconds=arr, departure=endpoint(actual_dep), arrival=endpoint(actual_arr),
                dropoff_allowed=True, pickup_allowed=True, outcome='MEASURED')


def training_data():
    rows = [measured(0, 'a', 0, 0, 1, 600, 620, 600, 620),
            measured(1, 'a', 1, 1, 2, 620, 640, 620, 640),
            measured(2, 'b', 0, 0, 1, 700, 720, 700, 720),
            measured(3, 'b', 1, 1, 2, 720, 740, 740, 760)]
    return dict(train_dates=['train'], dates={'train': {'connections': rows},
                                            'evaluation': {'connections': [measured(4, 'z', 0, 0, 1, 0, 20, 0, 9999)]}})


def graph():
    result = TransitGraph()
    for sid in range(3):
        result.add_stop(Stop(sid, str(sid), min_transfer_time=0))
    result.add_connection(Connection(0, 'R', 'trip', 0, 1, 600, 620))
    result.add_connection(Connection(1, 'R', 'trip', 1, 2, 620, 640))
    return result


def test_fit_uses_training_only_and_retains_actual_zero_elapsed_atom():
    data = training_data()
    fit = fit_empirical_durations(data)
    assert fit['groups']['travel']['20'] == dict(values=[20], counts=[4], n=4)
    assert fit['groups']['dwell']['0'] == dict(values=[0, 20], counts=[1, 1], n=2)
    changed = copy.deepcopy(data)
    changed['dates']['evaluation']['connections'][0]['arrival']['time_seconds'] = 1000000
    assert fit_empirical_durations(changed) == fit


def test_unmatched_vehicle_and_negative_elapsed_do_not_enter_training_atoms():
    data = training_data()
    rows = data['dates']['train']['connections']
    rows[0]['arrival']['official_journey'] = 'different'
    rows[3]['arrival']['time_seconds'] = rows[3]['departure']['time_seconds']-1
    fit = fit_empirical_durations(data)
    assert fit['excluded'] == dict(travel_unmatched_vehicle_identity=1,
                                  dwell_unmatched_vehicle_identity=1, travel_negative_elapsed=1)
    assert fit['groups']['travel']['20']['n'] == 2


def test_projection_preserves_mass_zero_and_endpoints_without_rejection_sampling():
    matrix = empirical_clock_matrix([0, 5, 30], [0, 10, 20], [0, 15, 100], [.2, .3, .5])
    np.testing.assert_allclose(matrix, [[.2, .15, .65], [0, .2, .8], [0, 0, 0]])


def test_interior_projection_preserves_mean_at_coarse_and_fine_resolutions():
    elapsed, weights = np.array([0., 3., 7., 19.]), np.array([.2, .2, .3, .3])
    previous = np.array([0., 10.])
    for step in (10, 5, 1):
        following = np.arange(0, 31, step)
        matrix = empirical_clock_matrix(previous, following, elapsed, weights)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.)
        np.testing.assert_allclose(matrix @ following, previous+elapsed @ weights)


def test_empirical_chain_matches_fitted_elapsed_and_keeps_initial_and_service_laws():
    fit = fit_empirical_durations(training_data())
    net = graph()
    params = {'R': dict(mean=0., std=.5, cancel_rate=.2)}
    belief = EmpiricalDurationBelief(net, duration_fit=fit, grid_step_seconds=10, public_occurrences=False)
    original = EvolvingTripBelief(net, grid_step_seconds=10, public_occurrences=False)
    law = belief._law('trip', params)
    np.testing.assert_allclose(law.initial, original._law('trip', params).initial)
    for index in range(3):
        edge = np.exp(law._transition(index))
        np.testing.assert_allclose(edge.sum(axis=1), 1.)
        assert not edge[law.times[index+1][None, :] < law.times[index][:, None]].any()
        if index != 1:
            assert not edge[law.canceled[index][:, None] != law.canceled[index+1][None, :]].any()
        else:
            np.testing.assert_allclose(edge[~law.canceled[index]][:, law.canceled[index+1]].sum(axis=1), .02)
    # A public first departure anchors the path while preserving future dwell uncertainty.
    branch = belief.copy()
    branch.observe_connection_event(0, 'departure', 0)
    times, mass, canceled = branch.event_distribution(1, 'departure', params)
    assert canceled == pytest.approx(.02)
    assert dict(zip(times[mass > 0], mass[mass > 0])) == pytest.approx({620: .5, 640: .5})
    assert belief._known_times == {}


def test_unseen_schedule_duration_uses_training_residuals_and_records_fallback():
    fit = fit_empirical_durations(training_data())
    belief = EmpiricalDurationBelief(graph(), duration_fit=fit, public_occurrences=False)
    values, weights = belief.duration_atoms('travel', 60, route='R')
    np.testing.assert_array_equal(values, [60])
    np.testing.assert_array_equal(weights, [1.])
    assert belief.fallback_duration_groups == {('R', 'travel', 60)}
    assert belief.residual_duration_groups == {('R', 'travel', 60)}
    assert belief.used_duration_groups == {('R', 'travel', 60)}


def test_observed_route_groups_stay_separate_and_unknown_routes_use_pooled_support():
    data = training_data()
    for row in data['dates']['train']['connections'][2:]:
        row['route_id'] = 'S'
        row['arrival']['time_seconds'] += 10
    fit = fit_empirical_durations(data)
    belief = EmpiricalDurationBelief(graph(), duration_fit=fit, public_occurrences=False)
    assert belief.duration_atoms('travel', 20, route='R')[0].tolist() == [20]
    assert belief.duration_atoms('travel', 20, route='S')[0].tolist() == [30]
    values, weights = belief.duration_atoms('travel', 20, route='unknown')
    assert values.tolist() == [20, 30]
    np.testing.assert_allclose(weights, [.5, .5])
    assert belief.pooled_duration_groups == {('unknown', 'travel', 20)}
    assert not belief.residual_duration_groups


def test_exact_receipts_outside_empirical_elapsed_support_fail_without_nan_or_smoothing():
    belief = EmpiricalDurationBelief(graph(), duration_fit=fit_empirical_durations(training_data()),
                                     grid_step_seconds=10, public_occurrences=False)
    belief.observe_connection_event(0, 'departure', 0)
    belief.observe_connection_event(0, 'arrival', -3)  # Actual elapsed17; every training travel atom is20.
    with pytest.raises(ValueError, match='no support under the empirical elapsed law'):
        belief.event_distribution(1, 'departure', {'R': dict(mean=0., std=.5, cancel_rate=0.)})
