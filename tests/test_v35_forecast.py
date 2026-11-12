from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v34 import gated_rollout
from experiments.reconstruction_v34.conditional_reference import restore_snapshot
from experiments.reconstruction_v34.forecast import MessageHistory
from experiments.reconstruction_v33.nodes import Ride
from experiments.reconstruction_v35.fit import fit_emissions
from experiments.reconstruction_v35.forecast import FittedMessageHistory
from src.evolving_trip_belief import _ChainLaw
from src.transit_graph import Connection, Stop, TransitGraph


def fixture():
    graph = TransitGraph()
    for stop in range(2):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    graph.add_connection(Connection(0, 'R', 'a', 0, 1, 0, 10))
    law = _ChainLaw([np.array([0.]), np.array([10., 70.])], np.array([0.]),
        lambda slot: np.log(np.array([[.5, .5]])), [np.zeros(1), np.zeros(2)],
        [np.zeros(1, bool), np.zeros(2, bool)], (True,))
    model = SimpleNamespace(graph=graph, _positions={(0, 'departure'): ('a', 0), (0, 'arrival'): ('a', 1)},
                            _law=lambda trip, parameters: law)
    parameters = {'R': {'std': 1.}}
    fitted = dict(pooled={kind: dict(bias_seconds=0., sd_seconds=30., n=10)
                         for kind in ('departure', 'arrival')},
                  route_kind={'R': {'arrival': dict(bias_seconds=60., sd_seconds=30., n=2)}})
    return model, parameters, fitted, law


def frame(receipt, prediction=10., status='FORECAST'):
    return dict(receipt_seconds=receipt, trip_statuses=[],
                endpoint_updates=[[0, 'arrival', status, prediction]])


def test_fit_uses_only_eligible_training_errors_and_shrinks_both_moments():
    rows = [dict(date='2023-10-01', split='training', error_eligible=True,
                 route_id=route, kind=kind, error_seconds=error)
            for route, kind, error in [('R', 'departure', 0), ('R', 'departure', 20),
                                       ('S', 'departure', 100), ('R', 'arrival', 5)]]
    rows += [dict(date='2023-10-02', split='training', error_eligible=False),
             dict(date='2023-10-03', split='validation', error_eligible=True,
                  route_id='R', kind='departure', error_seconds=100000)]
    fit = fit_emissions(rows, training_dates=['2023-10-01', '2023-10-02'])
    pool = fit['pooled']['departure']
    assert pool['n'] == 3 and pool['bias_seconds'] == 40.
    route = fit['route_kind']['R']['departure']
    assert route['bias_seconds'] == pytest.approx((20+20*40)/22)
    assert route['second_moment_seconds_squared'] == pytest.approx((400+20*(10400/3))/22)
    assert fit['pooled']['arrival']['sd_seconds'] == 30.
    assert fit['route_kind']['R']['arrival']['sd_seconds'] == 30.
    assert fit['eligible_errors_by_date'] == {'2023-10-01': 4}


def test_fit_rejects_a_training_label_outside_the_frozen_two_dates():
    with pytest.raises(ValueError, match='outside'):
        fit_emissions([dict(date='2023-10-25', split='training')],
                      training_dates=['2023-10-01', '2023-10-02'])


def test_error_bias_sign_is_correct_and_raw_public_forecast_is_not_changed():
    model, parameters, fitted, prior = fixture()
    history = FittedMessageHistory(model, parameters, fit=fitted, frames=[frame(1)])
    history.advance(1)
    expected = np.exp(np.array([-2., 0.]))
    expected /= expected.sum()
    assert history.law('a').marginal(1) == pytest.approx(expected)
    forecast = next(r for r in history.public_state() if r['evidence'] == 'forecast')
    assert forecast['predicted_time'] == 10. and forecast['receipt_seconds'] == 1.
    assert forecast['emission_bias_seconds'] == 60. and forecast['emission_sd_seconds'] == 30.
    assert prior.marginal(1) == pytest.approx([.5, .5])
    assert history.law('a')._transition(0) == pytest.approx(prior._transition(0))
    assert history.law('a').initial == pytest.approx(prior.initial)


def test_missing_route_kind_uses_the_kind_pool_and_not_another_route():
    model, parameters, fitted, _ = fixture()
    history = FittedMessageHistory(model, parameters, fit=fitted)
    assert history.emission_parameters(0, 'departure') is fitted['pooled']['departure']
    fitted['route_kind'] = {}
    assert history.emission_parameters(0, 'arrival') is fitted['pooled']['arrival']


def test_revisions_replace_likelihood_and_hard_observation_overrides_it():
    model, parameters, fitted, _ = fixture()
    history = FittedMessageHistory(model, parameters, fit=fitted,
        frames=[frame(1), frame(2), frame(3, None, 'NO_DATA'), frame(4)])
    history.advance(1)
    mass = history.law('a').marginal(1).copy()
    history.advance(2)
    assert history.law('a').marginal(1) == pytest.approx(mass)
    history.advance(3)
    assert history.law('a').marginal(1) == pytest.approx([.5, .5])
    history.observe(0, 'arrival', 10)
    history.advance(4)
    assert history.law('a').marginal(1) == pytest.approx([1., 0.])
    assert not history._forecasts


def test_clone_snapshot_equals_previous_reconstruction_for_legacy_history():
    model, parameters, _, _ = fixture()
    history = MessageHistory(model, parameters, frames=[frame(1), frame(99, 70)])
    history.advance(1)
    history.observe(0, 'departure', 0)
    history.law('a')
    rebuilt = restore_snapshot(model, parameters, history.public_state(), 1)
    snapshot = gated_rollout.freeze_snapshot(history, 1)
    assert snapshot._laws == {} and snapshot.frames == () and snapshot.frame_index == 0
    assert snapshot.public_state() == rebuilt.public_state()
    assert snapshot.signature() == rebuilt.signature()
    assert snapshot.law('a').marginal(1) == pytest.approx(rebuilt.law('a').marginal(1))
    assert snapshot.advance(100)['frames_consumed'] == 0
    assert history.frame_index == 1


def test_conditional_rollout_preserves_the_fitted_type_bias_and_snapshot(monkeypatch):
    model, parameters, fitted, _ = fixture()
    history = FittedMessageHistory(model, parameters, fit=fitted, frames=[frame(1), frame(99, 70)])
    history.advance(1)
    history.observe(0, 'departure', 0)
    expected = history.law('a').marginal(1).copy()
    sampled = []

    def worlds(belief, parameters, paths, count, seed):
        sampled.append(belief._law('a', parameters).marginal(1).copy())
        return SimpleNamespace(belief=belief)

    class Continuation:
        def __init__(self, *args, **kwargs):
            pass

        def forced(self, node, row, action, snapshot, world):
            assert isinstance(snapshot, FittedMessageHistory)
            assert snapshot.fit is fitted and snapshot.frames == ()
            assert snapshot.law('a').marginal(1) == pytest.approx(expected)
            return dict(loss=1., status='arrived')

    monkeypatch.setattr(gated_rollout, 'TripWorlds', worlds)
    monkeypatch.setattr(gated_rollout, 'SnapshotExecutor', Continuation)
    monkeypatch.setattr(gated_rollout, 'ForecastPlanner', lambda *args: SimpleNamespace(stats={}))
    result = gated_rollout.conditional_action_check(model.graph, model, parameters, 100,
        Ride(0, (0,), {0: None}), dict(kind='onboard', cid=0, clock=1), history,
        'continue', 'alight', count=2, seed=1)
    assert sampled[0] == pytest.approx(expected)
    assert result['selected_action'] == 'alight'
    assert history.frames[1]['receipt_seconds'] == 99 and history.frame_index == 1
