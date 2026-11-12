from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v34.forecast import ForecastPlanner
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from src.evolving_trip_belief import _ChainLaw
from src.transit_graph import Connection, Stop, TransitGraph


def fixture(frames=()):
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    graph.add_connection(Connection(0, 'R', 'a', 0, 1, 0, 10))
    graph.add_connection(Connection(1, 'R', 'a', 1, 2, 30, 40))
    with np.errstate(divide='ignore'):
        deterministic = np.log(np.eye(2))
    law = _ChainLaw([np.array([0.]), np.array([10., 70.]), np.array([30., 90.]), np.array([40., 100.])],
        np.array([0.]), lambda slot: np.log(np.array([[.5, .5]])) if slot == 0 else deterministic,
        [np.zeros(1), np.zeros(2), np.zeros(2), np.zeros(2)],
        [np.zeros(1, bool), np.zeros(2, bool), np.zeros(2, bool), np.zeros(2, bool)], (True, True, True))
    positions = {(cid, kind): ('a', slot) for slot, (cid, kind) in
                 enumerate(((0, 'departure'), (0, 'arrival'), (1, 'departure'), (1, 'arrival')))}
    model = SimpleNamespace(graph=graph, _positions=positions, _law=lambda trip, params: law)
    fit = dict(pooled={kind: dict(bias_seconds=0., sd_seconds=30., n=10)
                       for kind in ('departure', 'arrival')}, route_kind={})
    return SingleMessageHistory(model, {'R': {'std': 1.}}, fit=fit, frames=frames), law


def frame(receipt, updates):
    return dict(receipt_seconds=receipt, trip_statuses=[], endpoint_updates=updates)


def test_selector_prefers_latest_receipt_then_physical_slot_without_using_forecast_value():
    history, _ = fixture([frame(1, [[0, 'arrival', 'FORECAST', 10], [1, 'arrival', 'FORECAST', 100]]),
                          frame(2, [[1, 'arrival', 'FORECAST', -1000]]),
                          frame(3, [[0, 'arrival', 'FORECAST', 9999], [1, 'departure', 'FORECAST', -2000]])])
    history.advance(1)
    assert history.selected_forecast('a')[:2] == (0, 'arrival')
    history.advance(2)
    assert history.selected_forecast('a')[:2] == (1, 'arrival')
    history.advance(3)
    assert history.selected_forecast('a')[:2] == (0, 'arrival')
    assert len(history.raw_forecast_records('a')) == 3 and len(history.forecast_records('a')) == 1


def test_only_selected_forecast_enters_the_joint_trip_likelihood_and_raw_records_remain():
    history, prior = fixture([frame(1, [[0, 'arrival', 'FORECAST', 10], [1, 'arrival', 'FORECAST', 100]])])
    history.advance(1)
    expected = np.exp(np.array([0., -2.]))
    expected /= expected.sum()
    assert history.law('a').marginal(3) == pytest.approx(expected)
    rows = [r for r in history.public_state() if r['evidence'] == 'forecast']
    assert [(r['cid'], r['predicted_time'], r['emission_selected']) for r in rows] == [(0, 10., True), (1, 100., False)]
    assert all(r['emission_sd_seconds'] == 30. for r in rows)
    assert prior.marginal(3) == pytest.approx([.5, .5])


def test_observed_passage_excludes_the_entire_completed_prefix_and_reselects_suffix():
    history, _ = fixture([frame(1, [[1, 'arrival', 'FORECAST', 100]]),
                          frame(2, [[0, 'arrival', 'FORECAST', 10]])])
    history.advance(2)
    assert history.selected_forecast('a')[:2] == (0, 'arrival')
    history.law('a')
    # The earlier arrival's exact time was never supplied, but passage at the
    # following departure proves that physical slot is already complete.
    history.observe(1, 'departure', 90)
    assert history.selected_forecast('a')[:2] == (1, 'arrival')
    assert (0, 'arrival') in history._forecasts
    assert history.law('a').marginal(3) == pytest.approx([0., 1.])
    history.observe(1, 'arrival', 100)
    assert history.selected_forecast('a') is None
    assert len(history.raw_forecast_records('a')) == 1


def test_advance_without_a_new_receipt_does_not_filter_a_delayed_scheduled_event():
    history, _ = fixture([frame(1, [[1, 'arrival', 'FORECAST', 100]])])
    history.advance(1)
    before = history.selected_forecast('a')
    history.advance(1000)
    assert history.selected_forecast('a') == before


def test_clone_revisions_and_withdrawal_do_not_reuse_a_stale_selected_law():
    history, _ = fixture([frame(1, [[0, 'arrival', 'FORECAST', 10], [1, 'arrival', 'FORECAST', 100]]),
                          frame(2, [[1, 'arrival', 'FORECAST', 100]]),
                          frame(3, [[1, 'arrival', 'NO_DATA', None]])])
    history.advance(1)
    first = history.law('a').marginal(3).copy()
    branch = history.clone()
    branch.advance(2)
    assert branch.selected_forecast('a')[:2] == (1, 'arrival')
    assert branch.law('a').marginal(3)[1] > .8
    assert history.law('a').marginal(3) == pytest.approx(first)
    branch.advance(3)
    assert branch.selected_forecast('a')[:2] == (0, 'arrival')
    assert branch.law('a').marginal(3) == pytest.approx(first)


def test_all_public_service_notices_apply_even_to_an_unselected_endpoint():
    history, _ = fixture([frame(1, [[0, 'arrival', 'FORECAST', 10], [1, 'departure', 'FORECAST', 90]]),
                          frame(2, [[1, 'departure', 'SKIPPED', None]])])
    history.advance(2)
    assert history.selected_forecast('a')[:2] == (0, 'arrival')
    planner = ForecastPlanner(history.belief.graph, history.belief, history.parameters, 200)
    assert planner.departure_distribution(1, history)[1] == pytest.approx([0., 0.])
    history.observe(1, 'departure', 90)
    assert not history.announced_unavailable(1, 'departure')


def test_rollout_snapshot_keeps_the_single_message_type_fit_and_raw_public_forecasts():
    history, _ = fixture([frame(1, [[0, 'arrival', 'FORECAST', 10], [1, 'arrival', 'FORECAST', 100]]),
                          frame(99, [[1, 'arrival', 'FORECAST', 100]])])
    history.advance(1)
    history.law('a')
    snapshot = freeze_snapshot(history, 1)
    assert isinstance(snapshot, SingleMessageHistory) and snapshot.fit is history.fit
    assert snapshot.frames == () and snapshot._laws == {}
    assert snapshot.public_state() == history.public_state()
    assert snapshot.advance(100)['frames_consumed'] == 0
    assert snapshot.selected_forecast('a')[:2] == (0, 'arrival')
    assert snapshot.law('a').marginal(3) == pytest.approx(history.law('a').marginal(3))


def test_unselected_future_forecast_does_not_prevent_a_valid_prior_future_edge():
    history, _ = fixture([frame(1, [[0, 'arrival', 'FORECAST', 10], [1, 'arrival', 'FORECAST', 100]])])
    history.advance(1)
    planner = ForecastPlanner(history.belief.graph, history.belief, history.parameters, 200)
    assert planner._transition_key('a', 1, history)[1] is True
    history.apply_frame(frame(2, [[1, 'arrival', 'FORECAST', 100]]), 2)
    assert planner._transition_key('a', 1, history)[1] is False
