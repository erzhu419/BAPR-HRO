from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v34.forecast import MessageHistory, ForecastPlanner
from experiments.reconstruction_v33.nodes import Ride
from src.evolving_trip_belief import _ChainLaw
from src.transit_graph import Connection, Stop, TransitGraph


def fixture(frames=()):
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    graph.add_connection(Connection(0, 'R', 'a', 0, 1, 0, 10))
    graph.add_connection(Connection(1, 'R', 'b', 1, 2, 5, 20))
    times = [np.array([0.]), np.array([10., 30.])]
    law = _ChainLaw(times, np.array([0.]), lambda slot: np.log(np.array([[.5, .5]])),
                    [np.zeros(1), np.zeros(2)], [np.zeros(1, bool), np.zeros(2, bool)], (True,))
    positions = {(0, 'departure'): ('a', 0), (0, 'arrival'): ('a', 1),
                 (1, 'departure'): ('b', 0), (1, 'arrival'): ('b', 1)}
    belief = SimpleNamespace(graph=graph, _positions=positions, _law=lambda trip, parameters: law)
    return MessageHistory(belief, {'R': {'std': 1.}}, frames=frames, forecast_sd={0: 3., 1: 3.}), law


def frame(receipt, prediction=30., status='FORECAST', cid=0, kind='arrival'):
    return dict(receipt_seconds=receipt, trip_statuses=[],
                endpoint_updates=[[cid, kind, status, prediction]])


def test_future_message_is_not_available_early_and_a_forecast_is_not_exact_truth():
    history, prior = fixture([frame(5)])
    assert history.advance(4)['frames_consumed'] == 0
    assert history.law('a').marginal(1) == pytest.approx(prior.marginal(1))
    assert history.advance(5)['frames_consumed'] == 1
    mass = history.law('a').marginal(1)
    assert 0. < mass[0] < .01 and mass[1] < 1.
    assert not history._records
    with pytest.raises(ValueError, match='availability'):
        history.apply_frame(frame(6), 5)


def test_latest_prediction_replaces_previous_emission_and_repeats_do_not_sharpen_it():
    history, _ = fixture([frame(1, 30), frame(2, 30), frame(3, 10)])
    history.advance(1)
    once = history.law('a').marginal(1).copy()
    history.advance(2)
    assert history.law('a').marginal(1) == pytest.approx(once)
    history.advance(3)
    fresh, _ = fixture([frame(3, 10)])
    fresh.advance(3)
    assert history.law('a').marginal(1) == pytest.approx(fresh.law('a').marginal(1))
    assert history.law('a').marginal(1)[0] > .99


@pytest.mark.parametrize('status', ['NO_DATA', 'SKIPPED'])
def test_withdrawal_removes_soft_forecast_without_creating_actual_cancellation(status):
    history, prior = fixture([frame(1), frame(2, None, status)])
    history.advance(2)
    assert history.law('a').marginal(1) == pytest.approx(prior.marginal(1))
    assert history.endpoint_status(0, 'arrival') == (status, 2.)
    assert not history._records and not history._forecasts


def test_own_observation_overrides_same_event_forecasts_but_retains_other_trip_information():
    history, _ = fixture([frame(1), frame(1, cid=1), frame(3, 30)])
    history.advance(1)
    history.observe(0, 'arrival', 10)
    history.advance(3)
    assert history.law('a').marginal(1) == pytest.approx([1., 0.])
    assert (0, 'arrival') not in history._forecasts
    assert (1, 'arrival') in history._forecasts


def test_trip_notice_is_public_metadata_not_an_actual_service_emission():
    notice = dict(receipt_seconds=2, trip_statuses=[['a', 'CANCELED']], endpoint_updates=[])
    history, prior = fixture([frame(1), notice])
    history.advance(2)
    assert history.trip_status('a') == ('CANCELED', 2.)
    assert history.law('a').marginal(1) == pytest.approx(prior.marginal(1))
    assert not history._records
    assert any(r['evidence'] == 'trip_status' for r in history.public_state())


def test_an_absent_endpoint_update_does_not_withdraw_the_previous_prediction():
    history, _ = fixture([frame(1), dict(receipt_seconds=2, trip_statuses=[['a', 'SCHEDULED']],
                                        endpoint_updates=[])])
    history.advance(1)
    previous = history.law('a').marginal(1).copy()
    history.advance(2)
    assert history.law('a').marginal(1) == pytest.approx(previous)
    assert history._forecasts[0, 'arrival'][2] == 1.
    assert history.trip_status('a') == ('SCHEDULED', 2.)


def test_clones_and_relevant_trip_signatures_keep_revisions_separate():
    history, _ = fixture([frame(1), frame(2, cid=1), frame(3, 10)])
    history.advance(1)
    signature, law = history.signature(('a',)), history.law('a')
    branch = history.clone()
    branch.advance(2)
    assert branch.signature(('a',)) == signature
    assert branch.signature() != history.signature()
    assert branch.law('a') is law
    branch.advance(3)
    assert branch.signature(('a',)) != signature
    assert history.signature(('a',)) == signature
    assert branch.as_belief()._law('a', {}) is branch.law('a')


def test_future_forecast_cannot_be_lost_by_the_prior_future_edge_shortcut():
    history, _ = fixture([frame(1)])
    planner = ForecastPlanner(history.belief.graph, history.belief, history.parameters, 100.)
    before = planner._edge('a', 0, history).copy()
    history.advance(1)
    assert planner._transition_key('a', 0, history)[1] is False
    updated = planner._edge('a', 0, history)
    assert before == pytest.approx(np.array([[.5, .5]]))
    assert updated[0, 1] > .99
    ride = Ride(0, (0,), {0: None})
    assert planner.compile_ride(ride, history).score(0) > 29.9
    history.apply_frame(frame(2, None, 'NO_DATA'), 2)
    assert planner._transition_key('a', 0, history)[1] is True
    assert planner._edge('a', 0, history) == pytest.approx(before)
    assert planner.compile_ride(ride, history).score(0) == pytest.approx(20.)


def test_source_mapping_and_old_revision_cannot_restore_a_withdrawn_prediction():
    history, prior = fixture()
    history.source_to_local = {99: 0}
    history.apply_frame(frame(2, 30, cid=99), 2)
    history.apply_frame(frame(3, None, 'NO_DATA', cid=99), 3)
    history.apply_frame(frame(1, 10, cid=99), 3)
    assert history.law('a').marginal(1) == pytest.approx(prior.marginal(1))
    assert history.endpoint_status(0, 'arrival') == ('NO_DATA', 3.)


def test_trip_departure_mask_is_shared_and_later_schedule_or_local_offer_restores_it():
    history, prior = fixture()
    history.apply_frame(dict(receipt_seconds=1, trip_statuses=[['a', 'CANCELED']], endpoint_updates=[]), 1)
    planner = ForecastPlanner(history.belief.graph, history.belief, history.parameters, 100.)
    clocks, mass, canceled = planner.departure_distribution(0, history)
    assert mass.tolist() == [0.] and canceled == 1.
    assert history.law('a').marginal(0) == pytest.approx(prior.marginal(0))
    branch = history.clone()
    branch.observe(0, 'departure', 0)
    assert planner.departure_distribution(0, branch)[1] == pytest.approx([1.])
    history.apply_frame(dict(receipt_seconds=2, trip_statuses=[['a', 'SCHEDULED']], endpoint_updates=[]), 2)
    assert planner.departure_distribution(0, history)[1] == pytest.approx([1.])


@pytest.mark.parametrize('method', ['nominal', 'rolling_fixed', 'one_step_rollout', 'optimal'])
def test_every_controller_has_the_same_future_alighting_mask_and_recovery(method):
    history, _ = fixture()
    planner = ForecastPlanner(history.belief.graph, history.belief, history.parameters, 100.)
    ride = Ride(0, (0,), {0: None})
    history.apply_frame(frame(1, None, 'SKIPPED'), 1)
    assert planner.compile_ride(ride, history, method).score(0) == 100.
    history.apply_frame(frame(2, None, 'NO_DATA'), 2)
    assert planner.compile_ride(ride, history, method).score(0) == 100.
    branch = history.clone()
    branch.observe(0, 'arrival', 10)
    assert planner.compile_ride(ride, branch, method).score(0) == pytest.approx(10.)
    history.apply_frame(frame(3, None, 'SCHEDULED'), 3)
    assert planner.compile_ride(ride, history, method).score(0) == pytest.approx(20.)


def test_no_data_preserves_service_notice_in_cache_keys_and_forecast_can_revoke_skip():
    skipped, _ = fixture()
    ordinary, _ = fixture()
    skipped.apply_frame(frame(1, None, 'SKIPPED'), 1)
    ordinary.apply_frame(frame(1, 30), 1)
    for history in (skipped, ordinary):
        history.apply_frame(frame(2, None, 'NO_DATA'), 2)
    assert skipped.endpoint_status(0, 'arrival') == ordinary.endpoint_status(0, 'arrival')
    assert skipped.signature(('a',)) != ordinary.signature(('a',))
    assert skipped.announced_unavailable(0, 'arrival')
    assert not ordinary.announced_unavailable(0, 'arrival')
    skipped.apply_frame(frame(3, 10), 3)
    assert not skipped.announced_unavailable(0, 'arrival')
