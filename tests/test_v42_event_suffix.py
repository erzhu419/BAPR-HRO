import numpy as np
import pytest

from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v33.execute import DrawnWorld
from experiments.reconstruction_v33.nodes import Ride
from experiments.reconstruction_v38.planner import FastPlanner
from experiments.reconstruction_v39.state import DecisionEngine
from experiments.reconstruction_v41.continuation import science
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory, PhysicalPlanner
from experiments.reconstruction_v42.event_suffix import EventSuffixExecutor, evaluate, new_stats
from src import evolving_trip_belief as evolving
from test_v38_evaluator import fixture


def executor(graph, model, parameters, *, mode='event_live', suffixes=None, stats=None):
    planner = FastPlanner(graph, model, parameters, 100)
    return EventSuffixExecutor(graph, planner, 100, engine=DecisionEngine(planner),
        method='rolling_fixed', waiting='combined', mode=mode,
        suffixes={} if suffixes is None else suffixes,
        meeting_stats=new_stats() if stats is None else stats)


@pytest.mark.parametrize('mode', ['full', 'entry', 'event_full', 'event_live'])
def test_same_v41_bank_preserves_all_scientific_fields(mode):
    graph, model, parameters, node, history = fixture()
    args = (graph, model, parameters, 100, node, dict(kind='onboard', cid=0, clock=10), history)
    baseline = evaluate(*args, planner_type=FastPlanner, count=16, seed=7, mode='full')
    candidate = evaluate(*args, planner_type=FastPlanner, count=16, seed=7, mode=mode)
    assert science(candidate) == science(baseline)
    assert candidate['generated_count'] == 128
    assert history.frame_index == 1 and history.frames


@pytest.mark.parametrize('planner_type', [PhysicalPlanner, MatchedContinuationPlanner])
def test_v41_physical_and_shallow_policies_preserve_each_sample(monkeypatch, planner_type):
    monkeypatch.setattr(evolving, 'DELAYS', np.array([0., 1.]))
    graph, _, _, node, original = fixture()
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False,
        travel_sd=.5, dwell_sd=.5, cancel_persistence=.8)
    parameters = {'R': dict(mean=0., std=.5, cancel_rate=.05)}
    history = PhysicalHistory(model, parameters, fit=original.fit)
    history.apply_frame(original.frames[0], 10)
    history.observe(0, 'departure', 0)
    history.observe(0, 'arrival', 10)
    args = (graph, model, parameters, 180, node, dict(kind='onboard', cid=0, clock=10), history)
    baseline = evaluate(*args, planner_type=planner_type, count=16, seed=42, mode='full')
    for mode in ('event_full', 'event_live'):
        candidate = evaluate(*args, planner_type=planner_type, count=16, seed=42, mode=mode)
        assert science(candidate) == science(baseline)
        assert candidate['meeting_stats']['checks'] > 0


def test_dead_trip_records_merge_complete_suffix_but_full_history_does_not():
    graph, model, parameters, node, history = fixture()
    second = history.clone()
    second.observe(1, 'arrival', 30)  # source cannot reappear in this terminal menu.
    platform = node.exits[1]
    bank = TripWorlds(history.as_belief(), parameters, node.paths(), 128, 7)
    for mode in ('event_full', 'event_live'):
        suffixes, stats, outcomes = {}, new_stats(), []
        for h in (history, second):
            current = executor(graph, model, parameters, mode=mode, suffixes=suffixes, stats=stats)
            outcomes.append(current.forced(platform,
                dict(kind='platform', entry=30, clock=30), None, h.clone(), DrawnWorld(bank, 0)))
        for field in ('loss', 'status', 'trace', 'decisions'):
            assert outcomes[0][field] == outcomes[1][field]
        assert stats['hits'] == (1 if mode == 'event_live' else 0)
        assert stats['avoided_decisions'] == (1 if mode == 'event_live' else 0)


def test_live_same_trip_past_observations_remain_in_event_key():
    graph, model, parameters, _, history = fixture()
    current = executor(graph, model, parameters)
    view = Ride(3, (3,), {3: None})
    altered = history.clone()
    altered.observe(2, 'arrival', 25)
    assert current.event_key('onboard_event', view, history, 50, (3, 'middle')) != (
        current.event_key('onboard_event', view, altered, 50, (3, 'middle')))


def test_unused_forecast_remains_distinct_and_can_reenter_after_observation():
    graph, model, parameters, _, history = fixture()
    current = executor(graph, model, parameters)
    a, b = history.clone(), history.clone()
    for h, later_prediction in ((a, 48.), (b, 70.)):
        h.apply_frame(dict(receipt_seconds=6, trip_statuses=[], endpoint_updates=[
            [3, 'arrival', 'FORECAST', later_prediction]]), 6)
        h.apply_frame(dict(receipt_seconds=7, trip_statuses=[], endpoint_updates=[
            [2, 'arrival', 'FORECAST', 25.]]), 7)
    assert a.selected_forecast('middle') == b.selected_forecast('middle')
    view = Ride(2, (2, 3), {3: None})
    assert current.event_key('ride', view, a, 12, ()) != current.event_key('ride', view, b, 12, ())
    for h in (a, b):
        h.observe(2, 'arrival', 25)
    assert a.selected_forecast('middle') != b.selected_forecast('middle')


def test_platform_entry_ready_and_current_offers_are_not_forgotten():
    graph, model, parameters, node, history = fixture()
    current = executor(graph, model, parameters)
    platform = node.exits[1]
    original = current.event_key('platform_event', platform, history, 32, (30., 30., (4,), (4,)))
    for context in ((20., 30., (4,), (4,)), (30., 31., (4,), (4,)),
                    (30., 30., (), (4,)), (30., 30., (4,), ())):
        assert current.event_key('platform_event', platform, history, 32, context) != original


def test_live_basis_identity_is_part_of_execution_context():
    graph, model, parameters, node, history = fixture()
    current = executor(graph, model, parameters)
    a, b = history.clone(), history.clone()
    a._physical_bases = {'source': object(), 'last': object()}
    b._physical_bases = dict(a._physical_bases)
    platform = node.exits[1]
    context = (30., 30., (4,), (4,))
    key = current.event_key('platform_event', platform, a, 32, context)
    b._physical_bases['source'] = object()
    assert current.event_key('platform_event', platform, b, 32, context) == key
    b._physical_bases['last'] = object()
    assert current.event_key('platform_event', platform, b, 32, context) != key
