"""Regression cases for concrete probability and execution failures in the review."""

import numpy as np
import pytest

from src.pmf import PMF, prob_reachable
from src.transit_graph import TransitGraph, Stop, Connection, StopLabel
from src.durner.topocsa import HyperpathResult, topocsa
from src.bandit_router import RouteBeliefState
from src.bandit_router_v2 import RouteEnsembleBelief
from src.bocd.regime_detector import DelayObservation, RegimeDetector
from src.simulator import (JourneyScenario, VehicleEvent, RegimeSchedule,
                           generate_delay_observations)
from src.simulate_bandit import simulate_bandit_journey


def chain(same_trip=True):
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=2))
    graph.add_connection(Connection(0, 'A', 'trip', 0, 1, 0, 10))
    graph.add_connection(Connection(1, 'A', 'trip' if same_trip else 'other', 1, 2, 10, 20))
    graph.assign_distributions({'A': {'delay_probs': np.array([1.0])}})
    return graph


def full_labels(graph):
    labels = {sid: [] for sid in graph.stops}
    for c in graph.connections:
        labels[c.dep_stop].append(StopLabel(c.id, PMF.deterministic(c.arr_time), c.arr_time, 1.0))
    return HyperpathResult(labels, None, 0.0, set(range(len(graph.connections))), 0, set())


class RecordingRouter:
    """A deliberately simple policy; executor must enforce feasibility itself."""
    def __init__(self, graph):
        self.cached_result = full_labels(graph)
        self.delays = []
        self.cancels = []

    def route(self, *args):
        return self.cached_result

    def select_connection(self, stop, time, rng, top_k):
        labels = self.cached_result.stop_labels.get(stop, [])
        if not labels:
            return None
        label = min(labels, key=lambda x: x.mean_dest_arrival)
        return label, label.mean_dest_arrival

    def observe_delay(self, route, delay):
        self.delays.append((route, delay))

    def observe_cancel(self, route, kind):
        self.cancels.append((route, kind))


def test_cancellation_mass_is_never_a_departure():
    arr = PMF.deterministic(11)
    assert prob_reachable(arr, PMF(np.array([0.5]), 10), 0) == 0.0
    assert prob_reachable(arr, PMF(np.array([0.0]), 12), 0) == 0.0
    assert prob_reachable(arr, PMF(np.array([0.5]), 12), 0) == 0.5


def test_same_vehicle_continues_without_transfer_buffer():
    graph = chain()
    result = topocsa(graph, 0, 2, 0)
    assert result.mean_arrival == 20
    assert result.stop_labels[0][0].feasibility == 1.0
    assert result.stop_labels[0][0].dest_arrival.total_prob == 1.0
    assert topocsa(chain(False), 0, 2, 0).dest_arrival is None


def test_equal_means_do_not_delete_different_departures():
    graph = chain()
    graph.add_connection(Connection(2, 'B', 'alternate', 1, 2, 12, 20,
                                    PMF.deterministic(12), PMF.deterministic(20)))
    result = topocsa(graph, 0, 2, 0)
    assert {label.connection_id for label in result.stop_labels[1]} == {1, 2}


def test_explicit_cancellation_maps_once_to_sub_pmf():
    graph = chain()
    graph.assign_distributions({'A': {'delay_probs': np.array([0.5]), 'cancel_prob': 0.5}})
    assert graph.connections[0].dep_distribution.total_prob == 0.5
    assert graph.connections[0].cancel_prob == 0.5


def test_normal_gamma_mean_variance_and_censored_outcomes():
    belief = RouteBeliefState(prior_mean=1, prior_var=2, prior_n=2)
    assert belief.posterior_var == 1.0
    belief.update_delay(3)
    belief.update_delay(5)
    assert belief.posterior_mean == 2.5
    # beta_n=2 + .5*2 + (2*2*3^2)/(2*4)=7.5; alpha_n=3,kappa_n=4.
    assert belief.posterior_var == 7.5 / 8
    for state in (belief, RouteEnsembleBelief()):
        previous = state.cancel_rate
        state.update_cancel('late_no_show')
        state.update_cancel('feed_missing')
        assert state.cancel_rate == previous
        state.update_cancel('true')
        assert state.cancel_rate > previous


def test_scenario_samples_once_per_trip(monkeypatch):
    from src import simulator
    draws = []
    def sample(connection, regime, rng):
        draws.append(connection.id)
        return len(draws)
    monkeypatch.setattr(simulator, 'sample_actual_delay', sample)
    graph = chain()
    scene = JourneyScenario.sample(graph, RegimeSchedule([(0, 'normal')]), np.random.default_rng(1))
    assert len(draws) == 1
    assert scene.event(graph.connections[0]).delay == scene.event(graph.connections[1]).delay
    assert scene.event(graph.connections[1]).departure == scene.event(graph.connections[0]).arrival
    with pytest.raises(TypeError):
        scene.events[0] = scene.events[1]


def test_only_published_events_and_no_duplicate_message():
    graph = chain()
    scene = JourneyScenario({0: VehicleEvent(2, False, 2, 12),
                             1: VehicleEvent(2, False, 12, 22)})
    seen = set()
    args = (graph, 1, 10, 'normal', np.random.default_rng(2))
    assert generate_delay_observations(*args, scenario=scene, seen_events=seen) == []
    first = generate_delay_observations(graph, 1, 12, 'normal', np.random.default_rng(3),
                                       scenario=scene, seen_events=seen)
    assert len(first) == 2
    assert all(obs.actual_time <= obs.timestamp <= 12 for obs in first)
    assert generate_delay_observations(graph, 1, 12, 'normal', np.random.default_rng(4),
                                       scenario=scene, seen_events=seen) == []
    detector = RegimeDetector()
    detector.update(first)
    history_len = len(detector.regime_history)
    detector.update(first)
    assert len(detector.regime_history) == history_len


@pytest.mark.parametrize('same_trip,arrival', [(True, 20), (False, 60)])
def test_executor_distinguishes_continuation_and_transfer(same_trip, arrival):
    graph = chain(same_trip)
    scene = JourneyScenario({0: VehicleEvent(0, False, 0, 10),
                             1: VehicleEvent(0, False, 10, 20)})
    router = RecordingRouter(graph)
    result = simulate_bandit_journey(graph, router, 0, 2, 0, RegimeSchedule([(0, 'normal')]),
                                    np.random.default_rng(1), max_time=60, scenario=scene)
    assert result.arrival_time == arrival
    assert result.n_transfers == 0
    if same_trip:
        assert len(router.delays) == 1  # boarding, departure and arrival are one trip outcome


def test_canceling_one_departure_does_not_cancel_the_line():
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'start'))
    graph.add_stop(Stop(1, 'destination'))
    graph.add_connection(Connection(0, 'A', 'early', 0, 1, 1, 6))
    graph.add_connection(Connection(1, 'A', 'later', 0, 1, 5, 10))
    scene = JourneyScenario({0: VehicleEvent(999, True, 1, 6),
                             1: VehicleEvent(0, False, 5, 10)})
    router = RecordingRouter(graph)
    result = simulate_bandit_journey(graph, router, 0, 1, 0, RegimeSchedule([(0, 'normal')]),
                                    np.random.default_rng(1), max_time=30, scenario=scene)
    assert result.arrival_time == 10
    assert router.cancels == [('A', 'true')]
    assert router.delays == [('A', 0)]


def test_departed_connection_does_not_reappear_or_stall():
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'start'))
    graph.add_stop(Stop(1, 'destination'))
    graph.add_connection(Connection(0, 'A', 'early', 0, 1, 100, 110))
    graph.add_connection(Connection(1, 'B', 'later', 0, 1, 102, 112))
    scene = JourneyScenario({0: VehicleEvent(-2, False, 98, 108),
                             1: VehicleEvent(0, False, 102, 112)})
    result = simulate_bandit_journey(graph, RecordingRouter(graph), 0, 1, 100,
                                    RegimeSchedule([(0, 'normal')]), np.random.default_rng(1),
                                    max_time=30, scenario=scene)
    assert result.arrival_time == 112
    assert [event.route for event in result.events if event.event_type == 'board'] == ['B']


@pytest.mark.parametrize('online_mean,expected_arrival', [(True, 20), (False, 30)])
def test_public_observations_can_restore_an_empty_initial_menu(online_mean, expected_arrival):
    from src.matched_router import MatchedRouter

    graph = TransitGraph()
    for sid in range(4):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=2))
    graph.add_connection(Connection(0, 'A', 'main', 0, 1, 1, 5))
    graph.add_connection(Connection(1, 'B', 'transfer', 1, 2, 10, 20))
    scene_events = {0: VehicleEvent(0, False, 1, 5), 1: VehicleEvent(0, False, 10, 20)}
    for cid in range(2, 7):
        graph.add_connection(Connection(cid, 'A', f'observed-{cid}', 0, 3, 5, 6))
        scene_events[cid] = VehicleEvent(-5, False, 0, 1)
    priors = {'A': {'mean': 30., 'std': .5, 'cancel_rate': 0.},
              'B': {'mean': 0., 'std': .5, 'cancel_rate': 0.}}
    router = MatchedRouter(graph, priors, online_mean=online_mean, max_time=30)
    result = simulate_bandit_journey(graph, router, 0, 2, 0, RegimeSchedule([(0, 'normal')]),
                                    np.random.default_rng(1), max_time=30,
                                    scenario=JourneyScenario(scene_events))
    assert 0 not in router.initial_menu
    assert result.arrival_time == expected_arrival
    assert any(event.event_type == 'stuck' for event in result.events) == (not online_mean)


def test_explicit_same_trip_continuation_can_proceed_without_a_cached_label():
    graph = chain()
    scene = JourneyScenario({0: VehicleEvent(0, False, 0, 10),
                             1: VehicleEvent(0, False, 10, 20)})

    class UnlabeledContinuationRouter(RecordingRouter):
        def __init__(self):
            super().__init__(graph)
            self.continuation = self.cached_result.stop_labels[1].pop()

        def set_execution_context(self, trip, last_arrival, unavailable_ids):
            self.trip = trip
            self.unavailable = unavailable_ids

        def has_unlabeled_continuation(self, stop):
            return stop == 1 and self.trip == 'trip' and 1 not in self.unavailable

        def select_connection(self, stop, time, rng, top_k):
            if self.has_unlabeled_continuation(stop):
                return self.continuation, self.continuation.mean_dest_arrival
            return super().select_connection(stop, time, rng, top_k)

    router = UnlabeledContinuationRouter()
    result = simulate_bandit_journey(graph, router, 0, 2, 0, RegimeSchedule([(0, 'normal')]),
                                    np.random.default_rng(1), max_time=30, scenario=scene)
    assert result.arrival_time == 20
    assert [event.time for event in result.events if event.event_type == 'board'] == [0, 10]
    assert result.n_transfers == 0


@pytest.mark.parametrize('executor', ['online', 'generic'])
def test_unpublished_future_cannot_shorten_waiting_commitment(executor):
    from types import SimpleNamespace
    from src.simulator import simulate_journey
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'start'))
    graph.add_stop(Stop(1, 'destination'))
    graph.add_connection(Connection(0, 'A', 'future', 0, 1, 20, 30))
    for future in (VehicleEvent(0, False, 20, 30),
                   VehicleEvent(20, False, 40, 50),
                   VehicleEvent(999, True, 20, 30)):
        scene = JourneyScenario({0: future})
        if executor == 'online':
            router = RecordingRouter(graph)
            router.decision_times = []
            original_select = router.select_connection
            def select(stop, time, rng, top_k):
                router.decision_times.append(time)
                return original_select(stop, time, rng, top_k)
            router.select_connection = select
            result = simulate_bandit_journey(graph, router, 0, 1, 0,
                RegimeSchedule([(0, 'normal')]), np.random.default_rng(0),
                max_time=13, scenario=scene)
            assert router.decision_times == [0, 12]
            assert router.delays == router.cancels == []
        else:
            router = SimpleNamespace(route=lambda *args: SimpleNamespace(
                hyperpath=full_labels(graph), computation_ms=0))
            result = simulate_journey(graph, router, 0, 1, 0,
                RegimeSchedule([(0, 'normal')]), np.random.default_rng(0),
                max_time=13, scenario=scene)
        assert [event.time for event in result.events if event.event_type == 'wait_late'] == [12, 13]
        assert result.arrival_time == 13


@pytest.mark.parametrize('executor', ['online', 'generic'])
def test_zero_minute_self_loop_is_boarded_once_then_same_trip_continues(executor):
    from types import SimpleNamespace
    from src.simulator import simulate_journey
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'start'))
    graph.add_stop(Stop(1, 'destination'))
    graph.add_connection(Connection(0, 'A', 'trip', 0, 0, 0, 0))
    graph.add_connection(Connection(1, 'A', 'trip', 0, 1, 0, 10))
    graph.assign_distributions({'A': {'delay_probs': np.array([1.0])}})
    scene = JourneyScenario({0: VehicleEvent(0, False, 0, 0),
                             1: VehicleEvent(0, False, 0, 10)})
    if executor == 'online':
        router = RecordingRouter(graph)
        calls = []
        original_select = router.select_connection
        def select(stop, time, rng, top_k):
            calls.append(time)
            assert len(calls) <= 3, 'execution retried an already boarded connection'
            return original_select(stop, time, rng, top_k)
        router.select_connection = select
        simulate = simulate_bandit_journey
    else:
        labels = full_labels(graph)
        labels.stop_labels[0].sort(key=lambda x: -x.mean_dest_arrival)
        class BoundedLabels(list):
            iterations = 0
            def __reversed__(self):
                self.iterations += 1
                assert self.iterations <= 3, 'execution retried an already boarded connection'
                return super().__reversed__()
        labels.stop_labels[0] = BoundedLabels(labels.stop_labels[0])
        router = SimpleNamespace(route=lambda *args: SimpleNamespace(
            hyperpath=labels, computation_ms=0))
        simulate = simulate_journey
    result = simulate(graph, router, 0, 1, 0, RegimeSchedule([(0, 'normal')]),
                      np.random.default_rng(0), max_time=30, scenario=scene)
    boards = [event for event in result.events if event.event_type == 'board']
    assert result.arrival_time == 10
    assert len(boards) == 2
    assert [event.time for event in boards] == [0, 0]
    assert result.n_transfers == 0
