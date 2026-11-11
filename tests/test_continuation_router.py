import numpy as np
import pytest
from src.committed_transit import CommittedState, WAIT
from src.continuation_router import ContinuationRouter, nominal_action
from src.matched_router import MatchedRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import JourneyScenario, VehicleEvent, RegimeSchedule
from src.transit_graph import TransitGraph, Stop, Connection
from src.trip_belief import NominalEvent, TripBelief


def network(rows):
    graph = TransitGraph()
    for sid in range(4):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=2))
    for route, trip, source, target, dep, arr in rows:
        graph.add_connection(Connection(-1, route, trip, source, target, dep, arr))
    return graph


def priors(graph):
    return {c.route: {"mean": 0., "std": .5, "cancel_rate": .01} for c in graph.connections}


def test_known_shared_trip_delay_is_used_for_its_future_continuation():
    graph = network([("A", "T", 0, 1, 0, 2), ("A", "T", 1, 2, 3, 4)])
    scene = JourneyScenario({c.id: VehicleEvent(5, False, c.dep_time + 5, c.arr_time + 5)
                             for c in graph.connections})
    router = ContinuationRouter(graph, priors(graph), samples=2, max_time=30)
    result = simulate_bandit_journey(graph, router, 0, 2, 0, RegimeSchedule([(0, "x")]),
                                    np.random.default_rng(0), max_time=30, scenario=scene)
    assert result.arrival_time == 9
    continuation = next(p for p in router.predictions if p["connection"] == 1)
    assert continuation["time"] == 7
    assert continuation["score"] == 9
    assert continuation["first_attempt_failure_probability"] == 0


def test_hypothetical_future_does_not_update_real_trip_belief():
    graph = network([("A", "A", 0, 2, 1, 4), ("B", "B", 0, 2, 2, 6)])
    router = ContinuationRouter(graph, priors(graph), samples=2, max_time=20)
    router.route(0, 2, 0)
    router.set_execution_context(None, 0, set())
    router.set_information_context(0, 0, None, 0, set(), set(), set())
    router.select_connection(0, 0, np.random.default_rng(0))
    assert router.trip_belief.known == {}
    assert router.predictions[-1]["candidates"]


@pytest.mark.parametrize("rescue, expected", [(True, 12), (False, 60)])
def test_known_missed_attempt_has_a_causal_recovery_value(rescue, expected):
    graph = network([("A", "A", 0, 2, 3, 7), ("B", "B", 0, 2, 6, 12)])
    router = ContinuationRouter(graph, priors(graph), rescue=rescue, samples=2, max_time=60)
    router.route(0, 2, 0)
    # These vehicle facts can have been announced at their previous stops.
    router.observe_vehicle_event(0, False, 0)
    router.observe_vehicle_event(1, False, 0)
    router.set_execution_context(None, 0, set())
    router.set_information_context(0, 5, None, 0, set(), set(), set())
    chosen, _ = router.select_connection(0, 5, np.random.default_rng(0))
    assert chosen.connection_id == 1
    assert 0 not in router.predictions[-1]["candidates"]
    estimate = router._score_action(("select", 0))
    assert estimate["score"] == expected
    assert estimate["first_attempt_failure_probability"] == 1
    assert estimate["conditional_first_failure_loss"] == 12


def test_same_trip_boarding_survives_label_and_candidate_limits():
    graph = network([("A", "A", 0, 1, 0, 2), ("A", "A", 1, 2, 2, 3),
                     ("B", "B", 1, 2, 4, 5)])
    router = ContinuationRouter(graph, priors(graph), samples=2, candidate_limit=1)
    router.route(0, 2, 0)
    router.cached_result.stop_labels[1] = []
    router._fresh_menu.discard(1)
    router.observe_vehicle_event(0, False, 0)
    router.set_execution_context("A", 2, {0})
    router.set_information_context(1, 2, "A", 2, {0}, set(), {"A"})
    assert router.has_unlabeled_continuation(1)
    selected, score = router.select_connection(1, 2, np.random.default_rng(0))
    assert (selected.connection_id, score) == (1, 3)
    assert list(router.predictions[-1]["candidates"]) == [1, "wait"]


def test_nominal_tail_obeys_frozen_menu_after_the_first_leg():
    graph = network([("A", "A", 0, 1, 0, 2), ("B", "B", 0, 2, 0, 18),
                     ("X", "X", 1, 2, 21, 22), ("C", "C", 1, 2, 4, 29)])
    nominal = {c.id: NominalEvent(0, c.dep_time, c.arr_time, 0.) for c in graph.connections}
    nominal[2] = NominalEvent(-5, 16, 17, 0.)
    state = CommittedState(0, 0, None, 0)
    assert nominal_action(graph, state, 2, nominal, {0, 1, 3}, 20, 25) == ("select", 1)


def test_nominal_tail_waits_until_boarding_fits_commitment():
    graph = network([("A", "A", 0, 2, 22, 23), ("B", "B", 0, 2, 0, 25)])
    nominal = {c.id: NominalEvent(0, c.dep_time, c.arr_time, 0.) for c in graph.connections}
    assert nominal_action(graph, CommittedState(0, 0, None, 0), 2, nominal, {0, 1}, 30, 25) == WAIT
    assert nominal_action(graph, CommittedState(0, 10, None, 0), 2, nominal,
                          {0, 1}, 30, 25) == ("select", 0)


def test_rollout_uses_updated_route_parameters_without_explicit_prior():
    graph = network([("A", "A", 0, 2, 20, 21)])
    router = ContinuationRouter(graph, {})
    router.route(0, 2, 0)
    router.observe_delay("A", 20)
    assert router._parameters()["A"]["mean"] == router.route_beliefs["A"].posterior_mean
    assert router._parameters()["A"]["mean"] > 1


def test_first_failure_probability_uses_posterior_when_bank_misses_branch(monkeypatch):
    graph = network([("A", "A", 0, 2, 0, 4), ("B", "B", 0, 2, 2, 6)])
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                             for c in graph.connections})
    monkeypatch.setattr(TripBelief, "sample", lambda self, params, rng, n: [scene] * n)
    router = ContinuationRouter(graph, priors(graph), samples=2, max_time=20)
    router.route(0, 2, 0)
    router.set_execution_context(None, 0, set())
    router.set_information_context(0, 0, None, 0, set(), set(), set())
    router.select_connection(0, 0, np.random.default_rng(0))
    estimate = router.predictions[-1]["candidates"][0]
    fail = estimate["first_attempt_failure_probability"]
    assert estimate["raw_sample_first_failure_probability"] == 0
    assert 0 < fail < 1
    assert estimate["samples"] == 3
    assert estimate["conditional_first_failure_loss"] is not None
    expected = ((1 - fail) * estimate["conditional_first_success_loss"]
                + fail * estimate["conditional_first_failure_loss"])
    assert estimate["score"] == pytest.approx(expected)


def test_unresolvable_boarding_mass_does_not_take_a_candidate_slot():
    graph = network([("A", "A", 0, 2, 0, 4), ("B", "B", 0, 2, 58, 60)])
    router = ContinuationRouter(graph, {}, samples=2, candidate_limit=1, max_time=75)
    router.route(0, 2, 0)
    router.observe_vehicle_event(1, False, 0)
    router.set_execution_context(None, 0, set())
    router.set_information_context(0, 55, None, 0, set(), set(), set())
    state = router.information_state
    mass = router._board_probability(0, state, router._parameters())
    assert 0 < mass < np.finfo(float).eps / 4
    selected, _ = router.select_connection(0, 55, np.random.default_rng(0))
    assert selected.connection_id == 1
    assert list(router.predictions[-1]["candidates"]) == [1, "wait"]


def test_first_failure_cap_keeps_rescue_after_a_successful_first_boarding(monkeypatch):
    graph = network([("A", "A", 0, 1, 0, 2), ("B", "B", 1, 2, 5, 8),
                     ("C", "C", 1, 2, 7, 12)])
    scene = JourneyScenario({0: VehicleEvent(0, False, 0, 2),
                             1: VehicleEvent(999, True, 5, 8),
                             2: VehicleEvent(0, False, 7, 12)})
    monkeypatch.setattr(TripBelief, "sample", lambda self, params, rng, n: [scene] * n)
    router = ContinuationRouter(graph, priors(graph), rescue=False, samples=2, max_time=20)
    router.route(0, 2, 0)
    router.observe_vehicle_event(0, False, 0)
    router.set_execution_context(None, 0, set())
    router.set_information_context(0, 0, None, 0, set(), set(), set())
    router.select_connection(0, 0, np.random.default_rng(0))
    estimate = router.predictions[-1]["candidates"][0]
    assert estimate["first_attempt_failure_probability"] == 0
    assert estimate["conditional_first_success_loss"] == 12
    assert estimate["score"] == 12


def test_public_identity_hooks_leave_legacy_actions_and_cost_unchanged():
    import copy
    graph = network([("A", "A", 0, 2, 1, 4), ("B", "B", 0, 2, 2, 6)])
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                             for c in graph.connections})
    results = []
    for legacy in (False, True):
        g = copy.deepcopy(graph)
        router = (ContinuationRouter(g, priors(g), legacy=True, max_time=20) if legacy else
                  MatchedRouter(g, priors(g), online_mean=True, online_cancel=True, max_time=20))
        result = simulate_bandit_journey(g, router, 0, 2, 0, RegimeSchedule([(0, "x")]),
                                        np.random.default_rng(0), max_time=20, scenario=scene)
        results.append(result)
    assert results[0].events == results[1].events
    assert results[0].observations == results[1].observations
