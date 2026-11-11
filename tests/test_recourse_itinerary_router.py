from unittest.mock import patch

import numpy as np

from src.committed_transit import CommittedState
from src.evolving_trip_belief import EvolvingTripBelief
from src.recourse_itinerary_router import RecourseItineraryRouter, expand_trip_blocks
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import JourneyScenario, RegimeSchedule, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def network():
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=2))
    for row in [('R', 'A', 0, 1, 5, 10), ('R', 'A', 1, 2, 10, 15)]:
        graph.add_connection(Connection(-1, *row))
    return graph


def router(graph, optimize=False):
    belief = EvolvingTripBelief(graph, public_occurrences=False, travel_sd=.001, dwell_sd=.001)
    result = RecourseItineraryRouter(graph, {'R': dict(mean=0., std=.5, cancel_rate=0.)},
        optimize=optimize, event_belief=belief, max_time=200, max_itineraries=4)
    result.online_mean = result.online_cancel = False
    return result


def test_preissued_stay_has_no_new_planning_latency_at_zero_dwell():
    graph = network()
    policy = router(graph)
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                             for c in graph.connections}, minute=60, publications={})
    with patch('src.simulate_bandit.perf_counter', side_effect=[0., 2., 2.1, 4.1, 4.2]):
        result = simulate_bandit_journey(graph, policy, 0, 2, 0, RegimeSchedule([(0, '')]),
            np.random.default_rng(0), max_time=200, scenario=scene, charge_computation=True)
    assert result.arrival_time == 15
    computations = [event for event in result.events if event.event_type == 'compute']
    assert [event.time for event in computations] == [2, 10]
    assert 'preissued_stay=True' in computations[-1].details
    assert len(policy.predictions) == 2 and policy.predictions[-1]['compiled']


def test_newly_computed_first_action_cannot_claim_to_be_preissued():
    graph = network()
    policy = router(graph)
    policy.route(0, 2, 0)
    policy.set_information_context(0, 0, None, 0, set(), set(), set())
    assert policy.precommitted_connection(0, 0, None) is None
    policy.select_connection(0, 0, np.random.default_rng(0))
    assert policy.precommitted_connection(0, 0, None) is None


def test_exact_router_has_complete_model_bounds_and_legal_compiled_action():
    graph = network()
    policy = router(graph, optimize=True)
    policy.route(0, 2, 0)
    policy.set_information_context(0, 0, None, 0, set(), set(), set())
    selected = policy.select_connection(0, 0, np.random.default_rng(0))
    result = policy.predictions[-1]
    assert selected is not None and selected[0].connection_id == 0
    assert result['full_model_lower'] <= result['value'] <= 200
    assert result['full_model_gap'] >= 0
    assert result['work'] == 0
    assert result['analytic_work']['law_queries'] > 0
    assert result['itinerary_evaluations'] <= 4


def test_expansion_retains_same_trip_intermediate_loops_and_rejects_trip_revisit():
    graph = network()
    graph.add_connection(Connection(-1, 'R', 'A', 1, 1, 10, 10))
    # The loop's event order precedes the final leg by arrival-time tie break.
    assert expand_trip_blocks(graph, (0, 1), 2) == (0, 2, 1)
    graph.add_connection(Connection(-1, 'R', 'B', 1, 1, 10, 10))
    assert expand_trip_blocks(graph, (0, 3, 1), 2) is None
