from unittest.mock import patch

from src.committed_transit import CommittedState
from src.evolving_trip_belief import EvolvingTripBelief
from src.latency_recourse_router import LatencyRecourseRouter
from src.recourse_itinerary_router import CompiledPolicy
from src.risk_bounds import CommitmentBounds
from src.transit_graph import Connection, Stop, TransitGraph


def test_dispatch_revalues_same_routes_after_their_departure_opportunity_changes():
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'origin'))
    graph.add_stop(Stop(1, 'goal'))
    graph.add_connection(Connection(0, 'R', 'A', 0, 1, 1, 2))
    graph.add_connection(Connection(1, 'R', 'B', 0, 1, 4, 6))
    parameters = {'R': dict(mean=0., std=.5, cancel_rate=0.)}
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False, travel_sd=.001)
    policy = LatencyRecourseRouter(graph, parameters, event_belief=belief, max_time=10)
    policy.route(0, 1, 0)
    state = CommittedState(0, 0, None, 0, initialized=True)
    bounds = CommitmentBounds(belief, parameters, graph, 1, 10)
    candidates = [CompiledPolicy((cid,), (), bounds.itinerary(state, (cid,)).upper, ('select', cid))
                  for cid in (0, 1)]
    assert min(candidates, key=lambda p: p.upper).root_action == ('select', 0)
    with patch('src.latency_recourse_router.perf_counter', return_value=120.):
        chosen, metadata = policy._dispatch_policy(candidates[0], candidates, bounds, state, 0.)
    assert metadata['estimated_execution_clock'] == 2
    assert chosen.root_action == ('select', 1)
    assert metadata['bounds_scope'] == 'estimated_execution_clock'
    assert metadata['dispatch_lower'] <= chosen.upper
