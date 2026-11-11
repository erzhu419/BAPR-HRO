import numpy as np

from src.evolving_trip_belief import EvolvingTripBelief
from src.native_event_router import NativeEventRouter, minute_snapshot
from src.transit_graph import Connection, Stop, TransitGraph


def network():
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=120))
    graph.add_connection(Connection(-1, 'R', 'A', 0, 1, 61, 121))
    graph.add_connection(Connection(-1, 'R', 'A', 1, 2, 121, 181))
    return graph


def test_minute_projection_keeps_candidates_and_separates_cancel_once():
    graph = network()
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False)
    pars = {'R': dict(mean=0., std=.5, cancel_rate=.2)}
    temp, work = minute_snapshot(belief, pars)
    assert len(temp.connections) == len(graph.connections)
    assert graph.connections[0].dep_time == 61
    assert graph.connections[0].dep_distribution is None
    for c in temp.connections:
        assert abs(c.dep_distribution.total_prob - 1.) < 1e-12
        assert abs(c.arr_distribution.total_prob - 1.) < 1e-12
        assert 0 < c.cancel_prob < 1
    assert work['max_clock_quantization_seconds'] <= 30
    assert work['posterior_events'] == 4


def policy(objective='capped'):
    graph = network()
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False)
    for c in graph.connections:
        for kind in ('departure', 'arrival'):
            belief.observe_connection_event(c.id, kind, 0)
    result = NativeEventRouter(graph, {'R': dict(mean=0., std=.5, cancel_rate=0.)},
        event_belief=belief, menu='full', samples=4, algorithm='uct', work_budget=64,
        max_time=600, native_objective=objective)
    result.online_mean = result.online_cancel = False
    result._initial_cancels = {'R': 0.}
    result.route(0, 2, 0)
    return result


def test_same_trip_native_continuation_uses_original_seconds_execution():
    p = policy()
    p.set_information_context(0, 0, None, 0, set(), set(), set())
    selected = p.select_connection(0, 0, np.random.default_rng(0))
    assert selected[0].connection_id == 0
    p.set_information_context(1, 121, 'A', 121, {0}, set(), {'A'})
    selected = p.select_connection(1, 121, np.random.default_rng(0))
    assert selected[0].connection_id == 1
    assert p.graph.connections[1].dep_time == 121
    assert p.predictions[-1]['native_work']['native_scans'] == 1


def test_physical_root_transfer_and_no_pickup_are_respected():
    p = policy('conditional_mean')
    p.graph.connections[1].pickup_allowed = False
    p.set_information_context(1, 121, 'OTHER', 121, {0}, set(), set())
    assert p.select_connection(1, 121, np.random.default_rng(0)) is None
    p.graph.connections[1].pickup_allowed = True
    assert p.select_connection(1, 121, np.random.default_rng(0)) is None
    p.set_information_context(1, 121, 'A', 121, {0}, set(), {'A'})
    assert p.select_connection(1, 121, np.random.default_rng(0))[0].connection_id == 1
