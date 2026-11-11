from src.committed_transit import CommittedState
from src.continuation_router import nominal_action
from src.evolving_trip_belief import EvolvingTripBelief
from src.indexed_nominal_tail import NominalDepartureIndex
from src.transit_graph import Connection, Stop, TransitGraph


def test_changed_trip_can_reenter_an_index_after_observed_departure():
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    graph.add_connection(Connection(-1, 'R', 'A', 0, 1, 120, 180))
    graph.add_connection(Connection(-1, 'R', 'A', 1, 2, 240, 300))
    graph.add_connection(Connection(-1, 'R', 'B', 0, 2, 510, 570))
    parameters = {'R': {'mean': 0, 'std': 0.5, 'cancel_rate': 0}}
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False,
                               travel_sd=0.1, dwell_sd=0.1)
    index = NominalDepartureIndex(belief, parameters, 800)
    state = CommittedState(0, 300, None, 300, initialized=True)
    original = index.branch(belief)
    assert [c.id for c in original.candidates(0, 300)] == [2]
    changed = belief.copy()
    changed.observe_connection_event(0, 'departure', 240)
    branch = index.branch(changed)
    assert [c.id for c in branch.candidates(0, 300)] == [0, 2]
    kwargs = dict(menu={0, 1, 2}, deadline=800, lookahead=1500,
                  root_boardings={0, 2}, minute=60)
    assert nominal_action(graph, state, 2, changed.mean_events(parameters), **kwargs) == ('select', 0)
    assert nominal_action(graph, state, 2, branch, connection_candidates=branch.candidates,
                          **kwargs) == ('select', 0)
    assert [c.id for c in original.candidates(0, 300)] == [2]
