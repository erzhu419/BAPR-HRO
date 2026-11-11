from fractions import Fraction

import pytest

from src.committed_transit import CommittedState, build_committed_model
from src.information_planner import ExactPlanner
from src.sparse_certificate import SparseCertificateIndex as FullIndex
from src.sparse_certificate_cutoff import SparseCertificateIndex
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import TransitGraph, Stop, Connection


@pytest.mark.parametrize('cutoff', [None, Fraction(7, 2), Fraction(25, 4), 0])
def test_cutoff_never_exceeds_full_lower_or_same_information_q(cutoff):
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'origin'))
    graph.add_stop(Stop(1, 'goal'))
    for cid, dep, arr in [(0, 1, 4), (1, 1, 8), (2, 0, 2)]:
        graph.add_connection(Connection(cid, str(cid), str(cid), 0, 1, dep, arr))
    events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections}
    bank = [(Fraction(1, 2), JourneyScenario(events, publications={})),
            (Fraction(1, 2), JourneyScenario(dict(events) | {0: VehicleEvent(1, False, 2, 5)}, publications={}))]
    state = CommittedState(0, 1, None, 1, initialized=True)
    model = build_committed_model(graph, bank, 0, 1, 1, 10, initial_state=state)
    optimum = ExactPlanner(model).solve()
    full = FullIndex(graph, bank, 1, 1, 10).action_lower(state, model.actions(state))
    index = SparseCertificateIndex(graph, bank, 1, 1, 10)
    bounded = index.action_lower(state, model.actions(state), cutoff=cutoff)
    assert all(bounded[a] <= full[a] <= Fraction(optimum.action_values[a]) for a in bounded)
    if cutoff is None:
        assert bounded == full
    else:
        assert index.stats['ea_skipped_by_cutoff'] > 0
