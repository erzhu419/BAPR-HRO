"""Physical effects of measured computation and deferred fixed-world access."""
from collections.abc import Mapping
from unittest.mock import patch

import numpy as np
import pytest

from experiments.reconstruction_v4.run_dated_replay import ScheduledClock
from src.causal_planning_router import CausalPlanningRouter
from src.committed_transit import CommittedState, build_committed_model
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import JourneyScenario, LazyTripScenario, RegimeSchedule, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def network():
    graph = TransitGraph()
    for sid in range(4):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=0))
    graph.add_connection(Connection(0, 'A', 'a', 0, 1, 1, 5))
    graph.add_connection(Connection(1, 'B', 'b', 2, 3, 1, 5))
    return graph


@pytest.mark.parametrize('charged,expected', [(False, 'arrived'), (True, 'timeout')])
def test_computation_can_use_up_the_selected_departure_slack(charged, expected):
    graph = network()
    priors = {r: dict(mean=0., std=1., cancel_rate=0.) for r in ('A', 'B')}
    router = CausalPlanningRouter(graph, priors, algorithm='nominal', max_time=10,
                                  event_belief=ScheduledClock(graph))
    events = {c.id: VehicleEvent(0, False, c.dep_time, c.arr_time) for c in graph.connections}
    with patch('src.simulate_bandit.perf_counter', side_effect=[0., 2., 2.1]):
        result = simulate_bandit_journey(graph, router, 0, 1, 0, RegimeSchedule([(0, '')]),
                                          np.random.default_rng(1), max_time=10,
                                          scenario=JourneyScenario(events, minute=60, publications={}),
                                          charge_computation=charged)
    assert result.events[-1].details == expected
    assert [e.time for e in result.events if e.event_type == 'compute'] == ([2] if charged else [])


def test_lazy_world_build_and_silent_publication_read_only_the_selected_trip():
    class FixedDraws(Mapping):
        def __init__(self):
            self.reads = []

        def __getitem__(self, cid):
            self.reads.append(cid)
            return VehicleEvent(0, False, 1, 5)

        def __iter__(self):
            return iter((0, 1))

        def __len__(self):
            return 2

    graph = network()
    draws = FixedDraws()
    scene = LazyTripScenario(draws, minute=60, publications={})
    model = build_committed_model(graph, [(1., scene)], 0, 1, 0, 20,
                                  initial_state=CommittedState(0, 0, None, 0, initialized=True))
    assert not draws.reads
    outcome = model.transition(model.scenarios[0], model.initial_state, ('select', 0))
    assert outcome.state.stop == 1
    assert draws.reads == [0]
    with pytest.raises(TypeError):
        scene.events[0] = VehicleEvent(0, False, 1, 5)
    bad = JourneyScenario({0: VehicleEvent(0, False, 5, 1), 1: VehicleEvent(0, False, 1, 5)})
    with pytest.raises(ValueError, match='arrival'):
        build_committed_model(graph, [(1., bad)], 0, 1, 0, 20)
