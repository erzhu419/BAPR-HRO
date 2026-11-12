from src.background_policy import execute_background
from src.first_opportunity_rescue import FirstOpportunityRescue
from src.platform_transit import PlatformState, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.tail_rescue import TailRescuePlan
from src.transit_graph import Connection, Stop, TransitGraph


def test_first_opportunity_obeys_readiness_and_ranks_simultaneous_rides():
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=2))
    for trip, start, end, departure, arrival, pickup in (
        ('early', 0, 2, 9, 13, True),
        ('long', 0, 2, 10, 20, True),
        ('short', 0, 1, 10, 13, True),
        ('short', 1, 2, 13, 16, False),
        ('tie', 0, 2, 10, 16, True),
        ('incoming', 3, 0, 5, 8, True),
    ):
        graph.add_connection(Connection(0, trip, trip, start, end, departure, arrival,
                                        pickup_allowed=pickup))
    root = PlatformState(0, 8, last_arrival=8, last_connection=5,
                         used_connections=frozenset({5}), local_since=8)
    plan = TailRescuePlan((1,), 0, (0,), departures=((1, 10),))
    paths = {0: (0,), 1: (1,), 2: (2, 3), 4: (4,)}
    policy = FirstOpportunityRescue(graph, plan, root, 2, 40, paths).bind('minimal')
    scene = JourneyScenario({c.id: VehicleEvent(0, False, c.dep_time, c.arr_time)
                             for c in graph.connections})
    model = build_platform_model(graph, [(1., scene)], 0, 2, 8, 40, initial_state=root, poll_interval=1)
    outcome = execute_background(model, model.scenarios[0], policy.policy, policy.policy, 8)
    rides = [step['ridden_connection'] for step in outcome['trace'] if step['ridden_connection'] is not None]
    assert (outcome['cost'], rides) == (8, [2, 3])
    assert policy.trace[0]['offers'] == (1, 2, 4)
