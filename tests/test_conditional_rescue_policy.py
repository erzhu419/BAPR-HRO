from dataclasses import replace

import pytest

from src.background_policy import execute_background
from src.conditional_rescue_policy import ConditionalItineraries
from src.evolving_trip_belief import EvolvingTripBelief
from src.offgrid_conditional import ActualClockRescue, ActualClockScores
from src.platform_transit import PlatformState, build_platform_model
from src.public_trip_conditionals import condition_on_departure
from src.simulator import JourneyScenario, VehicleEvent
from src.tail_rescue import TailRescuePlan
from src.transit_graph import Connection, Stop, TransitGraph


def test_visible_departure_score_integrates_random_arrival_and_deadline():
    graph = TransitGraph()
    graph.add_stop(Stop(0,'origin',min_transfer_time=0))
    graph.add_stop(Stop(1,'goal',min_transfer_time=0))
    graph.add_connection(Connection(0,'R','trip',0,1,20,28))
    params = {'R':{'mean':0.,'std':2.,'cancel_rate':.2}}
    belief = EvolvingTripBelief(graph,minute=1,public_occurrences=False)
    initial = PlatformState(0,10,local_since=10)
    deadline = 29
    table = ConditionalItineraries(belief,params,graph,initial,1,deadline,{0:TailRescuePlan((0,))})
    row = table.rows[0][20]
    arrival = condition_on_departure(belief,params,0,20).marginal(0)
    capped_gain = sum(max(0,deadline-clock)*p for clock,p in zip(arrival.clocks,arrival.probabilities))
    assert row.expected_arrival_or_deadline == pytest.approx(deadline-capped_gain)
    assert row.next_arrival_support > 1
    state = replace(initial,time=20,public_log=((20,0,'departure'),))
    assert table.future(0,state,21,deadline)==()


@pytest.mark.parametrize('visible_cid', [0, 2])
@pytest.mark.parametrize('departure', [600, 1500])
def test_actual_clock_boards_and_follows_a_visible_original_path(visible_cid, departure):
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=0))
    for cid, trip in ((0, 'A'), (2, 'B')):
        graph.add_connection(Connection(cid, 'R', trip, 0, 1, departure, departure + 10))
        graph.add_connection(Connection(cid + 1, 'R', trip, 1, 2, departure + 11, departure + 30))
    paths = {cid: TailRescuePlan((cid, cid + 1), first_action_now=True) for cid in (0, 2)}
    plan = TailRescuePlan(paths[0].primary, 0, paths[2].primary, True, ((0, 700), (2, 700)))
    root, deadline = PlatformState(0, 0), 2000
    params = {'R': {'mean': 0., 'std': 0., 'cancel_rate': 0.}}
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    scores = ConditionalItineraries(belief, params, graph, root, 2, deadline, paths)
    controller = ActualClockRescue(graph, plan, root, deadline, scores, {}, 0,
                                   ActualClockScores(belief, params, scores, 2), mode='projection')
    events = {cid: VehicleEvent(0, False, graph.connections[cid].dep_time,
                               graph.connections[cid].arr_time) for cid in (visible_cid, visible_cid + 1)}
    world = JourneyScenario(events, minute=1, publications={})
    model = build_platform_model(graph, ((1., world),), 0, 2, 0, deadline, initial_state=root)

    expired = departure > 720
    issued = controller.original.bind('issued')
    assert execute_background(model, model.scenarios[0], issued.policy, issued.policy, 0)['cost'] == (
        deadline if expired else departure + 30)

    actual = controller.bind('actual')
    result = execute_background(model, model.scenarios[0], actual.policy, actual.policy, 0)
    assert result['cost'] == departure + 30
    assert result['state'].used_connections == frozenset((visible_cid, visible_cid + 1))
    assert actual.stats['expiries'] == (2 if expired else 0)
    assert actual.stats['conditional_boardings'] == int(expired)
