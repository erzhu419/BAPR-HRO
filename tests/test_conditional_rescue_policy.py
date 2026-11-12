from dataclasses import replace

import pytest

from src.conditional_rescue_policy import ConditionalItineraries
from src.evolving_trip_belief import EvolvingTripBelief
from src.platform_transit import PlatformState
from src.public_trip_conditionals import condition_on_departure
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
