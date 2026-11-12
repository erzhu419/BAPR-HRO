import pytest

from src.conditional_rescue_policy import ConditionalItineraries
from src.evolving_trip_belief import EvolvingTripBelief
from src.platform_transit import PlatformState
from src.public_rescue_candidates import PublicItinerary
from src.public_wait_recourse import compile_through_scores
from src.tail_rescue import TailRescuePlan
from src.transit_graph import Connection, Stop, TransitGraph


def test_through_controls_preserve_wait_reference_and_reproduce_conditional_scores():
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=2))
    for trip, origin, target, departure, arrival in (
        ('A', 0, 3, 24, 49), ('B', 0, 3, 28, 55),
        ('C', 0, 1, 20, 30), ('C', 1, 2, 32, 40), ('C', 2, 3, 42, 50),
        ('D', 0, 2, 23, 36), ('E', 2, 3, 43, 48),
    ):
        graph.add_connection(Connection(-1, 'R', trip, origin, target, departure, arrival))
    parameters = {'R': {'mean': 0., 'std': 3., 'cancel_rate': .25}}
    belief = EvolvingTripBelief(graph, minute=1, public_occurrences=False)
    root = PlatformState(0, 12, last_arrival=12, local_since=12)
    deadline = 52
    ab = {cid: TailRescuePlan((cid,), first_action_now=True) for cid in (0, 1)}
    extras = {2: PublicItinerary((2, 3, 4), 50), 5: PublicItinerary((5, 6), 48)}
    paths = ab | {cid: TailRescuePlan(option.path, first_action_now=True) for cid, option in extras.items()}
    integrated = ConditionalItineraries(belief, parameters, graph, root, 3, deadline, paths)
    point = ConditionalItineraries(belief, parameters, graph, root, 3, deadline, paths, point_forecast=True)
    exact_control, exact_menu = compile_through_scores(belief, parameters, graph, root, 3, deadline,
                                                       ab, extras)
    point_control, point_menu = compile_through_scores(belief, parameters, graph, root, 3, deadline,
                                                       ab, extras, point=True)
    assert set(exact_menu) == set(point_menu) == {2}
    assert set(exact_control.rows) == set(point_control.rows) == {0, 1, 2}
    for cid in ab:
        assert exact_control.rows[cid] == point_control.rows[cid] == integrated.rows[cid]
    for clock, exact in exact_control.rows[2].items():
        original = integrated.rows[2][clock]
        assert exact.expected_arrival_or_deadline == pytest.approx(original.expected_arrival_or_deadline)
        assert exact.success_probability == pytest.approx(original.success_probability)
        assert point_control.rows[2][clock].expected_arrival_or_deadline == (
            point.rows[2][clock].expected_arrival_or_deadline)
    assert any(0. < row.success_probability < 1. for row in exact_control.rows[2].values())
