from src.evolving_trip_belief import EvolvingTripBelief
from src.platform_transit import PlatformState
from src.support_rescue_candidates import screen_support_itineraries
from src.transit_graph import Connection, Stop, TransitGraph


def test_model_supported_early_departure_extends_the_timetable_menu():
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=60))
    for route, trip, origin, dest, dep, arr, pickup in (
        ('early', 'r', 0, 1, 0, 60, True),
        ('early', 'r', 1, 2, 60, 120, False),
        ('existing', 'existing', 0, 2, 180, 240, True),
        ('excluded', 'excluded', 0, 2, 0, 30, False),
        ('outside', 'outside', 0, 2, -10000, -9940, True),
        ('previous', 'incoming', 3, 0, -30, 30, True),
    ):
        graph.add_connection(Connection(0, route, trip, origin, dest, dep, arr,
                                        pickup_allowed=pickup))
    root = PlatformState(0, 30, None, 30, 5, frozenset({5}), local_since=30)
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False)
    params = {c.route: dict(mean=2., std=.5, cancel_rate=.01) for c in graph.connections}
    existing = {2}
    added, audit = screen_support_itineraries(graph, belief, params, root, 2, 1000,
                                               existing, min_mass=.01)
    assert existing == {2}
    assert set(added) == {0}
    assert added[0].path == (0, 1)
    assert audit['ready'] == 90
    assert audit['rows'][0]['scheduled_departure'] < audit['ready']
    assert audit['rows'][0]['representative_departure'] >= audit['ready']
    assert audit['rows'][0]['operating_mass'] > .01
