"""Selection must retain real branch structure without inspecting outcomes."""

from experiments.reconstruction_v32.screen import (
    PROTOCOL, departures, retain_options, structure_roots,
)
from src.transit_graph import Connection, Stop, TransitGraph


def tiny_network():
    graph = TransitGraph()
    for stop in range(7):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=20))
    rows = [
        ('R', 'root', 0, 1, 100, 200),
        ('R', 'root', 1, 2, 210, 300),
        ('R', 'root', 2, 5, 310, 400),
        ('X', 'x', 1, 3, 240, 280),
        ('X', 'x', 3, 4, 290, 340),
        ('Y', 'shared', 1, 2, 260, 320),
        ('Y', 'shared', 2, 4, 340, 380),
        ('Z', 'z', 2, 6, 350, 390),
        ('Q', 'q', 6, 4, 430, 480),
    ]
    for route, trip, a, b, dep, arr in rows:
        graph.add_connection(Connection(len(graph.connections), route, trip, a, b, dep, arr))
    return graph


def test_distinct_hubs_shared_trip_and_second_transfer_survive_selection():
    graph = tiny_network()
    roots, _, _ = structure_roots(graph, 'example', 0)
    root = next(root for root in roots if root['goal'] == 4)
    assert {h['stop'] for h in root['hubs']} == {1, 2}
    assert root['shared_tail_trips']['shared'] == [1, 2]
    assert root['has_downstream_only_direct_tail']
    assert root['has_two_transfer_option']
    assert not root['selection_uses_actual']
    two = next(h for h in root['hubs'] if h['stop'] == 2)
    assert any(t['trip_ids'] == ['z', 'q'] for t in two['tails'])
    assert all(graph.connections[p[0]].trip_id == trip
               for h in root['hubs'] for t in h['tails']
               for p, trip in zip(t['rides'], t['trip_ids']))


def test_retaining_a_first_vehicle_keeps_all_alighting_variants():
    graph = tiny_network()
    a = dict(first_cid=3, first_path=(3,), tail_paths=[(8,)], scheduled_arrival=480)
    b = dict(first_cid=3, first_path=(3, 4), tail_paths=[], scheduled_arrival=340)
    c = dict(first_cid=5, first_path=(5, 6), tail_paths=[], scheduled_arrival=380)
    kept = retain_options(graph, [a, b, c], 1)
    assert kept == [a, b]


def test_static_departures_respect_mct_pickup_and_same_line_cap():
    graph = tiny_network()
    graph.add_connection(Connection(9, 'Y', 'late-y', 1, 4, 270, 400))
    graph.add_connection(Connection(10, 'Y', 'later-y', 1, 4, 280, 410))
    graph.add_connection(Connection(11, 'N', 'no-pickup', 1, 4, 290, 420,
                                    pickup_allowed=False))
    protocol = dict(PROTOCOL, services_per_route=1)
    selected = departures(graph, 1, 250, 500, protocol=protocol)
    assert selected == [5]


def test_single_repeated_final_service_is_not_a_distinct_opportunity():
    graph = tiny_network()
    # Only the onboard vehicle and the shared downstream vehicle remain.
    reduced = TransitGraph(stops=graph.stops.copy())
    for cid in [0, 1, 2, 5, 6]:
        c = graph.connections[cid]
        reduced.add_connection(Connection(len(reduced.connections), c.route, c.trip_id,
            c.dep_stop, c.arr_stop, c.dep_time, c.arr_time))
    roots, exclusions, _ = structure_roots(reduced, 'example', 0)
    assert not any(root['goal'] == 4 for root in roots)
    assert exclusions['equivalent_final_services'] > 0


def test_first_vehicle_direct_destination_is_not_hidden_by_leg_budget():
    graph = tiny_network()
    graph.add_connection(Connection(9, 'R', 'root', 5, 4, 410, 500))
    roots, _, _ = structure_roots(graph, 'example', 0, dict(PROTOCOL, max_first_legs=2))
    assert not any(root['goal'] == 4 for root in roots)
