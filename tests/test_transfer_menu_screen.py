import copy
import gzip
import json

from experiments.reconstruction_v24.screen import (
    DATES, build_menu, independent_menu, select_pair, static_graphs,
)
from src.recourse_itinerary_router import build_trip_blocks, trip_key
from src.transit_graph import Connection, Stop, TransitGraph


def graph_with(rows):
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), 0., 0., 10))
    for cid, (trip, origin, goal, depart, arrive) in enumerate(rows):
        graph.add_connection(Connection(cid, trip, trip, origin, goal, depart, arrive))
    return graph


def option(cid, arrival, transfer=None, tails=(), ready=None):
    return dict(first_cid=cid, first_path=(cid,), transfer_stop=transfer,
                tail_paths=tails, scheduled_arrival=arrival, scheduled_transfer_ready=ready)


def test_schedule_graph_is_unchanged_when_actual_endpoints_change(tmp_path):
    stops = {str(i): dict(name=str(i), latitude=0., longitude=0.) for i in range(2)}
    connection = dict(dropoff_allowed=True, pickup_allowed=True, route_id='r', trip_id='t',
                      dep_stop='0', arr_stop='1', scheduled_departure_seconds=10,
                      scheduled_arrival_seconds=20, outcome='MEASURED',
                      departure=dict(time_seconds=13), arrival=dict(time_seconds=24))
    archive = dict(dates={day: dict(stops=stops, connections=[connection]) for day in DATES})
    paths = [tmp_path/'a.gz', tmp_path/'b.gz']
    with gzip.open(paths[0], 'wt') as handle:
        json.dump(archive, handle)
    changed = copy.deepcopy(archive)
    for day in DATES:
        row = changed['dates'][day]['connections'][0]
        row.update(outcome='UNRESOLVED', departure=None, arrival=None)
    with gzip.open(paths[1], 'wt') as handle:
        json.dump(changed, handle)
    left, right = (static_graphs(path) for path in paths)
    for day in DATES:
        assert vars(left[day].connections[0]) == vars(right[day].connections[0])


def test_full_menu_keeps_direct_control_when_transfer_pair_omits_it():
    graph = graph_with([('a', 0, 2, 10, 90), ('b', 0, 2, 20, 50), ('c', 0, 1, 30, 40)])
    full = [option(0, 90), option(1, 50), option(2, 70, 1, ((3,), (4,)), 50)]
    pair, reason = select_pair(graph, full)
    assert [o['first_cid'] for o in pair] == [0, 2]
    assert [o['first_cid'] for o in full] == [0, 1, 2]
    assert reason == 'earliest_first_departure_and_best_alternative_with_transfer'


def test_root_and_prefix_trip_overlap_removal_requires_a_rescue():
    graph = graph_with([('a', 0, 1, 10, 20), ('b', 0, 2, 20, 70),
                        ('b', 1, 2, 40, 70), ('incoming', 1, 2, 50, 80),
                        ('d', 1, 2, 60, 90), ('e', 0, 1, 30, 40),
                        ('f', 1, 2, 70, 100), ('g', 1, 2, 80, 110)])
    catalog = [option(0, 70, 1, ((2,), (3,), (4,)), 30), option(1, 70),
               option(5, 90, 1, ((4,), (6,), (7,)), 50)]
    filtered, counts = independent_menu(graph, catalog, trip_key(graph.connections[3]))
    assert [o['first_cid'] for o in filtered] == [1, 5]
    assert counts['removed_overlapping_tail_paths'] == 2
    assert counts['excluded_options_without_independent_rescue'] == 1
    assert filtered[1]['tail_paths'] == ((4,), (6,), (7,))


def test_menu_requires_two_nominally_catchable_tail_trips():
    graph = graph_with([('a', 0, 1, 10, 20), ('b', 0, 2, 30, 45),
                        ('c', 1, 2, 40, 60), ('d', 1, 2, 60, 80),
                        ('e', 0, 3, 20, 30), ('f', 3, 2, 50, 70)])
    menu, counts = build_menu(graph, build_trip_blocks(graph), 0, 2, 0, 0, 100)
    assert {o['first_cid'] for o in menu} == {0, 1}
    transfer = next(o for o in menu if o['first_cid'] == 0)
    assert transfer['tail_paths'] == ((2,), (3,))
    assert transfer['scheduled_transfer_ready'] == 30
    assert counts['transfer_options'] == counts['direct_options'] == 1
