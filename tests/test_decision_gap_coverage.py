"""Uniform upstream coverage retains old actions without inspecting outcomes."""
from types import SimpleNamespace

from experiments.decision_gap_v1.candidate_coverage import (
    contains_menu, endpoint_coverage, expand_menu,
)
from experiments.reconstruction_v33.nodes import Platform, Ride, root_ride
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v37.controller import node_structure
from experiments.reconstruction_v50.execute import ObservationContract
from src.transit_graph import Connection, Stop, TransitGraph


def case():
    graph = TransitGraph()
    for stop in range(5):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=5))
    for row in (
        (0,'R',0,1,10,20), (1,'R',1,2,25,35),
        (2,'T',1,3,15,25), (3,'T',3,2,26,30), (4,'T',2,4,40,50),
        (5,'S',1,4,30,80),
    ):
        cid, trip, dep, arr, td, ta = row
        graph.add_connection(Connection(cid, trip, trip, dep, arr, td, ta,
                                        pickup_allowed=cid != 3))
    first = Platform(1, {5:Ride(5, (5,), {5:None})})
    later = Platform(2, {4:Ride(4, (4,), {4:None})})
    node = Ride(0, (0,1), {0:first, 1:later})
    return graph, node


def trip_model(graph):
    return SimpleNamespace(_positions={(c.id, kind):(c.trip_id, 2*c.id+i)
                           for c in graph.connections for i, kind in enumerate(('departure','arrival'))})


def depth(node):
    if node is None:
        return 0
    if isinstance(node, Ride):
        return 1+max((depth(p) for p in node.exits.values()), default=0)
    return max((depth(r) for r in node.rides.values()), default=0)


def test_uniform_rule_restores_scheduled_early_prefix_and_preserves_original():
    graph, old = case()
    before = node_structure(old)
    new, ledger = expand_menu(graph, old, 4)
    assert node_structure(old) == before
    assert contains_menu(new, old)
    assert new.exits[0].rides[2].path == (2,3,4)
    assert new.exits[0].rides[2].exits == {4:None}
    assert graph.connections[2].dep_time < graph.connections[0].arr_time+5
    assert not graph.connections[3].pickup_allowed  # Legal while staying aboard.
    assert 1 not in new.exits[0].rides  # Incoming root is not a new tail template.
    assert new.trips(trip_model(graph)) == old.trips(trip_model(graph))
    assert all(set(ObservationContract.from_menu(old).cids_at(stop)).issubset(
        ObservationContract.from_menu(new).cids_at(stop)) for stop in graph.stops)
    assert 2 in ObservationContract.from_menu(new).cids_at(1)
    assert any(r['first_cid']==2 and r['disposition']=='added' for r in ledger['candidate_occurrences'])


def test_pickup_and_disconnected_prefix_rejections_are_explicit():
    graph, old = case()
    graph.connections[2].pickup_allowed = False
    new, ledger = expand_menu(graph, old, 4)
    assert 2 not in new.exits[0].rides
    assert any(r['first_cid']==2 and r['reason']=='no_pickup' for r in ledger['candidate_occurrences'])
    graph.connections[2].pickup_allowed = True
    graph.connections[3].dep_stop = 0
    new, ledger = expand_menu(graph, old, 4)
    assert 2 not in new.exits[0].rides
    assert any(r['first_cid']==2 and r['reason']=='noncontiguous_same_trip_prefix'
               for r in ledger['candidate_occurrences'])


def test_early_destination_in_new_prefix_absorbs_before_old_boarding_position():
    graph, old = case()
    graph.connections[2].arr_stop = 4
    graph.connections[3].dep_stop = 4
    new, _ = expand_menu(graph, old, 4)
    assert new.exits[0].rides[2].path == (2,)
    assert new.exits[0].rides[2].exits == {2:None}
    assert contains_menu(new, old)


def test_existing_boarding_route_gets_new_continuation_without_losing_old_exit():
    graph, old = case()
    old.exits[0].rides[2] = Ride(2, (2,3), {3:old.exits[1]})
    new, ledger = expand_menu(graph, old, 4)
    route = new.exits[0].rides[2]
    assert route.path == (2,3,4)
    assert route.exits[4] is None
    assert contains_menu(route.exits[3], old.exits[0].rides[2].exits[3])
    assert contains_menu(new, old)
    assert 2 in next(r for r in ledger['platform_changes'] if r['stop']==1)['augmented_first_cids']


def test_template_reuse_never_increases_original_platform_ride_depth():
    graph, old = case()
    for cid, trip, dep, arr, td, ta in (
        (6,'U',2,3,45,55), (7,'V',3,4,70,80), (8,'U',1,2,5,20),
    ):
        graph.add_connection(Connection(cid, trip, trip, dep, arr, td, ta))
    old.exits[1].rides[6] = Ride(6, (6,), {6:Platform(3, {7:Ride(7, (7,), {7:None})})})
    new, ledger = expand_menu(graph, old, 4)
    assert depth(new.exits[0]) == depth(old.exits[0]) == 1
    assert depth(new) == depth(old)
    assert 8 not in new.exits[0].rides
    assert any(r['first_cid']==8 and r['reason']=='remaining_ride_depth'
               for r in ledger['candidate_occurrences'])
    assert new.trips(trip_model(graph)) == old.trips(trip_model(graph))


def test_labels_are_a_separate_audit_and_missing_endpoints_do_not_remove_candidate():
    graph, old = case()
    graph.connections[2].cancel_prob = 1.  # No static probability pruning.
    new, ledger = expand_menu(graph, old, 4)
    metadata = {c.id:{kind:dict(status='REAL', time_seconds=clock, official_journey=c.trip_id)
               for kind, clock in (('departure', c.dep_time), ('arrival', c.arr_time))}
               for c in graph.connections}
    metadata[2]['arrival']['status'] = 'MISSING'
    coverage = endpoint_coverage(graph, metadata, new, ObservationContract.from_menu(new))
    assert any(r.get('cid')==2 and r['reason']=='missing_or_ambiguous' for r in coverage['issues'])
    assert 2 in new.exits[0].rides and contains_menu(new, old)
    again, repeated = expand_menu(graph, old, 4)
    assert node_structure(again) == node_structure(new)
    assert repeated == ledger


def test_root4_recovery_is_produced_by_public_rule_without_replay_labels():
    root = read('experiments/reconstruction_v50/menus/004.json.gz')['wide_root']
    graph = context(read('experiments/reconstruction_v34/data.json.gz'), root['day'])[0]
    old = root_ride(graph, root)
    new, ledger = expand_menu(graph, old, root['goal'])
    assert 145 not in old.exits[111].rides
    recovered = new.exits[111].rides[145]
    assert recovered.path[:3] == (145,182,193)
    assert recovered.exits[193] is None
    assert contains_menu(new, old)
    assert new.trips(trip_model(graph)) == old.trips(trip_model(graph))
    assert 145 in ObservationContract.from_menu(new).cids_at(new.exits[111].stop)
    assert ledger['counts']['added_first_cids'] > 0
