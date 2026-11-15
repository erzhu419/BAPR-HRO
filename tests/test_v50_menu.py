"""Candidate caps are declared limits; scheduled earliness is not physical proof."""
from copy import deepcopy

import numpy as np

from experiments.reconstruction_v32.screen import downstream_options, retain_options
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v50.menu import generation_ledger, scheduled_early_support, wide_root
from src import evolving_trip_belief as evolving
from src.recourse_itinerary_router import build_trip_blocks
from src.transit_graph import Connection, Stop, TransitGraph


def graph_and_protocol():
    graph = TransitGraph()
    for stop in (0, 1):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for cid in range(7):
        graph.add_connection(Connection(cid, f'R{cid}', f'T{cid}', 0, 1, 10+cid, 20+cid))
    protocol = dict(max_first_cids_per_hub=4, services_per_route=2,
        tail_first_wait=900, max_tail_legs=8, max_tail_rides=2)
    options = downstream_options(graph, build_trip_blocks(graph), 0, 10, 60, (), protocol)[1]
    root = dict(goal=1, deadline=60, first_cid=0,
        hubs=[dict(stop=0, arrival_cid=99, scheduled_ready=10,
                   options=retain_options(graph, options, 4))])
    return graph, protocol, root


def test_wide_menu_removes_only_cap_and_never_mutates_original():
    graph, protocol, root = graph_and_protocol()
    before = deepcopy(root)
    wide, ledger = wide_root(graph, root, protocol)
    assert root == before
    assert len({o['first_cid'] for o in root['hubs'][0]['options']}) == 4
    assert len({o['first_cid'] for o in wide['hubs'][0]['options']}) == 7
    assert len(ledger) == 7
    assert sum(r['original_exclusion'] == 'four_first_vehicle_cap' for r in ledger) == 3
    assert all(not r['exclusion_is_safe_dominance'] for r in ledger)
    hub = wide['hubs'][0]
    assert set(hub['variants']) == {o['first_cid'] for o in hub['options']}
    assert [o for variants in hub['variants'].values() for o in variants] == hub['options']
    assert {tail['rides'][0][0] for tail in hub['tails']} == set(hub['variants'])
    assert len(hub['tails']) == sum(max(1,len(o['tail_paths'])) for o in hub['options'])
    assert hub['retained_option_count'] == hub['raw_option_count'] == len(hub['options'])


def test_schedule_only_ledger_keeps_unsafe_filter_reasons(monkeypatch):
    graph, protocol, root = graph_and_protocol()
    graph.add_connection(Connection(7, 'R7', 'early', 0, 1, 5, 8))
    graph.add_connection(Connection(8, 'R8', 'late', 0, 1, 1000, 1010))
    graph.add_connection(Connection(9, 'R9', 'closed', 0, 1, 15, 25, pickup_allowed=False))
    ledger = generation_ledger(graph, root, protocol)
    first = {r['cid']:r for r in ledger['departures'] if r['depth'] == 1}
    assert first[7]['reasons'] == ['scheduled_departure_before_ready']
    assert not first[7]['safe_pruning']
    assert first[8]['reasons'] == ['900_second_window', 'scheduled_departure_horizon']
    assert not first[8]['safe_pruning']
    assert first[9]['reasons'] == ['no_pickup'] and first[9]['safe_pruning']
    assert ledger['declared_support']['strict_dominance_pruning'] == 0
    assert ledger['declared_support']['public_physical_impossibility_pruning'] == 0


def test_scheduled_early_vehicle_has_legal_public_delay_support(monkeypatch):
    monkeypatch.setattr(evolving, 'DELAYS', np.array([0., 8/60]))
    graph = TransitGraph()
    for stop in (0, 1):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    graph.add_connection(Connection(0, 'R', 'early', 0, 1, 5, 8))
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False,
        travel_sd=.1, dwell_sd=.1, cancel_persistence=.8)
    params = {'R':dict(mean=0., std=.1, cancel_rate=0.)}
    fit = dict(route_kind={}, pooled={k:dict(bias_seconds=0., sd_seconds=5.)
                                     for k in ('departure', 'arrival')})
    history = PhysicalHistory(model, params, fit=fit)
    rows = scheduled_early_support(graph, model, history, 0, 10., 60., ())
    assert len(rows) == 1
    assert rows[0]['scheduled_departure'] < 10
    assert rows[0]['public_support_legal']
    assert 0 < rows[0]['public_legal_mass'] <= 1
    assert rows[0]['legal_support_min'] >= 10
    assert not rows[0]['in_action_menu']
    history.observe(0, 'departure', 5.)
    after = scheduled_early_support(graph, model, history, 0, 10., 60., ())
    assert after[0]['public_legal_mass'] == 0
    assert not after[0]['public_support_legal']
