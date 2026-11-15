"""Same-menu ordinary controls and public-only timetable replanning."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v33.execute import Executor as OriginalControls
from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine, PhysicalHistory
from experiments.reconstruction_v51.baselines import BaselinePolicy, CONTROLS, MeanSnapshotCSA, MenuLayout
from src import evolving_trip_belief as evolving
from src.transit_graph import Connection, Stop, TransitGraph
from test_v46_oracle import case


@pytest.mark.parametrize('label', tuple(CONTROLS))
def test_existing_controls_preserved_with_shared_information(monkeypatch, label):
    graph, model, params, history, platform = case(monkeypatch, forecasts=True)
    history.observe_platform(tuple(platform.rides), 9., 10., {0:10., 2:10.})
    planner = MatchedContinuationPlanner(graph, model, params, 60)
    actual = BaselinePolicy(graph, planner, 60, label)
    other = MatchedContinuationPlanner(graph, model, params, 60)
    engine = PhysicalDecisionEngine(other, variant='frontier')
    method, waiting = CONTROLS[label]
    control = OriginalControls(graph, other, 60, method=method, waiting=waiting)
    expected = engine.platform(platform, 9., 9., history, (0, 2), (0, 2), 10.,
                              choose=control.choose, method=method)
    assert actual.platform(platform, 9., 9., history, (0, 2), (0, 2), 10.) == expected
    history.observe(0, 'arrival', 20.)
    assert actual.onboard(platform.rides[0], 0, 20., history) == engine.onboard(
        platform.rides[0], 0, 20., history, method=method)


def deterministic_case(monkeypatch, transfer=0):
    monkeypatch.setattr(evolving, 'DELAYS', np.array([0.]))
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=transfer if stop == 1 else 0))
    for cid, trip, origin, target, dep, arr in (
        (0, 'A', 0, 1, 10, 20), (1, 'A', 1, 3, 22, 40),
        (2, 'B', 0, 3, 10, 35), (3, 'C', 1, 3, 25, 30)):
        graph.add_connection(Connection(cid, 'R', trip, origin, target, dep, arr))
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False,
                                      travel_sd=.05, dwell_sd=.05, cancel_persistence=.8)
    params = {'R':dict(mean=0., std=.05, cancel_rate=0.)}
    fit = dict(route_kind={}, pooled={kind:dict(bias_seconds=0., sd_seconds=3.)
                                     for kind in ('departure', 'arrival')})
    history = PhysicalHistory(model, params, fit=fit)
    tail = Platform(1, {3:Ride(3, (3,), {3:None})})
    platform = Platform(0, {0:Ride(0, (0, 1), {0:tail, 1:None}), 2:Ride(2, (2,), {2:None})})
    return graph, model, params, history, platform


def test_snapshot_scan_uses_legal_transfer_and_current_visible_offer(monkeypatch):
    graph, model, params, history, platform = deterministic_case(monkeypatch)
    history.observe_platform(tuple(platform.rides), 9., 10., {0:10., 2:10.})
    control = BaselinePolicy(graph, MatchedContinuationPlanner(graph, model, params, 60), 60, 'snapshot_csa')
    choice, receipt = control.platform(platform, 9., 9., history, (0, 2), (0, 2), 10.)
    assert choice == 0
    assert receipt['values'] == {0:30., 2:35.}
    history.observe(0, 'arrival', 20.)
    alight, receipt = control.onboard(platform.rides[0], 0, 20., history)
    assert alight and receipt['alight'] == 30. and receipt['continue_cost'] == 40.


def test_snapshot_scan_does_not_add_transfers_at_undeclared_shared_stops(monkeypatch):
    graph, model, params, history, platform = deterministic_case(monkeypatch)
    # C is in the common menu through B, but the A ride has no transfer exit
    # at stop1. Scanning the union of cids without lifting would board C.
    graph.connections[2].arr_stop = 1
    tail = platform.rides[0].exits.pop(0)
    platform.rides[2].exits = {2:tail}
    control = BaselinePolicy(graph, MatchedContinuationPlanner(graph, model, params, 60), 60, 'snapshot_csa')
    history.observe_platform(tuple(platform.rides), 9., 10., {0:10., 2:10.})
    _, receipt = control.platform(platform, 9., 9., history, (0, 2), (0, 2), 10.)
    assert receipt['values'][0] == 40.


def test_snapshot_respects_transfer_ready_and_does_not_reset_waiting(monkeypatch):
    graph, model, params, history, platform = deterministic_case(monkeypatch, transfer=6)
    control = BaselinePolicy(graph, MatchedContinuationPlanner(graph, model, params, 60), 60, 'snapshot_csa')
    history.observe(0, 'departure', 10.)
    history.observe(0, 'arrival', 20.)
    alight, receipt = control.onboard(platform.rides[0], 0, 20., history)
    assert not alight
    assert receipt['alight'] == 60.
    # A platform already entered at20 has ready26 even at a later decision.
    tail = platform.rides[0].exits[0]
    history.observe(3, 'departure', 25.)
    choice, receipt = control.platform(tail, 20., 26., history, (3,), (3,), 25.)
    assert choice is None and receipt['ready'] == 26.


def test_snapshot_uses_only_received_history_and_rebuilds_after_new_evidence(monkeypatch):
    graph, model, params, history, platform = case(monkeypatch, forecasts=False)
    control = BaselinePolicy(graph, MatchedContinuationPlanner(graph, model, params, 60), 60, 'snapshot_csa')
    before = control.csa._snapshot(platform, history, 9.)['means'][2, 'arrival']
    frame_index = history.frame_index
    history.frames = [dict(receipt_seconds=1000., trip_statuses=[], endpoint_updates=[
        [2, 'arrival', 'FORECAST', 1000.]])]
    same = control.csa._snapshot(platform, history, 9.)['means'][2, 'arrival']
    assert same == before and history.frame_index == frame_index
    history.observe(2, 'departure', 13.)
    after = control.csa._snapshot(platform, history, 13.)['means'][2, 'departure']
    assert after == 13.


def test_equal_clock_connections_do_not_depend_on_menu_creation_order():
    graph = SimpleNamespace(stops={1:SimpleNamespace(min_transfer_time=0)})
    tail = Platform(1, {1:Ride(1, (1,), {1:None})})
    root = Platform(0, {0:Ride(0, (0,), {0:tail})})
    layout = MenuLayout(root)
    times = {0:(10.,10.), 1:(10.,12.)}
    rows = [dict(row, departure=times[row['cid']][0], arrival=times[row['cid']][1])
            for row in layout.records]
    # Tail record is created first: its reachability is supplied later in the
    # same departure-time block, requiring the ordinary zero-time closure.
    assert [row['cid'] for row in rows] == [1, 0]
    scan = MeanSnapshotCSA(graph, None, 60)
    assert scan._scan(dict(layout=layout, records=rows), 9., platform=root, ready=9.) == 12.


def test_snapshot_layout_cache_does_not_confuse_equal_distinct_menu_objects(monkeypatch):
    graph, model, params, history, platform = deterministic_case(monkeypatch)
    other = Platform(platform.stop, dict(platform.rides))
    planner = MatchedContinuationPlanner(graph, model, params, 60)
    scan = MeanSnapshotCSA(graph, planner, 60)
    for node in (platform, other):
        snapshot = scan._snapshot(node, history, 9.)
        assert scan._scan(snapshot, 9., platform=node, ready=9.) == 30.
