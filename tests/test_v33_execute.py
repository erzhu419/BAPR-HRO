from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v33.execute import ArchivedWorld, Executor
from experiments.reconstruction_v33.history import ConditionalEngine
from experiments.reconstruction_v33.nodes import Ride, platform_from_options
from experiments.reconstruction_v33 import run
from src.evolving_trip_belief import _ChainLaw
from src.full_menu_opportunity import FullMenuOpportunity
from src.transit_graph import TransitGraph, Stop, Connection


def fixture():
    graph = TransitGraph()
    for sid in range(5):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=5 if sid == 1 else 0))
    rows = [('A',0,1,10,20), ('B',1,2,25,40), ('B',2,3,45,60),
            ('C',2,4,50,70), ('D',3,4,70,80), ('E',1,4,30,90)]
    metadata, positions, laws = {}, {}, {}
    by_trip = {}
    for cid, (trip, origin, goal, departure, arrival) in enumerate(rows):
        graph.add_connection(Connection(cid, trip, trip, origin, goal, departure, arrival))
        metadata[cid] = dict(trip_id=trip, dep_sequence=cid+1, arr_sequence=cid+2,
            scheduled_departure_seconds=departure, scheduled_arrival_seconds=arrival,
            departure=dict(status='REAL', time_seconds=departure, official_journey=trip),
            arrival=dict(status='REAL', time_seconds=arrival, official_journey=trip))
        by_trip.setdefault(trip, []).append(cid)
    for trip, cids in by_trip.items():
        times = []
        for cid in cids:
            for kind in ('departure', 'arrival'):
                positions[cid, kind] = trip, len(times)
                times.append(np.arange(0, 151, 5))
        size = len(times[0])
        kernel = np.full((size, size), -np.log(size))
        laws[trip] = _ChainLaw(times, np.full(size, -np.log(size)), lambda i,kernel=kernel:kernel,
            [np.zeros(size) for _ in times], [np.zeros(size, bool) for _ in times], (True,)*(len(times)-1))
    belief = SimpleNamespace(_positions=positions, _law=lambda trip, params:laws[trip])
    options = [dict(first_cid=1, first_path=[1], tail_paths=[[3]]),
               dict(first_cid=1, first_path=[1,2], tail_paths=[[4]]),
               dict(first_cid=5, first_path=[5], tail_paths=[])]
    platform = platform_from_options(graph, 1, options)
    return graph, metadata, belief, Ride(0, (0,), {0:platform})


class FakePlanner:
    """Deliberate public decisions; receives no world or metadata object."""
    def __init__(self, graph):
        self.graph, self.calls = graph, []

    def compile_ride(self, ride, history, method):
        def decision(cid, clock):
            self.calls.append(dict(kind='onboard', cid=cid, clock=clock,
                                   history=history.public_state(), signature=history.signature()))
            return cid != 1, dict(current_cid=cid)  # Stay on B at its first exit.
        return SimpleNamespace(decision=decision)

    def platform_reference(self, platform, history, entry, ready, method):
        self.calls.append(dict(kind='platform', stop=platform.stop, entry=entry,
                               history=history.public_state(), signature=history.signature()))
        graph = self.graph
        class Compiler:
            def departure_distribution(self, cid):
                return np.array([graph.connections[cid].dep_time]), np.array([1.]), 0.

            def clock_nominal_score(self, cid, clock):
                return graph.connections[cid].arr_time
        return FullMenuOpportunity(Compiler(), {cid:(cid,cid) for cid in platform.rides}, entry, ready, 150)


def execute(metadata=None):
    graph, original, belief, ride = fixture()
    planner, history = FakePlanner(graph), ConditionalEngine(belief, {})
    result = Executor(graph, planner, 150, waiting='first').run(
        ride, 0, history, ArchivedWorld(original if metadata is None else metadata))
    return result, planner, history, ride


def test_boarded_tail_vehicle_keeps_both_exits_open_and_replans_at_each():
    result, planner, _, ride = execute()
    assert ride.exits[0].rides[1].path == (1, 2)
    assert set(ride.exits[0].rides[1].exits) == {1, 2}
    decisions = [(e['cid'],e['choice']) for e in result['trace'] if e['kind']=='onboard']
    assert decisions == [(0,'alight'),(1,'continue'),(2,'alight')]
    assert result['status'] == 'arrived' and result['loss'] == 80
    assert [c['cid'] for c in planner.calls if c['kind']=='onboard'] == [0,1,2]


def test_platform_history_travels_onboard_without_remote_window_extension():
    _, planner, history, _ = execute()
    at_last_exit = next(c for c in planner.calls if c.get('cid') == 2)
    remote = next(e for e in at_last_exit['history'] if e['cid']==5)
    assert remote == dict(trip='E',cid=5,kind='departure',evidence='unseen',windows=((20.,25.),))
    assert at_last_exit['clock'] == 60
    assert any(e['cid']==1 and e['kind']=='departure' and e['clock']==25 for e in at_last_exit['history'])
    assert any(e['cid']==0 and e['kind']=='arrival' and e['clock']==20 for e in history.public_state())
    assert history.law('E').marginal(0)[6] > 0  # Its departure30 was never observed after leaving.


def test_same_public_prefix_with_different_future_has_identical_current_decisions():
    _, metadata, _, _ = fixture()
    changed = deepcopy(metadata)
    changed[2]['arrival']['time_seconds'] = 65
    changed[4]['arrival']['time_seconds'] = 85
    first, planner_a, _, _ = execute(metadata)
    second, planner_b, _, _ = execute(changed)
    early_a = [c for c in planner_a.calls if c.get('clock',0) <= 40 and c.get('stop',1)==1]
    early_b = [c for c in planner_b.calls if c.get('clock',0) <= 40 and c.get('stop',1)==1]
    assert early_a == early_b
    assert first['trace'][:3] == second['trace'][:3]
    assert (first['loss'], second['loss']) == (80,85)


def test_mct_removes_an_early_offer_but_keeps_its_public_receipt():
    _, metadata, _, _ = fixture()
    metadata[1]['departure']['time_seconds'] = 24  # Entry20 + MCT5 means unboardable.
    result, planner, _, _ = execute(metadata)
    event = next(e for e in result['trace'] if e['kind']=='platform')
    assert event['ready'] == 25 and event['visible'] == (5,) and event['choice'] == 5
    assert event['observed'] == {1:24,5:30}
    assert result['status']=='arrived' and result['loss']==90
    assert not any(c.get('cid') in (1,2) for c in planner.calls)


def test_missing_or_inconsistent_chosen_endpoint_remains_unresolved():
    _, metadata, _, _ = fixture()
    metadata[1]['departure']['status'] = 'MISSING'
    result, _, _, _ = execute(metadata)
    assert result['loss'] is None and result['status'].startswith('unresolved:1:departure:MISSING')
    _, metadata, _, _ = fixture()
    metadata[1]['arrival']['official_journey'] = 'different_vehicle'
    result, _, _, _ = execute(metadata)
    assert result['loss'] is None and 'inconsistent_arrival_identity' in result['status']


def supplementation(monkeypatch, tmp_path, *, sequence=2, old_status='MISSING'):
    graph, metadata, _, _ = fixture()
    row = deepcopy(metadata[1])
    row.update(day='2023-07-17', local_city_cid=1, dep_sequence=sequence)
    if old_status == 'MISSING':
        metadata[1]['departure'] = dict(status='MISSING', time_seconds=None, official_journey=None)
        metadata[1]['arrival'] = dict(status='MISSING', time_seconds=None, official_journey=None)
    else:
        row['arrival']['time_seconds'] += 1
    (tmp_path/'endpoint_supplement.json').write_text(json.dumps(dict(records=[row])))
    monkeypatch.setattr(run,'HERE',tmp_path)
    monkeypatch.setattr(run,'dated_graph',lambda *a,**k:(graph,None,{},None,metadata,None))
    monkeypatch.setattr(run,'SegmentDurationBelief',lambda *a,**k:object())
    fit = dict(duration_fit={},increments={},route_priors={'pooled':{}})
    return metadata, lambda:run.setup_day({},'2023-07-17',fit,{})


def test_supplement_only_fills_matching_missing_event_slots(monkeypatch,tmp_path):
    original, apply = supplementation(monkeypatch,tmp_path)
    _, replaced, _, _, applied = apply()
    assert applied == [(1,'departure'),(1,'arrival')]
    assert replaced[1]['arrival']['time_seconds']==40
    assert original[1]['arrival']['status']=='MISSING'
    _, apply = supplementation(monkeypatch,tmp_path,old_status='REAL')
    with pytest.raises(ValueError,match='already resolved'):
        apply()


def test_same_trip_different_event_sequence_is_not_a_supplement_match(monkeypatch,tmp_path):
    _, apply = supplementation(monkeypatch,tmp_path,sequence=99)
    with pytest.raises(ValueError,match='sequence|event|slot'):
        apply()


def test_partial_departure_consensus_keeps_conflicting_arrival_missing(monkeypatch,tmp_path):
    original, apply = supplementation(monkeypatch,tmp_path)
    path = tmp_path/'endpoint_supplement.json'
    data = json.loads(path.read_text())
    data['records'][0].pop('arrival')
    path.write_text(json.dumps(data))
    _, replaced, _, _, applied = apply()
    assert applied == [(1,'departure')]
    assert replaced[1]['departure']['status'] == 'REAL'
    assert replaced[1]['arrival'] == original[1]['arrival']
    assert replaced[1]['arrival']['status'] == 'MISSING'
