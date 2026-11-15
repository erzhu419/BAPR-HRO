"""Public freezing, construction billing and compressed-source separation."""
from datetime import date, timedelta
import struct

import pytest

from experiments.decision_gap_v1.panel import read, write
from experiments.natural_gap_v2 import freeze as freezer, prepare as preparer
from experiments.natural_gap_v2 import source_actuals
from experiments.reconstruction_v33.nodes import Platform, Ride
from src.transit_graph import Connection, Stop, TransitGraph


DAYS = [(date(2023,3,23)+timedelta(days=i)).isoformat() for i in range(14)]


@pytest.mark.parametrize('reason', ['already_developed', 'original_capture_window_incomplete_or_unverified'])
def test_ineligible_day_is_retained_and_prevents_partial_freeze(tmp_path, reason):
    write(tmp_path/'protocol.json', {'date_selection': {'target_days': 14}})
    write(tmp_path/'exposure_inventory.json', {'known_exposed_service_dates':
        [DAYS[3]] if reason == 'already_developed' else []})
    write(tmp_path/'rt_source_metadata.json', {'candidate_days': [{'date': day} for day in DAYS]})
    write(tmp_path/'static_source_manifest.json', {'source_version_by_date': {
        day: '2023-03-22' if day <= '2023-03-29' else '2023-03-29' for day in DAYS}})
    write(tmp_path/'actual_sources.json', {'days': [dict(day=day,
        official_daily_member_count=1, members=[{'name': day+'_istdaten.csv'}]) for day in DAYS]})
    write(tmp_path/'static_acquisition.json', {'dates': DAYS})
    for day in DAYS:
        write(tmp_path/'rt_clocks'/f'{day}.json', {'capture_coverage_complete':
            day != DAYS[3] or reason == 'already_developed'})
    with pytest.raises(ValueError, match='Not fourteen'):
        freezer.freeze(tmp_path)
    assert not (tmp_path/'date_manifest.json').exists()
    receipt = read(tmp_path/'date_consideration.json')
    assert len(receipt['rows']) == 14 and receipt['selected_count'] == 13
    assert receipt['rows'][3]['reasons'] == [reason]


def public_fixture(path, monkeypatch):
    write(path/'date_manifest.json', dict(dates_frozen=True, selected_dates=DAYS,
                                       frozen_at_utc='2026-11-15T00:00:00+00:00'))
    write(path/'exposure_inventory.json', {'known_exposed_service_dates': []})
    write(path/'protocol.json', {'fixed': {'policies': ['cheap','full128'], 'rollout_limits': {}}})
    write(path/'scheduled.json.gz', {'stops': {}, 'dates': {day: {'connections': []} for day in DAYS}})
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=5))
    for cid,trip,a,b,dep,arr in [(0,'R',0,1,10,20),(1,'R',1,2,25,35),
                                (2,'T',1,2,15,30),(3,'T',2,3,40,50),(4,'S',1,3,30,80)]:
        graph.add_connection(Connection(cid,trip,trip,a,b,dep,arr))
    original = Ride(0,(0,1),{0:Platform(1,{4:Ride(4,(4,),{4:None})}),
                            1:Platform(2,{3:Ride(3,(3,),{3:None})})})
    root = dict(root_id='public_root',day=DAYS[0],goal=3,start=0,deadline=100)
    metadata = {c.id:{kind:dict(status='REAL',time_seconds=clock,official_journey=c.trip_id)
                for kind,clock in [('departure',c.dep_time),('arrival',c.arr_time)]}
                for c in graph.connections}
    monkeypatch.setattr(preparer,'context',lambda *_:(graph,None,None,{},metadata,None))
    monkeypatch.setattr(preparer,'static_catalog',lambda *_:{'candidates':[root]})
    monkeypatch.setattr(preparer,'select_public',lambda *_:{'selected':[root]})
    monkeypatch.setattr(preparer,'root_ride',lambda *_:original)
    clock = [0.]
    monkeypatch.setattr(preparer,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(preparer,'process_time',lambda:clock[0])
    def wide(*_):
        clock[0] += 2.
        return root,{}
    monkeypatch.setattr(preparer,'wide_root',wide)
    expand = preparer.expand_menu
    def closure(*args):
        result = expand(*args)
        clock[0] += 3.
        return result
    monkeypatch.setattr(preparer,'expand_menu',closure)
    empty_calls = []
    def placeholders(day, connections, rows):
        assert rows == []
        empty_calls.append(day)
        return {'n_scheduled_connections':len(connections)}
    monkeypatch.setattr(preparer,'attach_outcomes',placeholders)
    preparer.prepare(path)
    return metadata,empty_calls


def test_public_menu_uses_empty_labels_and_charges_wide_plus_closure(tmp_path, monkeypatch):
    _,calls = public_fixture(tmp_path,monkeypatch)
    assert calls == DAYS
    menu = read(tmp_path/'menus/000.json.gz')
    assert menu['actual_values_used'] is False
    assert menu['cost'] == {'process_cpu':5.,'wall':5.}
    assert read(tmp_path/'selection.json')['selection_uses_target_values'] is False
    assert read(tmp_path/'scheduled.json.gz')['dates'][DAYS[0]] == {'connections':[]}
    assert read(tmp_path/'contract.json')['original_72_additional_budget'] == 0


def test_expanded_only_missing_endpoint_stays_in_registered_root(tmp_path, monkeypatch):
    metadata,_ = public_fixture(tmp_path,monkeypatch)
    metadata[2]['arrival']['status'] = 'MISSING'  # Added prefix, absent from old menu.
    before = read(tmp_path/'menus/000.json.gz')
    preparer.attach(tmp_path)
    coverage = read(tmp_path/'endpoint_coverage.json')
    assert len(coverage['roots']) == 1 and coverage['missing_roots_dropped'] is False
    assert any(issue.get('cid') == 2 for issue in coverage['roots'][0]['issues'])
    assert coverage['days'][DAYS[0]]['actual_source_available'] is False
    assert read(tmp_path/'menus/000.json.gz') == before


def test_compressed_cache_never_decodes_and_reuses_completed_ranges(tmp_path,monkeypatch):
    day = DAYS[0]
    # Deliberately uninterpretable bytes: caching must not examine payload/schema.
    payload = b'not decompressed before menu freeze' * 8
    write(tmp_path/'actual_metadata_2023-03.json',dict(url='official',members=[dict(
        name=day+'_istdaten.csv',absolute_local_header_offset=1000,
        compressed_bytes=len(payload),uncompressed_bytes=999)]))
    header = struct.pack('<4s5H3L2H',b'PK\x03\x04',20,0,8,0,0,0,0,0,0,0)
    boundaries = [len(payload)*i//8 for i in range(9)]
    fail_start = 1030+boundaries[3]
    calls = []
    failed = [False]
    def fetch(url,start,length):
        calls.append(start)
        if start == 1000:
            return header
        if start == fail_start and not failed[0]:
            failed[0] = True
            raise TimeoutError('Reachable interrupted source range')
        return payload[start-1030:start-1030+length]
    monkeypatch.setattr(source_actuals,'fetch_range',fetch)
    monkeypatch.setattr(source_actuals.zlib,'decompress',lambda *_:pytest.fail('Premature decode'))
    with pytest.raises(TimeoutError):
        source_actuals.cache_day(tmp_path,day)
    assert len(list((tmp_path/'actual_cache_parts'/day).glob('*.deflate'))) == 7
    source_actuals.cache_day(tmp_path,day)
    for start in [1030+b for b in boundaries[:-1]]:
        assert calls.count(start) == (2 if start == fail_start else 1)
    before = len(calls)
    source_actuals.cache_day(tmp_path,day)
    assert len(calls) == before
    receipt = read(tmp_path/'actual_cache'/f'{day}.cache.json')
    assert receipt['target_values_read'] is False
