"""Filtered official static inputs retain the original schedule recipe."""
import csv
import gzip
from io import StringIO
import json
from zipfile import ZipFile

import pytest

from experiments.decision_calibration_v2.prepare_static import (
    dated_connections_from_static, extract_static, original,
)


def write_table(archive, name, fields, rows):
    handle=StringIO()
    writer=csv.DictWriter(handle,fieldnames=fields.split(',')); writer.writeheader(); writer.writerows(rows)
    archive.writestr(name,handle.getvalue())


def fixture(path):
    trips=[]; times=[]
    def trip(name, route='R', service='S', short='12', sequence=None):
        trips.append(dict(trip_id=name,route_id=route,service_id=service,trip_short_name=short))
        for seq,stop,clock in sequence or [(1,'8503000:0:1','08:00:00'),(2,'8503003:0:2','08:05:00')]:
            times.append(dict(trip_id=name,arrival_time=clock,departure_time=clock,stop_id=stop,
                stop_sequence=seq,pickup_type='1' if name=='A' and seq==1 else '0',drop_off_type='0'))
    trip('A'); trip('non_sbb','OTHER'); trip('unnamed',short='')
    trip('gap',sequence=[(1,'8503000:0:1','08:00:00'),(2,'outside','08:02:00'),(3,'8503003:0:2','08:05:00')])
    trip('late',sequence=[(1,'8503000:0:1','09:34:00'),(2,'8503003:0:2','09:36:00')])
    trip('edge',sequence=[(1,'8503000:0:1','09:30:00'),(2,'8503003:0:2','09:35:00')])
    trip('exception',service='T')
    stops=[dict(stop_id=key,stop_name=key,stop_lat='47.0',stop_lon='8.0',location_type=kind,parent_station=parent)
           for key,kind,parent in [('8503000','0','Parent8503000'),('8503000:0:1','0','Parent8503000'),
             ('Parent8503000','1',''),('8503003','0','Parent8503003'),('8503003:0:2','0','Parent8503003'),
             ('Parent8503003','1',''),('outside','0','')]]
    with ZipFile(path,'w') as archive:
        write_table(archive,'routes.txt','route_id,agency_id,route_short_name,route_long_name',[
            dict(route_id='R',agency_id='11',route_short_name='S',route_long_name=''),
            dict(route_id='OTHER',agency_id='99',route_short_name='Z',route_long_name='')])
        write_table(archive,'trips.txt','trip_id,route_id,service_id,trip_short_name',trips)
        write_table(archive,'stop_times.txt','trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type',times)
        write_table(archive,'stops.txt','stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station',stops)
        write_table(archive,'calendar.txt','service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date',[
            dict(service_id=s,**{day:active for day in ('monday','tuesday','wednesday','thursday','friday','saturday','sunday')},
                 start_date='20230101',end_date='20231231') for s,active in [('S','1'),('T','0')]])
        write_table(archive,'calendar_dates.txt','service_id,date,exception_type',[
            dict(service_id='S',date='20230309',exception_type='2'),dict(service_id='T',date='20230309',exception_type='1')])
        write_table(archive,'transfers.txt','from_stop_id,to_stop_id,transfer_type,min_transfer_time',[
            dict(from_stop_id='Parent8503000',to_stop_id='8503000:0:1',transfer_type='2',min_transfer_time='420'),
            dict(from_stop_id='outside',to_stop_id='8503000:0:1',transfer_type='2',min_transfer_time='100')])
        write_table(archive,'feed_info.txt','feed_start_date,feed_end_date,feed_version',[
            dict(feed_start_date='20230101',feed_end_date='20231231',feed_version='20230308')])


def test_region_filter_retains_platforms_parent_rules_and_original_adjacency(tmp_path):
    source=tmp_path/'official.zip'; fixture(source)
    runtime=tmp_path/'runtime'
    audit=extract_static(source,runtime)
    saved_root=original.ROOT
    stops,days=dated_connections_from_static(['2023-03-08','2023-03-09'],runtime)
    assert original.ROOT==saved_root
    assert set(stops)=={'8503000','8503003'}
    assert {r['trip_id'] for r in days['2023-03-08']}=={'A','edge'}
    assert {r['trip_id'] for r in days['2023-03-09']}=={'exception'}
    early=next(r for r in days['2023-03-08'] if r['trip_id']=='A')
    assert early['dep_platform_id']=='8503000:0:1' and early['arr_platform_id']=='8503003:0:2'
    assert (early['dep_sequence'],early['arr_sequence'])==(1,2)
    assert early['pickup_allowed'] is False and early['dropoff_allowed'] is True
    assert audit['row_counts']['transfers.txt']['retained_rows']==1
    assert 'Parent8503000' in (runtime/'data/swiss_gtfs/stops.txt').read_text()
    assert 'outside' not in (runtime/'data/zurich_gtfs/stop_times.txt').read_text()


def test_invalid_feed_date_is_rejected_before_replay_and_global_root_stays_fixed(tmp_path):
    source=tmp_path/'official.zip'; fixture(source)
    runtime=tmp_path/'runtime'; extract_static(source,runtime)
    saved_root=original.ROOT
    with pytest.raises(ValueError,match='outside'):
        dated_connections_from_static(['2024-01-01'],runtime)
    assert original.ROOT==saved_root
    with pytest.raises(ValueError,match='separate'):
        extract_static(source,saved_root)


def test_server_entry_uses_manifest_dates_and_cached_public_zip(tmp_path, monkeypatch):
    from experiments.decision_calibration_v2 import acquire_static
    code=tmp_path/'source'; code.mkdir()
    runtime=tmp_path/'runtime'; (runtime/'cache').mkdir(parents=True)
    source=runtime/'cache/official.zip'; fixture(source)
    (code/'date_manifest.json').write_text(json.dumps(dict(dates_frozen=True,
        days=[dict(day='2023-03-09',static_version='2023-03-08')])))
    (code/'static_metadata_2023-03-08.json').write_text(json.dumps(dict(
        url='https://example.invalid/official.zip', archive_bytes=source.stat().st_size)))
    monkeypatch.setattr(acquire_static,'HERE',code)
    receipts=acquire_static.acquire(runtime)
    assert receipts[0]['reused_complete_zip'] and receipts[0]['downloaded_bytes']==0
    assert receipts[0]['scheduled_counts']=={'2023-03-09':1}
    linked=code/'static/2023-03-08'
    assert linked.resolve()==runtime/'inputs/static_2023-03-08'
    with gzip.open(linked/'scheduled.json.gz','rt') as handle:
        scheduled=json.load(handle)
    assert set(scheduled['dates'])=={'2023-03-09'}
    assert scheduled['dates']['2023-03-09']['connections'][0]['trip_id']=='exception'
