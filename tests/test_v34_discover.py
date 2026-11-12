from copy import deepcopy
from types import SimpleNamespace

from experiments.reconstruction_v34 import discover
from experiments.reconstruction_v34.discover import label_coverage, message_eligibility


def fixture():
    graph = SimpleNamespace(connections=[
        SimpleNamespace(id=0, trip_id='A', dep_stop=0, arr_stop=1, dep_time=10, arr_time=20),
        SimpleNamespace(id=1, trip_id='B', dep_stop=1, arr_stop=2, dep_time=30, arr_time=40)])
    root = dict(first_cid=0, first_path=[0], hubs=[dict(arrival_cid=0, stop=1,
        options=[dict(first_cid=1, first_path=[1], tail_paths=[])])])
    metadata = {c.id:dict(departure=dict(status='REAL',time_seconds=c.dep_time,official_journey=c.trip_id),
                        arrival=dict(status='REAL',time_seconds=c.arr_time,official_journey=c.trip_id))
                for c in graph.connections}
    return graph, root, metadata


def test_entire_retained_menu_must_have_endpoints_not_only_selected_path():
    graph, root, metadata = fixture()
    edges, missing = label_coverage(graph, metadata, root)
    assert edges == [0, 1] and missing == []
    metadata[1]['arrival']['status'] = 'MISSING'
    _, missing = label_coverage(graph, metadata, root)
    assert missing == [dict(cid=1,reason='missing_or_ambiguous',departure='REAL',arrival='MISSING')]


def test_signal_eligibility_uses_receipt_not_prediction_or_future_outcome():
    graph, root, metadata = fixture()
    frames = [dict(receipt_seconds=19, endpoint_updates=[[100,'departure','FORECAST',10000]]),
              dict(receipt_seconds=21, endpoint_updates=[[100,'arrival','FORECAST',40]])]
    expected = [dict(cid=1,kind='departure',receipt=19,decision_clock=20,scheduled_target=30)]
    assert message_eligibility(graph,metadata,{100:1},root,frames) == expected
    changed = deepcopy(metadata)
    changed[1]['departure']['time_seconds'] = 1000
    changed[1]['arrival']['time_seconds'] = 2000
    frames[0]['endpoint_updates'][0][-1] = -10000
    assert message_eligibility(graph,changed,{100:1},root,frames) == expected


def test_unrelated_vehicle_and_withdrawal_do_not_count_as_numeric_evidence():
    graph, root, metadata = fixture()
    frames = [dict(receipt_seconds=19,endpoint_updates=[
        [0,'departure','FORECAST',10], [100,'departure','NO_DATA',None],
        [999,'departure','FORECAST',30]])]
    assert message_eligibility(graph,metadata,{0:0,100:1},root,frames) == []


def test_new_source_archive_receives_official_mct_before_graph_construction(monkeypatch):
    data = dict(stops={'X':dict(name='X',latitude=0,longitude=0),
                       'Y':dict(name='Y',latitude=0,longitude=0)},
                dates={'day':dict(connections=[])})
    monkeypatch.setattr(discover,'read',lambda path:dict(stations={
        'X':dict(conservative_max_stated_seconds=300)}))
    graph,_,ids,_,_,_ = discover.context(data,'day')
    assert graph.stops[ids['X']].min_transfer_time == 300
    assert graph.stops[ids['Y']].min_transfer_time == 120
