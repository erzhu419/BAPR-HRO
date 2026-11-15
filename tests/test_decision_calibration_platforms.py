"""Static platform provenance must not become an actual-platform claim."""
from experiments.decision_calibration_v1.platform_pairs import endpoint, transfer_pair


def case():
    incoming = dict(trip_id='A', arr_sequence=2, arr_stop='8503000', scheduled_arrival_seconds=300)
    outgoing = dict(trip_id='B', dep_sequence=4, dep_stop='8503000', scheduled_departure_seconds=800)
    index = {('A',2):[dict(stop_id='8503000:0:1',arrival_time='00:05:00')],
             ('B',4):[dict(stop_id='8503000:0:2',departure_time='00:13:20')]}
    stops = {key:dict(parent_station='Parent8503000',location_type='0')
             for key in ('8503000:0:1','8503000:0:2')}
    stops['Parent8503000'] = dict(parent_station='',location_type='1')
    stops['8503000'] = dict(parent_station='Parent8503000',location_type='0')
    record = dict(from_stop_id='8503000:0:1',to_stop_id='8503000:0:2',transfer_type='2',min_transfer_time='420')
    return incoming,outgoing,index,stops,record


def test_exact_scheduled_pair_recovers_the_rule_but_never_actual_platform():
    a,b,index,stops,row = case()
    result = transfer_pair(a,b,420,index,stops,{('8503000:0:1','8503000:0:2'):[row]})
    assert result['status'] == 'planned_pair_verified'
    assert result['planned_pair_source_verified'] and result['fixed_value_matches']
    assert result['stated_pair_seconds'] == 420
    assert not result['actual_platform_pair_verified']
    assert not result['arrival']['actual_platform_verified']
    a['scheduled_arrival_seconds'] = 301
    assert endpoint(a,'arrival',index,stops)['status'] == 'schedule_mismatch'


def test_bare_bpuic_and_reverse_rules_cannot_verify_platform_pair():
    a,b,index,stops,row = case()
    parent = dict(row,from_stop_id='8503000')
    result = transfer_pair(a,b,420,index,stops,
        {('8503000','8503000:0:2'):[parent],('8503000:0:2','8503000:0:1'):[row]})
    assert result['status'] == 'exact_pair_rule_missing'
    assert result['station_alias_records'] == [parent]
    assert not result['applicable_inherited_rule']
    assert result['exact_records'] == [] and not result['planned_pair_source_verified']


def test_true_parent_station_rule_applies_to_children_including_same_platform():
    a,b,index,stops,row = case()
    index['B',4][0]['stop_id']='8503000:0:1'
    parent = dict(row,from_stop_id='Parent8503000',to_stop_id='8503000:0:1')
    result = transfer_pair(a,b,420,index,stops,{('Parent8503000','8503000:0:1'):[parent]})
    assert result['status']=='planned_pair_verified'
    assert result['applicable_inherited_rule'] and not result['exact_direct_pair']
    assert result['stated_pair_seconds']==420
    assert not result['actual_platform_pair_verified']


def test_ambiguous_sequence_and_station_only_endpoints_stay_unknown():
    a,b,index,stops,row = case()
    index['A',2].append(dict(stop_id='8503000:0:2',arrival_time='00:05:00'))
    assert transfer_pair(a,b,420,index,stops,{})['status'] == 'endpoint_unresolved'
    index['A',2] = [dict(stop_id='8503000',arrival_time='00:05:00')]
    stops['8503000'] = dict(parent_station='',location_type='1')
    result = transfer_pair(a,b,420,index,stops,{})
    assert result['status'] == 'station_only_endpoint'
    assert not result['planned_pair_source_verified']


def test_same_platform_missing_rule_is_not_invented_zero_and_conflicts_remain():
    a,b,index,stops,row = case()
    index['B',4][0]['stop_id']='8503000:0:1'
    assert transfer_pair(a,b,420,index,stops,{})['status'] == 'same_platform_without_stated_rule'
    index['B',4][0]['stop_id']='8503000:0:2'
    pair = ('8503000:0:1','8503000:0:2')
    result = transfer_pair(a,b,240,index,stops,{pair:[row]})
    assert result['status'] == 'fixed_mct_differs_from_planned_pair'
    assert not result['fixed_value_matches'] and result['fixed_mct_seconds'] == 240
    assert transfer_pair(a,b,420,index,stops,{pair:[dict(row,transfer_type='3')]})['status']=='planned_pair_prohibited'
