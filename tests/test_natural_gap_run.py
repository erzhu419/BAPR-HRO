"""Natural control source availability and saved initial public evidence."""
from types import SimpleNamespace

from experiments.decision_gap_v1.panel import read, write
from experiments.natural_gap_v2 import run as runner


def test_current_forecast_counts_use_saved_public_history_not_old_publications():
    graph=SimpleNamespace(connections=[SimpleNamespace(trip_id=t,dep_time=d,arr_time=a)
        for t,d,a in [('incoming',5,20),('tail',8,50),('cleared',30,60),('incoming',22,40)]])
    row=dict(outcome={'status':'arrived'},states=[dict(kind='onboard',cid=0,clock=20,
        public_history=[dict(evidence='observed',cid=0,kind='arrival',trip='incoming'),
                        dict(evidence='forecast',cid=3,kind='departure',trip='incoming',emission_selected=False),
                        dict(evidence='forecast',cid=1,kind='departure',trip='tail',emission_selected=True)])])
    frames=[dict(receipt_seconds=10,endpoint_updates=[[2,'arrival','FORECAST',60]]),
            dict(receipt_seconds=15,endpoint_updates=[[2,'arrival','NO_DATA',None]]),
            dict(receipt_seconds=25,endpoint_updates=[[1,'arrival','FORECAST',50]])]
    result=runner.initial_evidence(row,graph,{0:0,1:1,2:2,3:3},frames,'incoming',{0,1,2,3})
    assert result['predecision_numeric_messages']==2
    assert result['model_selected_numeric_messages']==1
    assert result['other_trip_numeric_messages']==1
    assert result['historical_future_tail_publications']==1
    assert result['historical_forecast_witnesses'][0]['cid']==2


def test_no_initial_state_keeps_explicit_missing_status():
    row=dict(states=[],outcome={'status':'unresolved:initial arrival:MISSING'})
    result=runner.initial_evidence(row,None,{},[],'incoming',set())
    assert result['initial_state']['clock'] is None
    assert result['initial_state']['status']=='unresolved:initial arrival:MISSING'
    assert result['predecision_numeric_messages']==0


def source_fixture(tmp_path):
    days=['empty','no_numeric','absent','bad_identity','incomplete_parse']
    write(tmp_path/'contract.json',dict(control_dates=days))
    for day,count in [('empty',0),('no_numeric',17)]:
        write(tmp_path/'messages'/f'{day}.json.gz',dict(frames=[],summary=dict(sampled_snapshots=count,
            source_parse_complete=True,identity_mapping_complete=True)))
    for day,parse,identity in [('bad_identity',True,False),('incomplete_parse',False,True)]:
        write(tmp_path/'messages'/f'{day}.json.gz',dict(frames=[],summary=dict(sampled_snapshots=17,
            source_parse_complete=parse,identity_mapping_complete=identity)))
    for name,value in [('data.json.gz',{}),('physical_fit.json',{}),('repaired_fit.json',{}),
                       ('endpoint_coverage.json',{'roots':[]})]:
        write(tmp_path/name,value)


def test_missing_capture_source_is_distinct_from_complete_source_without_numeric_messages(tmp_path):
    source_fixture(tmp_path)
    runner.initialize(tmp_path)
    assert runner.RESIDENT['source_available']=={'empty':False,'no_numeric':True,'absent':False,
                                               'bad_identity':False,'incomplete_parse':False}
    assert runner.RESIDENT['source_reasons']['bad_identity']==['message_identity_unresolved']


def test_missing_source_preserves_all_eight_unexecuted_roots(tmp_path,monkeypatch):
    menu=dict(root=dict(root_id='root',day='day',start=0,deadline=300,
                       origin_name='O',goal_name='G',first_trip='incoming'),
              new_observation_contract={'station_to_cids':[]},cost={'process_cpu':.5,'wall':.6})
    write(tmp_path/'menus/000.json.gz',menu)
    resident=dict(directory=tmp_path,graphs={'day':(object(),None,None,{},None,None)},
                  source_available={'day':False},source_reasons={'day':['no_predecision_capture']},
                  coverage={0:{'missing_endpoints':2}},load_cost={})
    monkeypatch.setattr(runner,'RESIDENT',resident)

    def forbidden(*args,**kwargs):
        raise AssertionError('Missing source cannot execute as a no-forecast controller')

    monkeypatch.setattr(runner,'one_arm',forbidden)
    runner.run_root(0)
    for policy in runner.POLICIES:
        row=read(tmp_path/'roots'/f'000_new_{policy}.json.gz')
        assert row['executed'] is False and row['outcome']['loss'] is None
        assert row['loss_cap']==300 and row['states']==row['gates']==[]
        assert row['cost']['process_cpu_including_menu'] is None
        assert row['cost']['shared_menu']==menu['cost']
        assert row['coverage']['full_menu']==resident['coverage'][0]


def test_source_complete_initial_actual_failure_is_executed_and_kept(tmp_path,monkeypatch):
    menu=dict(root=dict(root_id='root',day='day',start=0,deadline=300,
                       origin_name='O',goal_name='G',first_trip='incoming'),
              expanded_node={},new_observation_contract={'station_to_cids':[]},
              cost={'process_cpu':.5,'wall':.6})
    write(tmp_path/'menus/000.json.gz',menu)
    graph=SimpleNamespace(connections=[SimpleNamespace(trip_id='incoming')])
    context=(graph,None,None,{0:0},None,None)
    resident=dict(directory=tmp_path,graphs={'day':context},source_available={'day':True},
                  coverage={0:{'missing_endpoints':1}},load_cost={},messages={'dates':{'day':{'frames':[]}}})
    monkeypatch.setattr(runner,'RESIDENT',resident)
    monkeypatch.setattr(runner,'restore_node',lambda node:SimpleNamespace(paths=lambda:[(0,)]))
    called=[]

    def one(index,model,policy,*args):
        called.append((model,policy))
        return dict(states=[],outcome=dict(loss=None,status='unresolved:initial arrival:MISSING'),
                    cost=dict(process_cpu_including_menu=1.))

    monkeypatch.setattr(runner,'one_arm',one)
    runner.run_root(0)
    assert set(called)=={('repaired_fit',p) for p in runner.POLICIES}
    for policy in runner.POLICIES:
        row=read(tmp_path/'roots'/f'000_new_{policy}.json.gz')
        assert row['executed'] and row['outcome']['loss'] is None
        assert row['coverage']['source_available']
        assert row['coverage']['initial_state']['status']=='unresolved:initial arrival:MISSING'
