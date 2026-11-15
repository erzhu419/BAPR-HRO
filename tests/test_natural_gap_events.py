"""Saved-history event wiring and one final frozen-date calibration call."""
from copy import deepcopy
from datetime import date, timedelta

import pytest

from experiments.decision_gap_v1.panel import read, write
from experiments.decision_gap_v1.transfer_events import register_opportunities
from experiments.natural_gap_v2 import events as report
from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v37.controller import node_structure
from test_decision_gap_events import case


@pytest.fixture
def saved_root(monkeypatch):
    graph,node,model,history,_,metadata,_ = case()
    root=dict(root_id='root',day='2023-11-05',start=0,deadline=100,first_trip='B')
    menu=dict(root=root,original_node=node_structure(node),expanded_node=node_structure(node),
              registration=register_opportunities(graph,node,0,10))
    receipt=dict(states=[dict(kind='onboard',cid=0,clock=10.,public_history=[
        dict(evidence='observed',cid=0,kind='arrival',trip='B',clock=10.)])],
        outcome={'status':'arrived'},coverage={'source_available':True})
    physical_fit,emission = {'frozen_physical':'fixture'},{'frozen_emission':'repaired_fixture'}
    calls=[]

    def setup(g,fit):
        assert g is graph and fit is physical_fit
        return model,{'parameters':'fixture'}

    def restore(m,parameters,fit,public,clock):
        assert m is model and fit is emission and clock==10.
        assert public==receipt['states'][0]['public_history']
        calls.append((fit,public))
        return history

    monkeypatch.setattr(report,'setup',setup)
    monkeypatch.setattr(report,'restore_physical_snapshot',restore)
    source=dict(source_parse_complete=True,identity_mapping_complete=True,
                sampled_snapshots=17,capture_range=[0,90])
    return graph,metadata,menu,receipt,physical_fit,emission,source,calls


def evaluate(fixture):
    graph,metadata,menu,receipt,physical,emission,source,_ = fixture
    return report.root_events(graph,metadata,menu,receipt,0,physical,emission,source)


def test_repaired_saved_history_without_numeric_forecasts_still_has_model_and_actual_labels(saved_root):
    rows,root = evaluate(saved_root)
    assert len(rows)==1 and len(saved_root[-1])==1
    assert rows[0]['model']['status']=='resolved'
    assert rows[0]['actual']['actual_partition']=='01'
    assert rows[0]['groups']['forecast_age']=='none'
    assert root['registered']['candidate_pairs']==saved_root[2]['registration']['candidate_pairs']


@pytest.mark.parametrize('unknown',['unsupported','late_captures','identity_failure'])
def test_unknown_probability_never_erases_known_actual_partition(saved_root,monkeypatch,unknown):
    if unknown=='unsupported':
        def fail(*args,**kwargs):
            raise UnsupportedHistory('synthetic inconsistent evidence')
        monkeypatch.setattr(report,'event_probabilities',fail)
    elif unknown=='late_captures':
        saved_root[6]['capture_range']=[11,90]
    else:
        saved_root[6]['identity_mapping_complete']=False
    rows,_=evaluate(saved_root)
    assert rows[0]['model']['status']=='probability_unresolved'
    assert rows[0]['actual']['actual_partition']=='01'
    if unknown!='unsupported':
        assert saved_root[-1]==[]


def test_missing_initial_history_preserves_frozen_events_without_schedule_clock_imputation(saved_root):
    saved_root[3]['states']=[]
    saved_root[3]['outcome']['status']='unresolved:initial arrival:MISSING'
    rows,root=evaluate(saved_root)
    assert len(rows)==len(saved_root[2]['registration']['rows'])==1
    assert rows[0]['event']['clock'] is None
    assert rows[0]['actual']['possible_partitions']==['00','01','10','11']
    assert rows[0]['model']['status']=='probability_unresolved'
    assert root['initial_state']['status']=='unresolved:initial arrival:MISSING'
    assert saved_root[-1]==[]


def test_registered_counterfactual_candidates_cannot_change_after_actual_readout(saved_root):
    saved_root[2]['registration']['candidate_pairs'][0]['hub_cids']=[]
    with pytest.raises(ValueError,match='frozen event population'):
        evaluate(saved_root)


@pytest.fixture
def event_days(tmp_path):
    dates=[str(date(2023,11,5)+timedelta(days=i)) for i in range(14)]
    write(tmp_path/'date_manifest.json',dict(selected_dates=dates,dates_frozen=True))
    write(tmp_path/'selection.json',dict(selected=[{'day':day} for day in dates],
        registered=[dict(day=day,opportunities=1) for day in dates]))
    for i,day in enumerate(dates):
        event=dict(day=day,root_index=i,event_index=0,incoming_trip='A',outgoing_trip='B',
                   model={'status':'probability_unresolved'},
                   actual={'status':'resolved','actual_partition':'10','possible_partitions':['10']})
        write(tmp_path/'events/results'/f'{day}.json.gz',dict(day=day,events=[event],
            roots=[dict(root_index=i,initial_state={'status':'observed'},
                        registered={'rows':[event],'candidate_pairs':[]})],platform_pairs=[],
            actual_available=True,messages_available=i!=0))
    return tmp_path,dates


def test_final_calibration_waits_for_every_day_and_runs_once_without_claiming_actual_platforms(event_days,monkeypatch):
    directory,dates=event_days
    path=directory/'events/results'/f'{dates[-1]}.json.gz';last=read(path);path.unlink()
    calls=[]

    def bootstrap(rows,manifest):
        calls.append((rows,manifest))
        assert len(rows)==len(manifest['days'])==14
        assert manifest['days'][0]['available'] is False
        assert rows[0]['actual']['possible_partitions']==['10']
        return {'verdict':'unresolved'}

    monkeypatch.setattr(report,'calibration',bootstrap)
    with pytest.raises(FileNotFoundError):
        report.finalize(directory)
    assert calls==[]
    write(path,last)
    summary=report.finalize(directory)
    assert len(calls)==1 and summary['platform_verification']['status']=='unresolved'
    assert summary['platform_verification']['actual_platform_pairs_verified']==0
    assert not summary['reconstruction_admitted']
    with pytest.raises(ValueError,match='already recorded'):
        report.finalize(directory)
    assert len(calls)==1
