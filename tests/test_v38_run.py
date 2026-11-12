"""Receipt normalization and online seed assignment regressions; no sampling."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v38 import run


def outcome():
    public = [dict(trip='a',cid=2,kind='departure',evidence='unseen',windows=((2.,9.),(12.,15.))),
              dict(trip='a',cid=0,kind='arrival',evidence='observed',clock=10.)]
    return dict(loss=40.,status='arrived',decisions=dict(onboard=1,platform=1),world_index=0,
        trace=[dict(kind='onboard',cid=0,clock=10.,choice='alight',public_history=public,
                    receipt=dict(alight=50.,continuation=60.)),
               dict(kind='platform',stop=1,clock=20.,choice=2,visible=(2,),observed={2:20.},
                    public_history=public,receipt=dict(board=50.,wait=70.))],total_wall=.1)


def bank():
    row = outcome()
    return dict(actions=dict(base='alight',alternative='continue'),
        paired=dict(full_bank_mean_gain_interval=(-2.,3.),unresolved_pairs=1),
        selected_action='alight',seed=37050,count=128,public_history=row['trace'][0]['public_history'],
        rows=dict(base=[row],alternative=[deepcopy(row)]),cost=dict(total_wall=.2))


def test_bank_science_normalizes_saved_containers_without_changing_content():
    live = bank()
    saved = json.loads(json.dumps(live))
    assert isinstance(live['public_history'][0]['windows'],tuple)
    assert isinstance(saved['public_history'][0]['windows'],list)
    assert run.bank_science(live)==run.bank_science(saved)
    saved['cost']['total_wall']=99.
    saved['rows']['base'][0]['total_wall']=99.
    assert run.bank_science(live)==run.bank_science(saved)


@pytest.mark.parametrize('change',['history_order','window_order','number','trace_order','trace_status'])
def test_bank_science_retains_real_order_number_and_status_differences(change):
    live = bank()
    saved = json.loads(json.dumps(live))
    if change=='history_order':
        saved['public_history'].reverse()
    elif change=='window_order':
        saved['public_history'][0]['windows'].reverse()
    elif change=='number':
        saved['public_history'][0]['windows'][0][1]=9.01
    elif change=='trace_order':
        saved['rows']['base'][0]['trace'].reverse()
    else:
        saved['rows']['base'][0]['status']='unsupported_public_history:example'
    assert run.bank_science(live)!=run.bank_science(saved)


def test_physical_outcome_normalizes_windows_and_ignores_only_internal_receipts_and_timing():
    live = outcome()
    saved = json.loads(json.dumps(live))
    saved['trace'][0]['receipt']={'different_internal_computation':True}
    saved['total_wall']=99.
    assert run.physical_outcome(live)==run.physical_outcome(saved)


@pytest.mark.parametrize('change',['history_order','number','trace_order','action','status'])
def test_physical_outcome_preserves_public_trace_and_execution_differences(change):
    live = outcome()
    saved = json.loads(json.dumps(live))
    if change=='history_order':
        saved['trace'][0]['public_history'].reverse()
    elif change=='number':
        saved['trace'][0]['public_history'][0]['windows'][0][1]=9.01
    elif change=='trace_order':
        saved['trace'].reverse()
    elif change=='action':
        saved['trace'][0]['choice']='continue'
    else:
        saved['status']='no_legal_offer'
    assert run.physical_outcome(live)!=run.physical_outcome(saved)


def test_online_evaluates_from_manifest_seed_even_when_current_call_ordinal_changed(monkeypatch):
    planner = SimpleNamespace(model=object(),parameters={},compile_ride=lambda *args:
        SimpleNamespace(decision=lambda cid,clock:(True,dict(cid=cid,clock=clock))))
    history = SimpleNamespace(public_state=lambda:[dict(trip='a',cid=2,kind='departure',
        evidence='unseen',windows=((2.,9.),))])
    ride = Ride(0,(0,1),{0:Platform(1),1:None})
    diagnostic = run.current_diagnostics(planner,ride,0,10.,history)
    state = run.onboard_state(ride,0,10.,history,100.,root_index=2,day='2023-10-25',
        count=128,diagnostics=diagnostic)
    entry = dict(state=state,seed=39053,count=128,source='reference_cases/002_003.json.gz')
    executor = run.OnlineExecutor(None,planner,100.,root_index=2,day='2023-10-25',
        manifest=[entry],variant='shared')
    executor.gates.extend([{} for _ in range(6)])  # Original call 3 is now call 6.
    calls = []

    def evaluate(graph,model,parameters,current_state,node,current_history,seed,variant):
        assert current_state==state and current_state['count']==128
        assert node is ride and current_history is history
        calls.append((seed,variant))
        return dict(selected_action='continue',paired={'full_bank_mean_gain_interval':[-2.,-1.]},
            cost={'total_process_cpu':.1})

    def reject_old_receipt_read(*args,**kwargs):
        raise AssertionError('Online action must be recomputed, not loaded from an old outcome')

    monkeypatch.setattr(run,'evaluate',evaluate)
    monkeypatch.setattr(run,'read',reject_old_receipt_read)
    alight,receipt = executor.choose_onboard(ride,0,10.,history)
    assert calls==[(39053,'shared')]
    assert not alight and receipt['selected_action']=='continue'
    assert receipt['seed']==39053 and len(executor.gates)==7
    assert executor.gates[-1]['source']==entry['source']
