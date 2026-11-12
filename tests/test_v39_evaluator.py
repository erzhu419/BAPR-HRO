"""Synthetic-chain equivalence and physical-execution regressions only."""
from types import SimpleNamespace

import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v34.conditional_reference import AnchoredWorld, SnapshotExecutor as OriginalExecutor
from experiments.reconstruction_v38.evaluator import evaluate_actions as evaluate_v38
from experiments.reconstruction_v39.evaluator import evaluate_actions
from experiments.reconstruction_v39.executor import SnapshotExecutor
from experiments.reconstruction_v39 import state
from test_v38_evaluator import fixture, scientific


@pytest.mark.parametrize('variant',['v38','fullmemo','frontier'])
def test_all_variants_keep_full_fixed_bank_outcomes_and_charge_all_stages(variant):
    graph,model,parameters,node,history=fixture()
    before=history.public_state()
    args=(graph,model,parameters,100,node,dict(kind='onboard',cid=0,clock=10),history,'continue','alight')
    original=evaluate_v38(*args,count=8,seed=37050,variant='fast')
    result=evaluate_actions(*args,count=8,seed=37050,variant=variant)
    assert scientific(result)==scientific(original)
    assert history.public_state()==before and history.frame_index==1
    cost=result['cost']
    assert cost['planner_instances']==1
    for unit,total in (('process_cpu','total_process_cpu'),('wall','total_wall')):
        assert cost[total]>=cost['preparation'][unit]+cost['decision'][unit]+sum(
            arm[unit] for arm in cost['action_continuations'].values())
    if variant=='v38':
        assert cost['engine_stats_total']=={}
    else:
        assert cost['engine_stats_total']['platform_hits']>0
        for key,total in cost['engine_stats_total'].items():
            assert sum(arm['engine_stats'][key] for arm in cost['action_continuations'].values())==pytest.approx(total)


def test_the_same_engine_is_shared_across_both_arms_and_inputs_keep_future_frames(monkeypatch):
    graph,model,parameters,node,history=fixture()
    engines=[]
    original=state.DecisionEngine

    class Engine(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            engines.append(self)

    monkeypatch.setattr(state,'DecisionEngine',Engine)
    result=evaluate_actions(graph,model,parameters,100,node,dict(kind='onboard',cid=0,clock=10),
        history,'continue','alight',count=8,seed=37050,variant='frontier')
    assert len(engines)==1
    assert all(arm['engine_stats']['platform_requests']>0 for arm in result['cost']['action_continuations'].values())
    assert result['future_messages']==0 and len(history.frames)==2 and history.frame_index==1


@pytest.mark.parametrize('departure,arrival,departure_canceled,arrival_canceled,expected',[
    (33,32,False,False,'projection_incompatible:'),
    (33,40,True,False,'service_or_deadline_failure'),
    (33,40,False,True,'service_failure'),
    (33,40,False,False,'arrived'),
])
def test_hook_executor_preserves_anchor_projection_cancellation_and_legal_terminal_execution(
        departure,arrival,departure_canceled,arrival_canceled,expected):
    graph,model,parameters,node,history=fixture()
    # A new first-hand departure is off the frozen two-atom departure grid.
    public=[] if departure_canceled else [dict(cid=4,kind='departure',evidence='observed',clock=departure)]
    bank=SimpleNamespace(belief=model,events={
        (4,'departure'):([32],[departure_canceled]),(4,'arrival'):([arrival],[arrival_canceled])})
    platform=Platform(2,{4:Ride(4,(4,),{4:None})})
    row=dict(kind='platform',clock=departure,visible=[4])
    original=OriginalExecutor(graph,None,100,method='rolling_fixed',waiting='combined')
    hooked=SnapshotExecutor(graph,None,100,engine=object(),method='rolling_fixed',waiting='combined')
    left=original.forced(platform,row,4,history.clone(),AnchoredWorld(bank,0,public))
    right=hooked.forced(platform,row,4,history.clone(),AnchoredWorld(bank,0,public))
    assert {key:left[key] for key in ('loss','status','trace','decisions')}=={
        key:right[key] for key in ('loss','status','trace','decisions')}
    assert right['status'].startswith(expected)
