"""Timing failures that the old departure-plus-grace executor could conceal."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.trc_revision_v1 import executor as mod


def command(*, issued=0., mode='first', target=None):
    return dict(mode=mode,target=target,issued_at=issued,ranks={1:0,2:1},ready_by_cid={1:0,2:0})


def test_command_must_exist_before_departure():
    assert mod.command_choice(command(issued=5),(1,),5) is None
    assert mod.command_choice(command(issued=4.9),(1,),5)==1
    assert mod.command_choice(command(issued=5),(1,),5,ideal=True)==1
    assert mod.command_choice(command(issued=4,mode='target',target=2),(1,),5) is None


def test_command_scores_only_positive_support_and_excludes_seen_departures():
    calls=[]
    class Compiler:
        def departure_distribution(self,cid):
            if cid==1:
                return np.array([0,5,10,15]),np.array([0.,.4,.4,0.]),.2
            assert cid==2
            return np.array([0,5,10,15]),np.zeros(4),1.
        def clock_nominal_scores(self,cid,clocks):
            calls.append((cid,clocks.tolist()))
            assert cid==1 and clocks.tolist()==[5,10] or cid==2 and len(clocks)==0
            return np.array([20.,30.]) if cid==1 else np.empty(0)
    class Planner:
        deadline=100.
        def ranks(self,platform):return {1:0,2:1,3:2}
        def platform_compiler(self,platform,history,method):
            assert set(platform.rides)=={1,2}
            return Compiler()
    chosen,receipt=mod.choose_command(Planner(),mod.PairPlatform(0,{1:'a',2:'b',3:'seen'}),
                                     History(),0.,0.,{3})
    assert calls==[(1,[5,10]),(2,[])]
    assert receipt['values']=={1:40.,2:100.}
    assert chosen['mode']=='first' and 3 not in chosen['ranks']


class History:
    def __init__(self,frames=()):self.frames=frames;self.observations=[]
    def observe_platform(self,cids,entry,clock,observed):
        self.observations.append((clock,dict(observed)))
    def public_state(self):return []
    def observe(self,*args):pass


class Commands(mod.PreissuedPlatformMixin):
    command_latency=True
    clock_contract='async'
    def __init__(self):
        self.deadline=100.;self.planner=SimpleNamespace();self.root_index=0
        self.observation_contract=None;self.counts={'platform':0}
        self.states=[];self.publications=[];self.decision_costs=[];self.trace=[]
    def observation_evidence(self,history,clock):return {}
    def record(self,row):self.trace.append(row)
    def ride(self,ride,clock,history,world):
        return dict(ride=ride,clock=clock,busy_until=self._controller_available_at)


class World:
    def __init__(self,events):self.events=events
    def event(self,cid,kind):return self.events[cid,kind]


def test_first_query_cannot_retroactively_board_departure(monkeypatch):
    times=iter([0.,6.,6.,6.5]);monkeypatch.setattr(mod,'process_time',lambda:next(times))
    inputs=[]
    def plan(planner,platform,history,entry,clock,seen,**kw):
        inputs.append((clock,set(seen)))
        return command(),{}
    monkeypatch.setattr(mod,'choose_command',plan)
    e=Commands();h=History()
    result=e.platform(mod.PairPlatform(0,{1:'a',2:'b'}),0,h,
        World({(1,'departure'):(5,False,'a'),(2,'departure'):(10,False,'b')}))
    assert result['ride']=='b' and result['clock']==10
    assert inputs==[(0,set()),(6,{1})]
    assert next(r for r in e.trace if r['kind']=='platform' and r['clock']==5)['choice'] is None


def test_prior_command_operates_during_new_computation(monkeypatch):
    times=iter([0.,1.,1.,9.]);monkeypatch.setattr(mod,'process_time',lambda:next(times))
    calls=[]
    def plan(planner,platform,history,entry,clock,seen,**kw):
        calls.append(clock)
        return command(mode='first' if len(calls)==1 else 'wait'),{}
    monkeypatch.setattr(mod,'choose_command',plan)
    e=Commands();h=History([{'receipt_seconds':2}])
    result=e.platform(mod.PairPlatform(0,{1:'a'}),0,h,World({(1,'departure'):(5,False,'a')}))
    assert calls==[0,2]
    assert result==dict(ride='a',clock=5,busy_until=10)
    assert len(e.publications)==2


def test_onboard_alighting_uses_consumption_time_for_transfer_entry():
    class E:
        deadline=100;counts={'onboard':0}
        def record(self,row):pass
        def decision_history(self,h,t):return {}
        def choose_onboard(self,*args):self._last_consumed_clock=12;return True,{}
        def platform(self,node,entry,*args):return entry,'transfer'
    e=E();ride=mod.PairPlatform(0,{})
    node=SimpleNamespace(path=(1,2),exits={1:ride})
    world=World({(1,'departure'):(0,False,'t'),(1,'arrival'):(10,False,'t'),
                 (2,'departure'):(20,False,'t')})
    assert mod.BoundedArmExecutor.ride(e,node,0,History(),world)==(12,'transfer')


def test_busy_controller_cannot_redecide_at_past_arrival():
    class E:
        deadline=100;counts={'onboard':0};_controller_available_at=15
        def __init__(self):self.trace=[]
        def record(self,row):self.trace.append(row)
        def choose_onboard(self,*args):raise AssertionError('No query while busy')
    e=E();node=SimpleNamespace(path=(1,2),exits={1:mod.PairPlatform(0,{}),2:None})
    world=World({(1,'departure'):(0,False,'t'),(1,'arrival'):(10,False,'t'),
                 (2,'departure'):(20,False,'t'),(2,'arrival'):(30,False,'t')})
    assert mod.BoundedArmExecutor.ride(e,node,0,History(),world)==(30,'arrived')
    assert e.trace[0]['kind']=='controller_busy'


def test_late_serial_proposal_cannot_alight_retroactively(monkeypatch):
    times=iter([0.,30.]);monkeypatch.setattr(mod,'process_time',lambda:next(times))
    e=SimpleNamespace(clock_contract='async',publications=[])
    selected,row=mod.BoundedArmExecutor._publish_action(e,opens_at=10,closes_at=20,
        legal=('alight','continue'),fallback='alight',base_cpu=1,
        compute_fn=lambda:'continue',forced_on_total_expiry='continue')
    assert selected=='continue' and row['consumed']['consumed_wall']==20
    assert e._controller_available_at==41 and e._last_consumed_clock==20
    assert row['background_accepted'] is False and row['fallback_published'] is False
    assert row['base_proposal_timely'] is True and row['final_proposal_timely'] is False


def test_message_during_query_triggers_one_coalesced_update(monkeypatch):
    times=iter([0.,4.,4.,5.]);monkeypatch.setattr(mod,'process_time',lambda:next(times))
    inputs=[]
    def plan(planner,platform,history,entry,clock,seen,**kw):
        inputs.append(clock)
        return command(),{}
    monkeypatch.setattr(mod,'choose_command',plan)
    e=Commands();h=History([{'receipt_seconds':1},{'receipt_seconds':2},{'receipt_seconds':3}])
    result=e.platform(mod.PairPlatform(0,{1:'a'}),0,h,World({(1,'departure'):(10,False,'a')}))
    assert inputs==[0,4]
    assert result['clock']==10


def test_snapshot_mandatory_exit_needs_no_score():
    class Engine:
        def onboard(self,*args,**kwargs):raise AssertionError('Final exit is compulsory')
    e=mod.PairSnapshotExecutor(None,None,100.,engine=Engine(),trace=True)
    e.platform=lambda node,entry,history,world:(entry,'transfer')
    node=SimpleNamespace(path=(1,),exits={1:mod.PairPlatform(0,{})})
    world=World({(1,'departure'):(0,False,'t'),(1,'arrival'):(10,False,'t')})
    assert e.ride(node,0,History(),world)==(10,'transfer')
    assert e.counts['onboard']==0
    assert e.trace[-1]['receipt']['method']=='mandatory_menu_exit'
    forced=e.forced(node,dict(kind='onboard',cid=1,clock=10),'continue',History(),world)
    assert forced['executed_root_action']=='alight' and forced['loss']==0


def test_snapshot_zero_dwell_defaults_to_continue_in_suffix_and_forced_root():
    class Engine:
        def onboard(self,*args,**kwargs):raise AssertionError('No positive door interval')
    e=mod.PairSnapshotExecutor(None,None,100.,engine=Engine(),trace=True)
    def forbidden_alight(*args):raise AssertionError('No alighting at closed doors')
    e.platform=forbidden_alight
    node=SimpleNamespace(path=(1,2),exits={1:mod.PairPlatform(0,{}),2:None})
    world=World({(1,'departure'):(0,False,'t'),(1,'arrival'):(10,False,'t'),
                 (2,'departure'):(10,False,'t'),(2,'arrival'):(20,False,'t')})
    assert e.ride(node,0,History(),world)==(20,'arrived')
    assert e.counts['onboard']==0
    forced=e.forced(node,dict(kind='onboard',cid=1,clock=10),'alight',History(),world)
    assert forced['executed_root_action']=='continue' and forced['loss']==10


def test_canceled_departure_is_an_unknown_closing_endpoint_not_zero_dwell():
    from experiments.reconstruction_v34.conditional_reference import ProjectionIncompatible

    class Engine:
        def onboard(self,*args,**kwargs):raise AssertionError('Closing endpoint is unknown')
    e=mod.PairSnapshotExecutor(None,None,100.,engine=Engine(),trace=True)
    def forbidden_alight(*args):raise AssertionError('Cancellation is not a public alight command')
    e.platform=forbidden_alight
    node=SimpleNamespace(path=(1,2),exits={1:mod.PairPlatform(0,{}),2:None})
    world=World({(1,'departure'):(0,False,'t'),(1,'arrival'):(10,False,'t'),
                 (2,'departure'):(20,True,'t')})
    with pytest.raises(ProjectionIncompatible,match='canceled_departure_has_no_door_closing_endpoint'):
        e.ride(node,0,History(),world)
    for action in ('alight','continue'):
        forced=e.forced(node,dict(kind='onboard',cid=1,clock=10),action,History(),world)
        assert forced['loss'] is None and forced['executed_root_action'] is None
        assert 'canceled_departure_has_no_door_closing_endpoint' in forced['status']
    with pytest.raises(mod.MissingEndpoint,match='canceled_departure_has_no_door_closing_endpoint'):
        mod.BoundedArmExecutor.ride(e,node,0,History(),world)
