import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform,Ride
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v46.shared_oracle import SharedObservationOracle,shared_structure
from experiments.reconstruction_v46.small_oracle import evaluate
from src import evolving_trip_belief as evolving
from src.transit_graph import Connection,Stop,TransitGraph


def shared_case(monkeypatch,*,reverse=False,forecasts=True,cancel=0.):
    monkeypatch.setattr(evolving,'DELAYS',np.array([0.,8/60]))
    graph=TransitGraph()
    for stop in range(3):graph.add_stop(Stop(stop,str(stop),min_transfer_time=0))
    x0=(1,0) if reverse else (0,1)
    x1=(0,2) if reverse else (1,2)
    ydep,yarr=(15,17) if reverse else (11,13)
    for cid,trip,origin,target,dep,arr in (
            (0,'X',*x0,10,12),(1,'X',*x1,14,20),
            (2,'Y',0,1,ydep,yarr),(3,'Y',1,2,19 if reverse else 17,30),
            (4,'X',2,2,22,35)):
        graph.add_connection(Connection(cid,'R',trip,origin,target,dep,arr))
    model=evolving.EvolvingTripBelief(graph,public_occurrences=False,
        travel_sd=.1,dwell_sd=.1,cancel_persistence=.8)
    params={'R':dict(mean=0.,std=.1,cancel_rate=cancel)}
    fit=dict(route_kind={},pooled={kind:dict(bias_seconds=0.,sd_seconds=5.) for kind in ('departure','arrival')})
    history=PhysicalHistory(model,params,fit=fit)
    root_cid=1 if reverse else 0
    if forecasts:
        history.apply_frame(dict(receipt_seconds=1.,trip_statuses=[],endpoint_updates=[
            [4,'arrival','FORECAST',40.],[3,'arrival','FORECAST',35.]]),1.)
        history.apply_frame(dict(receipt_seconds=2.,trip_statuses=[],endpoint_updates=[
            [root_cid,'departure','FORECAST',14. if reverse else 10.],
            [2,'departure','FORECAST',ydep]]),2.)
    xride=Ride(root_cid,(1,4) if reverse else (0,1,4),{4:None})
    leaf=Ride(0,(0,1),{1:None}) if reverse else Ride(1,(1,),{1:None})
    tail=Platform(1,{leaf.first_cid:leaf})
    yride=Ride(2,(2,3),{2:tail,3:None})
    platform=Platform(0,{root_cid:xride,2:yride})
    return graph,model,params,history,platform,13. if reverse else 9.


@pytest.mark.parametrize('reverse,forecasts,cancel',[(False,False,0.),(False,True,.05),(True,True,0.)])
def test_shared_event_dp_matches_private_full_policy_enumeration(monkeypatch,reverse,forecasts,cancel):
    graph,model,params,history,platform,entry=shared_case(monkeypatch,reverse=reverse,forecasts=forecasts,cancel=cancel)
    check=shared_structure(model,platform)
    assert check['eligible'] and len(check['interface_cids'])==1
    oracle=SharedObservationOracle(graph,model,params,60,history)
    actual=oracle.platform(platform,entry)
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=entry,clock=entry),history,action='policy',max_evaluations=100000)
    assert independent['unresolved_probability']==0
    assert actual['mean']==pytest.approx(independent['mean']+entry,abs=1e-8)
    assert actual['unresolved_probability']==0
    assert actual['counters']['shared_policy_compiles']>0


def test_exact_shared_time_cannot_be_replaced_with_seen_mask(monkeypatch):
    graph,model,params,history,platform,entry=shared_case(monkeypatch,forecasts=False)
    oracle=SharedObservationOracle(graph,model,params,60,history)
    oracle._prepare(platform)
    left=oracle.event_history(platform,oracle.snapshot,entry,19,set(),(2,),((0,10.),))
    right=oracle.event_history(platform,oracle.snapshot,entry,19,set(),(2,),((0,18.),))
    # The same local clock and seen mask leave different P0 distributions of
    # the shared X at the later transfer; the exact previous receipt matters.
    a=oracle.root_law.departure_distribution(1,left)
    b=oracle.root_law.departure_distribution(1,right)
    assert not np.allclose(a[1],b[1])
    left.observe(2,'arrival',21.)
    right.observe(2,'arrival',21.)
    tail=platform.rides[2].exits[2]
    va=oracle.platform(tail,21,left)['mean']
    vb=oracle.platform(tail,21,right)['mean']
    assert abs(va-vb)>1e-3


def test_reverse_slot_uses_root_joint_conditioning_not_forward_prior(monkeypatch):
    graph,model,params,history,platform,entry=shared_case(monkeypatch,reverse=True)
    oracle=SharedObservationOracle(graph,model,params,60,history)
    oracle._prepare(platform)
    event=oracle.snapshot.clone();event.observe(1,'departure',14)
    times,p,_=oracle.root_law.departure_distribution(0,event)
    assert p[times>14].sum()==0
    prior=oracle.root_law.departure_distribution(0,oracle.snapshot)
    assert prior[1][prior[0]>14].sum()>0
    tail=platform.rides[2].exits[2]
    result=oracle.platform(tail,17,event)
    assert result['mean']==pytest.approx(60.)


@pytest.mark.parametrize('reverse',[False,True])
def test_one_backward_table_preserves_each_selected_departure(monkeypatch,reverse):
    graph,model,params,history,platform,entry=shared_case(monkeypatch,reverse=reverse)
    oracle=SharedObservationOracle(graph,model,params,60,history)
    oracle._prepare(platform)
    # Keep a legal shared observation while observing each possible Y root D.
    xcid=1 if reverse else 0
    xclock=14. if reverse else 10.
    times,p,_=oracle.root_law.departure_distribution(2,oracle.snapshot)
    for departure in times[p>0]:
        event=oracle.snapshot.clone();event.observe(xcid,'departure',xclock)
        event.observe(2,'departure',float(departure))
        reused=oracle.boarding_value(platform.rides[2],float(departure),oracle.snapshot,event)
        explicit_times,explicit=oracle.ride_marks(platform.rides[2],event)
        law=oracle.root_law.law('Y',event);slot=model._positions[2,'departure'][1]
        index=np.flatnonzero((explicit_times==departure)&~law.canceled[slot])[0]
        assert reused==pytest.approx(explicit[index],abs=1e-9)


def test_shared_budget_keeps_mass_bounds_and_prior_seen_ids(monkeypatch):
    graph,model,params,history,platform,entry=shared_case(monkeypatch)
    exact=SharedObservationOracle(graph,model,params,60,history).platform(platform,entry)
    bounded=SharedObservationOracle(graph,model,params,60,history,max_compiles=4).platform(platform,entry)
    assert bounded['mean'] is None
    assert bounded['lower']<=exact['mean']<=bounded['upper']
    assert 0<bounded['unresolved_probability']<=1+1e-10
    assert bounded['counters']['charged_compiles']<=4


def test_self_trip_tail_and_scoring_only_shared_interface_are_retained(monkeypatch):
    graph,model,params,history,platform,entry=shared_case(monkeypatch)
    own=platform.rides[0]
    own.exits[0]=Platform(1,{1:Ride(1,(1,),{1:None})})
    assert not shared_structure(model,platform)['eligible']
    graph,model,params,history,platform,entry=shared_case(monkeypatch)
    y=platform.rides[2]
    y.exits={2:None,3:y.exits[2]}
    check=shared_structure(model,platform)
    assert check['eligible'] and check['interface_cids']==[0]
    assert not check['references'][0]['physical_reachable']
