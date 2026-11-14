from itertools import product

import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform,Ride
from experiments.reconstruction_v39.executor import SnapshotExecutor
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine,PhysicalHistory
from experiments.reconstruction_v46.oracle import FullPolicyOracle,structural
from src import evolving_trip_belief as evolving
from src.transit_graph import Connection,Stop,TransitGraph


def case(monkeypatch,*,cancel=0.,forecasts=False,mct=0):
    monkeypatch.setattr(evolving,'DELAYS',np.array([0.,.05]))
    graph=TransitGraph()
    for stop in range(4):graph.add_stop(Stop(stop,str(stop),min_transfer_time=mct if stop==0 else 0))
    for cid,trip,origin,target,dep,arr in (
            (0,'A',0,1,10,20),(1,'A',1,3,22,40),
            (2,'B',0,3,10,35),(3,'C',1,3,25,30)):
        graph.add_connection(Connection(cid,'R',trip,origin,target,dep,arr))
    model=evolving.EvolvingTripBelief(graph,public_occurrences=False,
        travel_sd=.05,dwell_sd=.05,cancel_persistence=.8)
    params={'R':dict(mean=0.,std=.05,cancel_rate=cancel)}
    fit=dict(route_kind={},pooled={kind:dict(bias_seconds=0.,sd_seconds=3.) for kind in ('departure','arrival')})
    history=PhysicalHistory(model,params,fit=fit)
    if forecasts:
        history.apply_frame(dict(receipt_seconds=1.,trip_statuses=[],endpoint_updates=[
            [1,'arrival','FORECAST',55.],[2,'arrival','FORECAST',45.]]),1.)
        history.apply_frame(dict(receipt_seconds=2.,trip_statuses=[],endpoint_updates=[
            [0,'departure','FORECAST',10.],[2,'departure','FORECAST',10.]]),2.)
    tail=Platform(1,{3:Ride(3,(3,),{3:None})})
    platform=Platform(0,{0:Ride(0,(0,1),{0:tail,1:None}),2:Ride(2,(2,),{2:None})})
    return graph,model,params,history,platform


def joint_paths(law):
    states=[np.flatnonzero(law.marginal(slot)>0) for slot in range(len(law.times))]
    rows=[]
    for path in product(*states):
        logp=law.initial[path[0]]+sum(e[s] for e,s in zip(law.emissions,path))
        logp+=sum(law._transition(j)[path[j],path[j+1]] for j in range(len(path)-1))
        if np.isfinite(logp):rows.append((path,float(np.exp(logp))))
    total=sum(p for _,p in rows)
    return [(path,p/total) for path,p in rows]


def enumerate_platform(graph,model,params,history,platform,entry,deadline):
    """Full joint vehicle paths, then unmodified actual policy execution."""
    trips=sorted(platform.trips(model))
    laws={trip:history.law(trip) for trip in trips}
    rows={trip:joint_paths(laws[trip]) for trip in trips}
    planner=MatchedContinuationPlanner(graph,model,params,deadline)
    engine=PhysicalDecisionEngine(planner,variant='frontier')
    total,mass=0.,0.
    choices=set()
    for paths in product(*(rows[t] for t in trips)):
        selection={trip:row[0] for trip,row in zip(trips,paths)}
        probability=np.prod([row[1] for row in paths])
        class World:
            def event(self,cid,kind):
                trip,slot=model._positions[cid,kind];state=selection[trip][slot];law=laws[trip]
                return int(law.times[slot][state]),bool(law.canceled[slot][state]),trip
        executor=SnapshotExecutor(graph,planner,deadline,engine=engine,method='rolling_fixed',waiting='combined')
        value,status=executor.platform(platform,entry,history.clone(),World())
        mass+=probability;total+=probability*value
        choices.add(tuple((r['kind'],r.get('cid'),r['clock'],r['choice']) for r in executor.trace))
    return total,mass,choices


@pytest.mark.parametrize('forecasts,mct',[(False,0),(True,0),(True,3)])
def test_mixed_full_pi_matches_independent_joint_paths(monkeypatch,forecasts,mct):
    graph,model,params,history,platform=case(monkeypatch,forecasts=forecasts,mct=mct)
    oracle=FullPolicyOracle(graph,model,params,60,history)
    actual=oracle.platform(platform,10)
    expected,mass,choices=enumerate_platform(graph,model,params,history,platform,10,60)
    assert mass==pytest.approx(1.,abs=1e-10)
    assert actual['mean']==pytest.approx(expected,abs=1e-9)
    assert actual['unresolved_probability']==0
    assert len(choices)>1
    assert oracle.stats['board_compiles']>0 and oracle.stats['onboard_compiles']>0


def test_terminal_true_marks_include_intermediate_failure_and_deadline(monkeypatch):
    graph,model,params,history,platform=case(monkeypatch,cancel=.1)
    # A's two physical segments have no decision here, but can fail mid-ride.
    pure=Platform(0,{0:Ride(0,(0,1),{1:None})})
    oracle=FullPolicyOracle(graph,model,params,42,history)
    actual=oracle.platform(pure,10)
    expected,mass,_=enumerate_platform(graph,model,params,history,pure,10,42)
    assert mass==pytest.approx(1.,abs=1e-10)
    assert actual['mean']==pytest.approx(expected,abs=1e-9)
    assert 40<actual['mean']<=42


def test_budget_returns_all_mass_bounds_instead_of_an_exact_value(monkeypatch):
    graph,model,params,history,platform=case(monkeypatch,forecasts=True)
    complete=FullPolicyOracle(graph,model,params,60,history).platform(platform,10)
    bounded=FullPolicyOracle(graph,model,params,60,history,max_compiles=4).platform(platform,10)
    assert bounded['mean'] is None
    assert bounded['lower']<=complete['mean']<=bounded['upper']
    assert 0<bounded['unresolved_probability']<=1+1e-12
    assert bounded['counters']['charged_compiles']==4


def test_cross_candidate_reference_is_not_silently_treated_independent(monkeypatch):
    graph,model,params,history,platform=case(monkeypatch)
    platform.rides[0].exits[0]=Platform(1,{2:platform.rides[2]})
    check=structural(model,platform)
    assert not check['eligible']
    result=FullPolicyOracle(graph,model,params,60,history).platform(platform,10)
    assert result['mean'] is None and result['lower']==10 and result['upper']==60
    assert 'root_trip_reappears' in result['reason']


def test_wait_and_board_reselection_use_different_marks(monkeypatch):
    graph,model,params,history,platform=case(monkeypatch,forecasts=True)
    oracle=FullPolicyOracle(graph,model,params,60,history)
    oracle._prepare(platform)
    reference=oracle.planner.platform_reference(platform,oracle.snapshot,10,10,'rolling_fixed')
    wait=reference.score(2,10)
    board=oracle._board(platform.rides[2],10,oracle.snapshot)
    _,truth=oracle.ride_marks(platform.rides[2])
    assert abs(wait-board)>1e-3
    assert np.max(abs(truth[:,0]-board))>1e-3


def test_forced_both_actions_and_reserved_budget(monkeypatch):
    from experiments.reconstruction_v46.small_oracle import evaluate
    graph,model,params,history,platform=case(monkeypatch,forecasts=True)
    history.observe(0,'departure',10)
    history.observe(0,'arrival',20)
    row=dict(kind='onboard',cid=0,clock=20)
    ride=platform.rides[0]
    exact=FullPolicyOracle(graph,model,params,60,history).forced(ride,row)
    for arm in ('alight','continue'):
        independent=evaluate(graph,model,params,60,ride,row,history,action=arm)
        assert exact['actions'][arm]['mean']==pytest.approx(independent['mean']+20,abs=1e-9)
    bounded=FullPolicyOracle(graph,model,params,60,history,max_compiles=4).forced(ride,row)
    assert bounded['budget_allocation']['alight']['charged_compiles']<=2
    assert bounded['counters']['charged_compiles']<=4
    assert bounded['actions']['continue']['mean'] is not None
    for arm in ('alight','continue'):
        assert bounded['actions'][arm]['lower']<=exact['actions'][arm]['mean']<=bounded['actions'][arm]['upper']


def test_simultaneous_equal_board_scores_integrate_the_ranked_vehicle(monkeypatch):
    from dataclasses import replace
    graph,model,params,history,platform=case(monkeypatch)
    monkeypatch.setattr(evolving,'DELAYS',np.array([0.]))
    original=MatchedContinuationPlanner.compile_ride
    def tied(self,ride,h,method='optimal'):
        compiled=original(self,ride,h,method)
        return replace(compiled,board_scores=np.full_like(compiled.board_scores,40.))
    monkeypatch.setattr(MatchedContinuationPlanner,'compile_ride',tied)
    monkeypatch.setattr(MatchedContinuationPlanner,'ranks',
        lambda self,p:{cid:(0 if cid==2 else 1,cid) for cid in p.rides})
    oracle=FullPolicyOracle(graph,model,params,60,history)
    actual=oracle.platform(platform,10)
    expected,mass,choices=enumerate_platform(graph,model,params,history,platform,10,60)
    assert mass==pytest.approx(1.)
    assert actual['mean']==pytest.approx(expected)
    # B has priority when both visible scores are 40; its true arrival is 35.
    # Choosing the lower true mark instead would wrongly take A then C at 30.
    assert expected==pytest.approx(35.)
    assert any(trace[0][-1]==2 for trace in choices)


def test_mixed_early_destination_stops_physics_but_preserves_scoring_menu(monkeypatch):
    from experiments.reconstruction_v46.small_oracle import evaluate
    graph,model,params,history,platform=case(monkeypatch,forecasts=True)
    old=platform.rides[0]
    platform.rides[0]=Ride(0,(0,1),{0:None,1:old.exits[0]})
    assert structural(model,platform)['eligible']
    seen=[];original=MatchedContinuationPlanner.compile_ride
    def recorded(self,ride,h,method='optimal'):
        if ride.first_cid==0:seen.append((ride.path,tuple(ride.exits)))
        return original(self,ride,h,method)
    monkeypatch.setattr(MatchedContinuationPlanner,'compile_ride',recorded)
    actual=FullPolicyOracle(graph,model,params,60,history).platform(platform,10)
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=10,clock=10),history,action='policy')
    assert actual['mean']==pytest.approx(independent['mean']+10,abs=1e-9)
    assert seen and all(path==(0,1) and exits==(0,1) for path,exits in seen)
    assert actual['counters']['onboard_compiles']==0


def test_terminal_leaf_may_retain_scoring_only_exits_after_destination(monkeypatch):
    from experiments.reconstruction_v46.small_oracle import evaluate
    graph,model,params,history,platform=case(monkeypatch,forecasts=True)
    # C has a physical destination, followed by an unused alternative in its
    # same-vehicle path. It remains in W's original scoring menu.
    graph.add_connection(Connection(4,'R','C',3,1,32,45))
    # Rebuild the finite model now that its declared graph has another slot.
    model=evolving.EvolvingTripBelief(graph,public_occurrences=False,
        travel_sd=.05,dwell_sd=.05,cancel_persistence=.8)
    history=PhysicalHistory(model,params,fit=history.fit)
    leaf=platform.rides[0].exits[0]
    leaf.rides[3]=Ride(3,(3,4),{3:None,4:Platform(1,{2:platform.rides[2]})})
    # A's tail now contains current root B through a scoring-only dependency;
    # this still violates independence and must not silently be erased.
    assert not structural(model,platform)['eligible']
    assert structural(model,leaf)['eligible']
    actual=FullPolicyOracle(graph,model,params,60,history).platform(leaf,20)
    independent=evaluate(graph,model,params,60,leaf,
        dict(kind='platform_start',entry=20,clock=20),history,action='policy')
    assert actual['mean']==pytest.approx(independent['mean']+20,abs=1e-9)
