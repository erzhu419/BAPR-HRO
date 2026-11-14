from dataclasses import replace

import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform,Ride
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v46.oracle import FullPolicyOracle, structural
from experiments.reconstruction_v46.shared_oracle import SharedObservationOracle
from experiments.reconstruction_v46.small_oracle import evaluate
from experiments.reconstruction_v47.hybrid import HybridIncrementalOracle
from test_v46_oracle import case
from test_v46_shared import shared_case
from src import evolving_trip_belief as evolving


@pytest.mark.parametrize('forecasts,mct,cancel',[(False,0,0.),(True,0,.1),(True,3,0.)])
def test_bulk_terminal_checkpoints_match_original_policy_and_v46(monkeypatch,forecasts,mct,cancel):
    graph,model,params,history,platform=case(monkeypatch,forecasts=forecasts,mct=mct,cancel=cancel)
    platform.rides[0]=Ride(0,(0,1),{1:None})
    old=FullPolicyOracle(graph,model,params,60,history).platform(platform,10)
    oracle=HybridIncrementalOracle(graph,model,params,60,history,max_compiles=10000)
    result=oracle.platform(platform,10,checkpoints=tuple(range(1,100)))
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=10,clock=10),history,action='policy')
    truth=independent['mean']+10
    assert old['mean']==pytest.approx(truth,abs=1e-8)
    assert result['mean']==pytest.approx(truth,abs=1e-8)
    assert result['counters']['hybrid_terminal_platforms']==1
    assert result['counters']['onboard_compiles']==0
    assert result['counters']['charged_compiles']==old['counters']['charged_compiles']
    before=(-np.inf,np.inf)
    for snap in [*result['checkpoints'],result]:
        value=snap['actions']['policy']
        assert value['lower']>=before[0]-1e-8 and value['upper']<=before[1]+1e-8
        assert value['lower']<=truth+1e-8 and value['upper']>=truth-1e-8
        before=value['lower'],value['upper']
    # Suspensions inside the BOARD list retain every compiled score exactly once.
    resumed=HybridIncrementalOracle(graph,model,params,60,history,max_compiles=10000)
    resumed._initialize({'policy':resumed._platform_node(platform,10,resumed.snapshot)})
    for cap in range(1,result['counters']['charged_compiles']+2):
        resumed._phase('policy',cap,None,'test',(),[])
    assert resumed._values()==result['actions']
    assert resumed.stats['charged_compiles']==result['counters']['charged_compiles']


@pytest.mark.parametrize('reverse,forecasts,cancel',[(False,False,0.),(False,True,.05),(True,True,0.)])
def test_shared_root_keeps_event_interface_and_bulk_terminal_tail(monkeypatch,reverse,forecasts,cancel):
    graph,model,params,history,platform,entry=shared_case(monkeypatch,reverse=reverse,forecasts=forecasts,cancel=cancel)
    result=HybridIncrementalOracle(graph,model,params,60,history,max_compiles=10000).platform(
        platform,entry,checkpoints=tuple(range(1,100)))
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=entry,clock=entry),history,action='policy',max_evaluations=100000)
    truth=independent['mean']+entry
    assert result['mean']==pytest.approx(truth,abs=1e-8)
    assert result['counters']['shared_policy_compiles']>0
    if not reverse:  # The reverse fixture's original policy never takes its missed tail.
        assert result['counters']['hybrid_terminal_platforms']>0
    for snap in result['checkpoints']:
        value=snap['actions']['policy']
        assert value['lower']<=truth+1e-8 and value['upper']>=truth-1e-8


def test_bulk_wait_board_and_truth_stay_distinct_and_post_goal_menu_stays(monkeypatch):
    graph,model,params,history,platform=case(monkeypatch,forecasts=True)
    old=platform.rides[0]
    platform.rides[0]=Ride(0,(0,1),{0:None,1:old.exits[0]})
    assert structural(model,platform)['eligible']
    observed=[]
    original=MatchedContinuationPlanner.compile_ride
    def record(self,ride,h,method='optimal'):
        if ride.first_cid==0:observed.append((ride.path,tuple(ride.exits)))
        return original(self,ride,h,method)
    monkeypatch.setattr(MatchedContinuationPlanner,'compile_ride',record)
    result=HybridIncrementalOracle(graph,model,params,60,history,max_compiles=10000).platform(platform,10)
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=10,clock=10),history,action='policy')
    assert result['mean']==pytest.approx(independent['mean']+10,abs=1e-8)
    assert result['counters']['hybrid_terminal_platforms']==1
    assert observed and all(path==(0,1) and exits==(0,1) for path,exits in observed)


def test_scoring_only_shared_root_blocks_bulk_admission(monkeypatch):
    graph,model,params,history,platform,entry=shared_case(monkeypatch)
    y=platform.rides[2]
    y.exits={2:None,3:y.exits[2]}
    assert not structural(model,platform)['eligible']
    result=HybridIncrementalOracle(graph,model,params,60,history,max_compiles=10000).platform(platform,entry)
    old=SharedObservationOracle(graph,model,params,60,history).platform(platform,entry)
    assert result['mean']==pytest.approx(old['mean'],abs=1e-8)
    assert result['counters']['shared_policy_compiles']>0
    assert result['counters']['hybrid_terminal_platforms']==0


def test_unresolved_simultaneous_board_never_selects_by_physical_truth(monkeypatch):
    graph,model,params,history,platform=case(monkeypatch)
    monkeypatch.setattr(evolving,'DELAYS',np.array([0.]))
    platform.rides[0]=Ride(0,(0,1),{0:None})
    original=MatchedContinuationPlanner.compile_ride
    def tied(self,ride,h,method='optimal'):
        compiled=original(self,ride,h,method)
        return replace(compiled,board_scores=np.full_like(compiled.board_scores,40.))
    monkeypatch.setattr(MatchedContinuationPlanner,'compile_ride',tied)
    monkeypatch.setattr(MatchedContinuationPlanner,'ranks',
        lambda self,p:{cid:(0 if cid==2 else 1,cid) for cid in p.rides})
    bounded=HybridIncrementalOracle(graph,model,params,60,history,max_compiles=3).platform(platform,10)
    assert bounded['mean'] is None
    assert bounded['lower']<=35<=bounded['upper']
    exact=HybridIncrementalOracle(graph,model,params,60,history,max_compiles=10000).platform(platform,10)
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=10,clock=10),history,action='policy')
    assert exact['mean']==pytest.approx(35.)
    assert exact['mean']==pytest.approx(independent['mean']+10,abs=1e-8)


def test_hybrid_recycling_preserves_prefix_and_completed_continue(monkeypatch):
    from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
    from src.transit_graph import Connection
    graph,_,params,old_history,platform,entry=shared_case(monkeypatch,forecasts=False)
    graph.add_connection(Connection(5,'R','H',2,0,1,9))
    graph.add_connection(Connection(6,'R','H',0,2,10,50))
    model=evolving.EvolvingTripBelief(graph,public_occurrences=False,
        travel_sd=.1,dwell_sd=.1,cancel_persistence=.8)
    history=PhysicalHistory(model,params,fit=old_history.fit)
    history.observe(5,'departure',1)
    history.observe(5,'arrival',9)
    node=Ride(5,(5,6),{5:platform,6:None})
    row=dict(kind='onboard',cid=5,clock=9.)
    full=HybridIncrementalOracle(graph,model,params,60,history,max_compiles=10000).forced(node,row)
    cap=full['counters']['charged_compiles']-2
    fixed=HybridIncrementalOracle(graph,model,params,60,history,max_compiles=cap).forced(node,row,schedule='fixed_split')
    pooled=HybridIncrementalOracle(graph,model,params,60,history,max_compiles=cap).forced(node,row,schedule='pooled')
    assert fixed['unspent_compiles']>0 and pooled['recycled_compiles']>0
    assert pooled['counters']['charged_compiles']==cap
    assert next(r for r in pooled['checkpoints'] if r['phase']=='after_continue')['actions']==fixed['actions']
    assert pooled['actions']['continue']==fixed['actions']['continue']
    assert pooled['actions']['alight']['unresolved_probability']<fixed['actions']['alight']['unresolved_probability']
    for arm in ('alight','continue'):
        truth=full['actions'][arm]['mean']
        assert pooled['actions'][arm]['lower']<=truth+1e-8
        assert pooled['actions'][arm]['upper']>=truth-1e-8
