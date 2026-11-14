import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Ride
from experiments.reconstruction_v46.oracle import FullPolicyOracle, structural, terminal
from experiments.reconstruction_v46.shared_oracle import SharedObservationOracle
from experiments.reconstruction_v46.small_oracle import evaluate
from experiments.reconstruction_v47.bulk import BulkIncrementalOracle
from test_v46_oracle import case
from test_v46_shared import shared_case


@pytest.mark.parametrize('forecasts,mct,cancel',[(False,0,0.),(True,0,.1),(True,3,0.)])
def test_full_mixed_bulk_matches_v46_and_original_executor_at_every_checkpoint(monkeypatch,forecasts,mct,cancel):
    graph,model,params,history,platform=case(monkeypatch,forecasts=forecasts,mct=mct,cancel=cancel)
    assert structural(model,platform)['eligible']
    assert not all(terminal(ride) for ride in platform.rides.values())
    oracle=BulkIncrementalOracle(graph,model,params,60,history,max_compiles=10000)
    result=oracle.platform(platform,10,checkpoints=tuple(range(1,200)))
    old=FullPolicyOracle(graph,model,params,60,history).platform(platform,10)
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=10,clock=10),history,action='policy')
    truth=independent['mean']+10
    assert result['mean']==pytest.approx(truth,abs=1e-8)
    assert result['mean']==pytest.approx(old['mean'],abs=1e-8)
    assert result['counters']['bulk_mixed_platforms']==1
    assert result['counters']['bulk_terminal_platforms']>0
    previous=(-np.inf,np.inf)
    for snap in [*result['checkpoints'],result]:
        value=snap['actions']['policy']
        assert value['lower']<=truth+1e-8 and value['upper']>=truth-1e-8
        assert value['lower']>=previous[0]-1e-8 and value['upper']<=previous[1]+1e-8
        assert -1e-12<=value['unresolved_probability']<=1+1e-12
        previous=value['lower'],value['upper']


@pytest.mark.parametrize('reverse,forecasts,cancel',[(False,False,0.),(False,True,.05),(True,True,0.)])
def test_shared_root_and_reverse_conditioning_still_use_original_policy(monkeypatch,reverse,forecasts,cancel):
    graph,model,params,history,platform,entry=shared_case(monkeypatch,reverse=reverse,forecasts=forecasts,cancel=cancel)
    assert not structural(model,platform)['eligible']
    result=BulkIncrementalOracle(graph,model,params,60,history,max_compiles=10000).platform(platform,entry)
    old=SharedObservationOracle(graph,model,params,60,history).platform(platform,entry)
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=entry,clock=entry),history,action='policy',max_evaluations=100000)
    assert result['mean']==pytest.approx(old['mean'],abs=1e-8)
    assert result['mean']==pytest.approx(independent['mean']+entry,abs=1e-8)
    assert result['counters']['shared_policy_compiles']>0


def test_mixed_bulk_pauses_keep_partial_physics_and_board_work(monkeypatch):
    graph,model,params,history,platform=case(monkeypatch,forecasts=True)
    uninterrupted=BulkIncrementalOracle(graph,model,params,60,history,max_compiles=10000)
    full=uninterrupted.platform(platform,10)
    paused=BulkIncrementalOracle(graph,model,params,60,history,max_compiles=10000)
    paused._initialize({'policy':paused._platform_node(platform,10,paused.snapshot)})
    for count in range(full['counters']['charged_compiles']+2):
        paused._phase('policy',count,None,'test',(),[])
        interval=paused._values()['policy']
        assert interval['lower']<=full['mean']+1e-8 and interval['upper']>=full['mean']-1e-8
    assert paused._values()==full['actions']
    assert paused.stats['charged_compiles']==full['counters']['charged_compiles']
    assert paused.stats['onboard_compiles']==full['counters']['onboard_compiles']
    assert paused.stats['board_compiles']==full['counters']['board_compiles']


def test_terminal_and_post_goal_scoring_dependencies_remain_in_scope(monkeypatch):
    graph,model,params,history,platform=case(monkeypatch,forecasts=True)
    old=platform.rides[0]
    platform.rides[0]=Ride(0,(0,1),{0:None,1:old.exits[0]})
    result=BulkIncrementalOracle(graph,model,params,60,history,max_compiles=10000).platform(platform,10)
    truth=FullPolicyOracle(graph,model,params,60,history).platform(platform,10)
    assert result['mean']==pytest.approx(truth['mean'],abs=1e-8)
    assert result['counters']['bulk_terminal_platforms']==1
    graph,model,params,history,platform,entry=shared_case(monkeypatch)
    y=platform.rides[2]
    y.exits={2:None,3:y.exits[2]}
    assert not structural(model,platform)['eligible']
    result=BulkIncrementalOracle(graph,model,params,60,history,max_compiles=10000).platform(platform,entry)
    assert result['counters']['bulk_platforms']==0
    assert result['counters']['shared_policy_compiles']>0


def test_forced_mixed_alight_recycling_keeps_both_action_expectations(monkeypatch):
    from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
    from src import evolving_trip_belief as evolving
    from src.transit_graph import Connection
    graph,_,params,old_history,platform=case(monkeypatch,forecasts=False)
    graph.add_connection(Connection(4,'R','H',3,0,1,9))
    graph.add_connection(Connection(5,'R','H',0,3,10,50))
    model=evolving.EvolvingTripBelief(graph,public_occurrences=False,
        travel_sd=.05,dwell_sd=.05,cancel_persistence=.8)
    history=PhysicalHistory(model,params,fit=old_history.fit)
    history.observe(4,'departure',1)
    history.observe(4,'arrival',9)
    node=Ride(4,(4,5),{4:platform,5:None})
    row=dict(kind='onboard',cid=4,clock=9.)
    full=BulkIncrementalOracle(graph,model,params,60,history,max_compiles=10000).forced(node,row)
    cap=full['counters']['charged_compiles']-2
    fixed=BulkIncrementalOracle(graph,model,params,60,history,max_compiles=cap).forced(node,row,schedule='fixed_split')
    pooled=BulkIncrementalOracle(graph,model,params,60,history,max_compiles=cap).forced(node,row,schedule='pooled')
    assert pooled['recycled_compiles']>0 and pooled['counters']['charged_compiles']==cap
    assert next(r for r in pooled['checkpoints'] if r['phase']=='after_continue')['actions']==fixed['actions']
    assert pooled['actions']['continue']==fixed['actions']['continue']
    for arm in ('alight','continue'):
        independent=evaluate(graph,model,params,60,node,row,history,action=arm,max_evaluations=100000)
        truth=independent['mean']+9
        assert full['actions'][arm]['mean']==pytest.approx(truth,abs=1e-8)
        assert pooled['actions'][arm]['lower']<=truth+1e-8 and pooled['actions'][arm]['upper']>=truth-1e-8
