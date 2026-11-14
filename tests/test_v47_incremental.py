import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Ride
from experiments.reconstruction_v46.small_oracle import evaluate
from experiments.reconstruction_v47.incremental import IncrementalOracle
from src import evolving_trip_belief as evolving
from src.transit_graph import Connection
from test_v46_shared import shared_case
from test_v46_oracle import case


@pytest.mark.parametrize('reverse,forecasts,cancel',[(False,False,0.),(False,True,.05),(True,True,0.)])
def test_retained_event_dp_all_checkpoints_contain_independent_value(monkeypatch,reverse,forecasts,cancel):
    graph,model,params,history,platform,entry=shared_case(monkeypatch,reverse=reverse,forecasts=forecasts,cancel=cancel)
    oracle=IncrementalOracle(graph,model,params,60,history,max_compiles=10000)
    result=oracle.platform(platform,entry,checkpoints=tuple(range(1,100)))
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=entry,clock=entry),history,action='policy',max_evaluations=100000)
    truth=independent['mean']+entry
    assert result['mean']==pytest.approx(truth,abs=1e-8)
    assert result['unresolved_probability']==0
    previous=(-np.inf,np.inf)
    for snap in [*result['checkpoints'],result]:
        value=snap['actions']['policy']
        assert value['lower']<=truth+1e-8 and value['upper']>=truth-1e-8
        assert value['lower']>=previous[0]-1e-8 and value['upper']<=previous[1]+1e-8
        assert -1e-12<=value['unresolved_probability']<=1+1e-12
        previous=value['lower'],value['upper']
    assert result['counters']['unsupported_scores']==0


@pytest.mark.parametrize('forecasts,mct',[(False,0),(True,0),(True,3)])
def test_independent_tail_generic_event_kernel_keeps_original_policy(monkeypatch,forecasts,mct):
    graph,model,params,history,platform=case(monkeypatch,forecasts=forecasts,mct=mct)
    result=IncrementalOracle(graph,model,params,60,history,max_compiles=10000).platform(platform,10)
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=10,clock=10),history,action='policy')
    assert result['mean']==pytest.approx(independent['mean']+10,abs=1e-8)


def test_pauses_do_not_repeat_paid_work(monkeypatch):
    graph,model,params,history,platform,entry=shared_case(monkeypatch,forecasts=True)
    one=IncrementalOracle(graph,model,params,60,history,max_compiles=10000)
    result=one.platform(platform,entry)
    paused=IncrementalOracle(graph,model,params,60,history,max_compiles=10000)
    paused._initialize({'policy':paused._platform_node(platform,entry,paused.snapshot)})
    for quota in (1,2,4,8,16,32,64,128,10000):
        paused._phase('policy',quota,None,'test',(),[])
    resumed=paused._snapshot('final')
    assert resumed['actions']==result['actions']
    assert resumed['counters']['charged_compiles']==result['counters']['charged_compiles']
    assert resumed['counters']['dag_expansions']==result['counters']['dag_expansions']
    counts=paused.stats['charged_compiles']
    paused._phase('policy',10000,None,'already_done',(),[])
    assert paused.stats['charged_compiles']==counts


def test_pooled_returns_unused_continue_quota_to_alight(monkeypatch):
    from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
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
    exact=IncrementalOracle(graph,model,params,60,history,max_compiles=10000).forced(node,row)
    amount=exact['counters']['charged_compiles']
    assert amount>4
    budget=max(4,amount-2)
    fixed=IncrementalOracle(graph,model,params,60,history,max_compiles=budget).forced(node,row,schedule='fixed_split')
    pooled=IncrementalOracle(graph,model,params,60,history,max_compiles=budget).forced(node,row,schedule='pooled')
    assert fixed['unspent_compiles']>0
    assert pooled['recycled_compiles']>0
    assert pooled['counters']['charged_compiles']<=budget
    assert pooled['actions']['continue']==fixed['actions']['continue']
    assert pooled['actions']['alight']['upper']-pooled['actions']['alight']['lower'] < fixed['actions']['alight']['upper']-fixed['actions']['alight']['lower']
    after_c=next(s for s in pooled['checkpoints'] if s['phase']=='after_continue')
    assert after_c['actions']==fixed['actions']
    for arm in ('alight','continue'):
        independent=evaluate(graph,model,params,60,node,row,history,action=arm,max_evaluations=100000)
        truth=independent['mean']+9
        assert exact['actions'][arm]['mean']==pytest.approx(truth,abs=1e-8)
        for result in (fixed,pooled):
            assert result['actions'][arm]['lower']<=truth+1e-8
            assert result['actions'][arm]['upper']>=truth-1e-8


def test_zero_budget_and_scoring_only_post_goal_menu(monkeypatch):
    graph,model,params,history,platform=case(monkeypatch,forecasts=True)
    old=platform.rides[0]
    platform.rides[0]=Ride(0,(0,1),{0:None,1:old.exits[0]})
    empty=IncrementalOracle(graph,model,params,60,history,max_compiles=0).platform(platform,10)
    assert empty['lower']==10 and empty['upper']==60 and empty['mean'] is None
    assert empty['counters']['charged_compiles']==0
    full=IncrementalOracle(graph,model,params,60,history,max_compiles=10000).platform(platform,10)
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=10,clock=10),history,action='policy')
    assert full['mean']==pytest.approx(independent['mean']+10,abs=1e-8)


def test_shared_dependencies_are_paid_once_and_mixture_mass_is_not_dropped(monkeypatch):
    graph,model,params,history,platform,entry=shared_case(monkeypatch,forecasts=False)
    oracle=IncrementalOracle(graph,model,params,60,history,max_compiles=10000)
    common=oracle._platform_node(platform,entry,oracle.snapshot)
    # Two branches and both action roots can share one conditional continuation.
    mixture=oracle._mixture(('test_mixture',),entry,[(.4,common),(.6,common)])
    oracle._initialize({'alight':common,'continue':mixture})
    oracle._phase('alight',10000,None,'first',(),[])
    charges=oracle.stats['charged_compiles']
    oracle._phase('continue',10000,None,'shared',(),[])
    values=oracle._values()
    assert oracle.stats['charged_compiles']==charges
    assert values['continue']['mean']==pytest.approx(values['alight']['mean'],abs=1e-9)
    assert values['continue']['unresolved_probability']==0
    assert common.drained and mixture.drained
