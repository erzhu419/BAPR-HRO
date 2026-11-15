"""Causal equivalence of projected events and integrated ride suffixes."""
import json

import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Ride
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v46.small_oracle import evaluate
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v52.reference import CausalReference
from experiments.reconstruction_v53.reference import CompressedReference
from src.evolving_trip_belief import EvolvingTripBelief
from src.transit_graph import Connection, Stop
from test_v46_shared import shared_case
from test_v52_reference import frozen_chain, information_case
from test_v53_relevance import reference_with_nuisance


def compressed(base, *, history=None, max_states=None):
    reference=CompressedReference(base.graph,base.model,base.parameters,base.deadline,
        base.snapshot if history is None else history,base.observations,
        max_states=max_states,max_process_cpu=None,keep_decisions=True)
    reference.root_law.roots.update(base.root_law.roots)
    return reference


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('mct',[0,11])
def test_all_policies_preserve_information_ready_and_full_contract(mode,mct):
    base,platform=reference_with_nuisance(False,mct)
    new=compressed(base)
    row=dict(kind='platform_start',clock=0.,entry=0.)
    old=base.solve(platform,row,mode=mode)
    actual=new.solve(platform,row,mode=mode)
    assert old['value']['complete'] and actual['value']['complete']
    assert actual['value']['mean']==pytest.approx(old['value']['mean'],abs=1e-8)
    assert actual['observation_contract']==old['observation_contract']
    assert actual['future_public_frames']==0
    assert actual['counters']['states']<old['counters']['states']
    assert all(d['clock']>=mct for d in actual['decisions'])
    def choices(result):
        return {(d['clock'],tuple(d.get('visible',())),str(d['selected']),d['optimized'],
                 json.dumps([r for r in d['public_history'] if r['trip']!='C'],sort_keys=True))
                for d in result['decisions']}
    assert choices(actual)==choices(old)
    if mode=='optimal':
        assert actual['value']['mean']==pytest.approx(27.5 if mct==0 else 40.)


@pytest.mark.parametrize('reverse,forecasts,cancel',[(False,False,0.),(False,True,.05),(True,True,0.)])
def test_working_policy_matches_independent_joint_world_enumeration(monkeypatch,reverse,forecasts,cancel):
    graph,model,params,history,platform,entry=shared_case(
        monkeypatch,reverse=reverse,forecasts=forecasts,cancel=cancel)
    expected=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=entry,clock=entry),history,
        action='policy',max_evaluations=100000)
    actual=CompressedReference(graph,model,params,60,history,
        ObservationContract.from_menu(platform),max_states=None,max_process_cpu=None).solve(
            platform,dict(kind='platform_start',entry=entry,clock=entry),mode='cheap_working')
    assert expected['unresolved_probability']==0
    assert actual['value']['complete']
    assert actual['value']['mean']==pytest.approx(expected['mean'],abs=1e-8)


@pytest.mark.parametrize('mct,expected',[(0,27.5),(6,30.)])
def test_simultaneous_same_trip_offer_and_message_are_conditioned_jointly(mct,expected):
    base,platform,_=information_case(observed=True,mct=mct)
    base.root_law.roots['B']=frozen_chain([[5,15],[5,15],[5,20],[25,55]],[.5,.5])
    new=compressed(base)
    result=new.solve(platform,dict(kind='platform_start',clock=0.,entry=0.))
    assert result['value']['complete']
    assert result['value']['mean']==pytest.approx(expected)
    assert result['counters']['same_trip_observation_groups']>0
    assert all(1 not in d['visible'] for d in result['decisions'])


@pytest.mark.parametrize('mode',CausalReference.MODES)
def test_root_wait_removes_current_offer_without_reopening_it(mode):
    base,platform,_=information_case(observed=False)
    history=base.snapshot.clone()
    history.observe_platform((0,2),0.,10.,{0:10.})
    new=compressed(base,history=history)
    row=dict(kind='platform',clock=10.,entry=0.,visible=[0],observed=[0])
    result=new.solve(platform,row,mode=mode)
    assert result['actions']['0']['mean']==pytest.approx(20.)
    assert result['actions']['wait']['mean']==pytest.approx(30.)
    assert result['certified_action']==0
    assert all(d['clock']>10. for d in result['decisions'])
    assert all(0 not in d.get('visible',()) for d in result['decisions'])


@pytest.mark.parametrize('mode',CausalReference.MODES)
@pytest.mark.parametrize('limit',[None,2])
def test_forced_onboard_arms_keep_original_arrival_and_policy_class(mode,limit):
    template,platform,initial=information_case(observed=False)
    graph=template.graph
    graph.add_stop(Stop(2,'before',min_transfer_time=0))
    graph.add_connection(Connection(3,'R','H',2,0,0,0))
    graph.add_connection(Connection(4,'R','H',0,1,1,50))
    model=EvolvingTripBelief(graph,public_occurrences=False)
    history=PhysicalHistory(model,template.parameters,fit=initial.fit)
    history.message_clock=0.
    history.observe(3,'departure',0.);history.observe(3,'arrival',0.)
    root=Ride(3,(3,4),{3:platform,4:None})
    base=CausalReference(graph,model,template.parameters,60,history,
        template.observations,max_states=None,max_process_cpu=None)
    base.root_law.roots.update(template.root_law.roots)
    base.root_law.roots['H']=frozen_chain([[0],[0],[1],[50]],[1.])
    new=compressed(base,max_states=limit)
    row=dict(kind='onboard',cid=3,clock=0.)
    exact=base.solve(root,row,mode=mode)
    actual=new.solve(root,row,mode=mode)
    for action,expected in exact['actions'].items():
        bound=actual['actions'][action]
        assert bound['lower']-1e-8 <= expected['mean'] <= bound['upper']+1e-8
        if limit is None:
            assert bound['complete']
            assert bound['mean']==pytest.approx(expected['mean'],abs=1e-8)
    if limit is None and mode=='optimal':
        assert actual['actions']['alight']['mean']==pytest.approx(30.)
        assert actual['actions']['continue']['mean']==pytest.approx(50.)
        assert actual['certified_action']=='alight'


@pytest.mark.parametrize('mode',CausalReference.MODES)
@pytest.mark.parametrize('limit',[0,1,2,6])
def test_finite_budget_contains_the_same_declared_policy_value(mode,limit):
    base,platform,_=information_case(observed=True)
    expected=base.solve(platform,dict(kind='platform_start',clock=0.,entry=0.),mode=mode)
    new=compressed(base,max_states=limit)
    result=new.solve(platform,dict(kind='platform_start',clock=0.,entry=0.),mode=mode)
    value=expected['value']['mean']
    assert result['value']['lower']-1e-8 <= value <= result['value']['upper']+1e-8
    assert result['counters']['states']<=limit
    if not result['value']['complete']:
        assert result['value']['mean'] is None


def test_no_event_chain_and_dynamic_grid_preserve_long_finite_support():
    base,platform,_=information_case(observed=False)
    platform.rides.pop(2)
    base.observations=ObservationContract({0:(0,)})
    base.deadline=400.
    times=np.arange(1,251,dtype=float)
    base.root_law.roots['A']=frozen_chain([times,times+1],np.full(len(times),1/len(times)))
    result=compressed(base).solve(platform,dict(kind='platform_start',clock=0.,entry=0.))
    assert result['value']['complete']
    assert result['value']['mean']==pytest.approx(float(times.mean()+1),abs=1e-8)
    assert result['counters']['iterative_no_event_steps']>=249


def test_terminal_cache_merges_only_irrelevant_public_history():
    base,platform=reference_with_nuisance(False)
    new=compressed(base)
    histories=[]
    for when in (3.,8.):
        h=base.snapshot.clone();h.observe(3,'departure',when);histories.append(h)
    values=[new._ride(platform.rides[0],0,h,0.,-1) for h in histories]
    assert values==[(30.,30.,True)]*2
    assert new.stats['whole_tail_integrations']==1
    assert new.stats['whole_tail_hits']==1
    assert histories[0]._records!=histories[1]._records
