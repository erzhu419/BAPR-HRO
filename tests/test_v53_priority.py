"""Impact cutoffs retain probability and conditional bounds, not accuracy promises."""
import numpy as np
import pytest

from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v52.reference import CausalReference, point
from experiments.reconstruction_v53.priority import ImpactReference
from test_v52_reference import information_case, frozen_chain
from test_v53_reference import compressed


def impact(base,cutoff=0.,max_states=None):
    reference=ImpactReference(base.graph,base.model,base.parameters,base.deadline,
        base.snapshot,base.observations,max_states=max_states,max_process_cpu=None,
        impact_cutoff=cutoff,keep_decisions=True)
    reference.root_law.roots.update(base.root_law.roots)
    return reference


@pytest.mark.parametrize('mode',CausalReference.MODES)
@pytest.mark.parametrize('mct',[0,11])
def test_zero_cutoff_preserves_all_five_complete_policy_values(mode,mct):
    base,platform,_=information_case(observed=True,mct=mct)
    ordinary=compressed(base)
    screened=impact(base)
    row=dict(kind='platform_start',clock=0.,entry=0.)
    expected=ordinary.solve(platform,row,mode=mode)
    actual=screened.solve(platform,row,mode=mode)
    assert expected['value']['complete'] and actual['value']['complete']
    assert actual['value']['mean']==pytest.approx(expected['value']['mean'],abs=1e-8)
    assert actual['counters']['impact_pruned_branches']==0
    assert actual['observation_contract']==expected['observation_contract']
    assert screened.reach==1.


def test_many_small_branches_retain_total_probability_and_accumulated_width():
    base,_,_=information_case()
    base.deadline=100.
    screened=impact(base,cutoff=.01)
    calls=[]
    def rare():
        calls.append('rare')
        return point(100.)
    outcomes=[(.0001,rare)]*3+[(.9997,lambda:point(0.))]
    value=screened._expect(outcomes,0.)
    assert value[0]<=.03<=value[1]+1e-12
    assert value[1]-value[0]==pytest.approx(.03)
    assert value[1]-value[0]>screened.impact_cutoff
    assert not value[2] and not calls
    assert screened.stats['impact_pruned_branches']==3
    assert screened.reach==1.


@pytest.mark.parametrize('mode',CausalReference.MODES)
def test_positive_cutoff_preserves_rare_information_case_and_incomplete_flag(mode):
    base,platform,_=information_case(observed=True)
    base.root_law.roots['B']=frozen_chain([[5,15],[6,16],[20,40],[25,55]],[.0001,.9999])
    row=dict(kind='platform_start',clock=0.,entry=0.)
    exact=compressed(base).solve(platform,row,mode=mode)
    result=impact(base,cutoff=.01).solve(platform,row,mode=mode)
    assert result['value']['lower']-1e-8 <= exact['value']['mean'] <= result['value']['upper']+1e-8
    assert result['counters']['impact_pruned_branches']>0
    assert not result['value']['complete']
    assert result['value']['mean'] is None


def test_no_event_reach_multiplies_across_the_iterative_clock_chain(monkeypatch):
    base,platform,_=information_case(observed=False)
    platform.rides.pop(2)
    base.observations=ObservationContract({0:(0,)})
    base.deadline=100.
    times=np.arange(1,5,dtype=float)
    base.root_law.roots['A']=frozen_chain([times,times+1],[.5,.25,.125,.125])
    screened=impact(base,cutoff=13.)
    probes=[]
    original=screened._small
    def record(clock,reach=None):
        probes.append((float(clock),screened.reach if reach is None else reach))
        return original(clock,reach)
    monkeypatch.setattr(screened,'_small',record)
    result=screened.solve(platform,dict(kind='platform_start',clock=0.,entry=0.))
    assert any(t==4. and p==pytest.approx(.125) for t,p in probes)
    assert result['value']['lower']-1e-8<=2.875<=result['value']['upper']+1e-8
    assert not result['value']['complete']
    assert screened.reach==1.


def test_min_siblings_inherit_same_reach_and_chance_children_restore_it():
    base,_,history=information_case()
    screened=impact(base)
    screened.reach=.7
    incoming=[];children=[]
    def arm(depth):
        incoming.append(screened.reach)
        def child(value):
            children.append(screened.reach)
            return point(value)
        return screened._expect(((.2,lambda:child(10.)),(.8,lambda:child(20.))),0.)
    result=screened._choose({'a':arm,'b':arm},0.,-1,history,{},lambda h:'a')
    assert result==pytest.approx((18.,18.,True))
    assert incoming==pytest.approx([.7,.7])
    assert children==pytest.approx([.14,.56,.14,.56])
    assert screened.reach==.7


def test_larger_reach_reopens_partial_event_cache_and_small_reach_reuses_exact():
    base,platform,history=information_case(observed=True)
    screened=impact(base,cutoff=1.)
    screened._prepare(platform)
    screened.reach=.02
    low=screened._platform(platform,0.,history,depth=-1)
    assert not low[2] and low[0]<=27.5<=low[1]
    states=screened.stats['states']
    pruned=screened.stats['impact_pruned_branches']
    # Both reaches are large enough that the root itself is not screened;
    # only settled partial descendants make these repeats unnecessary.
    for reach in (.02,.019):
        screened.reach=reach
        repeated=screened._platform(platform,0.,history,depth=-1)
        assert repeated==low and not repeated[2]
        assert screened.stats['states']==states
        assert screened.stats['impact_pruned_branches']==pruned
    screened.reach=1.
    high=screened._platform(platform,0.,history,depth=-1)
    assert high==pytest.approx((27.5,27.5,True))
    assert screened.stats['partial_cache_hits']>0
    assert screened.stats['states']>states
    states=screened.stats['states']
    screened.reach=.0001
    reused=screened._platform(platform,0.,history,depth=-1)
    assert reused==high
    assert screened.stats['states']==states


def test_each_no_event_suffix_remembers_its_own_settled_reach():
    base,platform,history=information_case(observed=False)
    platform.rides.pop(2)
    base.observations=ObservationContract({0:(0,)})
    base.deadline=100.
    times=np.arange(1,5,dtype=float)
    base.root_law.roots['A']=frozen_chain([times,times+1],[.5,.25,.125,.125])
    screened=impact(base,cutoff=13.)
    screened._prepare(platform)
    full=screened._platform(platform,0.,history,depth=-1)
    assert not full[2]
    after_first=history.clone()
    after_first.observe_platform((0,),0.,1.,{})
    states=screened.stats['states']
    pruned=screened.stats['impact_pruned_branches']
    for reach in (.5,.49):
        screened.reach=reach
        repeated=screened._platform(platform,0.,after_first,depth=-1,after=1.)
        assert not repeated[2] and repeated[0]<=3.75<=repeated[1]
        assert screened.stats['states']==states
        assert screened.stats['impact_pruned_branches']==pruned
    # This suffix originally had reach .5, not its ancestor's reach 1.
    screened.reach=.6
    refined=screened._platform(platform,0.,after_first,depth=-1,after=1.)
    assert refined==pytest.approx((3.75,3.75,True))
    assert screened.stats['states']>states


def test_budget_interrupted_node_is_not_settled_and_can_resume_same_reach():
    base,_,_=information_case()
    screened=impact(base,cutoff=.01,max_states=2)
    key=('root',)
    builds=[]
    def build():
        builds.append(screened.reach)
        return screened._expect(((.5,lambda:screened._node(('child',),0.,lambda:point(20.))),
                                 (.5,lambda:point(40.))),0.)
    partial=screened._node(key,0.,build)
    assert not partial[2] and partial[0]<=30.<=partial[1]
    assert screened.stats['budget_unknowns']>0
    assert key not in screened.settled_reach
    screened._state_limit=100
    result=screened._node(key,0.,build)
    assert result==pytest.approx((30.,30.,True))
    assert builds==[1.,1.]


def test_partial_node_finishing_at_budget_boundary_is_not_settled():
    base,_,_=information_case()
    screened=impact(base,cutoff=1.,max_states=1)
    key=('atomic_partial',)
    builds=[]
    def build():
        builds.append(screened.reach)
        return screened._pruned(0.)
    first=screened._node(key,0.,build)
    assert not first[2] and screened.stats['budget_unknowns']==0
    assert key not in screened.settled_reach
    screened._state_limit=10
    second=screened._node(key,0.,build)
    assert second==first and builds==[1.,1.]


def test_min_siblings_share_settled_partial_without_changing_incoming_reach():
    base,_,history=information_case()
    screened=impact(base,cutoff=30.)
    screened.reach=.7
    incoming=[];builds=[]
    def build():
        builds.append(screened.reach)
        return screened._expect(((.5,lambda:point(10.)),(.5,lambda:point(20.))),0.)
    def arm(depth):
        incoming.append(screened.reach)
        return screened._node(('shared_child',),0.,build)
    result=screened._choose({'a':arm,'b':arm},0.,-1,history,{},lambda h:'a')
    assert not result[2] and result[0]<=15.<=result[1]
    assert incoming==pytest.approx([.7,.7])
    assert builds==pytest.approx([.7])
    assert screened.reach==.7


@pytest.mark.parametrize('mode',CausalReference.MODES)
@pytest.mark.parametrize('limit',[0,2])
def test_impact_screening_still_obeys_state_cap_and_contains_exact_policy(mode,limit):
    base,platform,_=information_case(observed=True)
    row=dict(kind='platform_start',clock=0.,entry=0.)
    exact=compressed(base).solve(platform,row,mode=mode)
    result=impact(base,cutoff=.01,max_states=limit).solve(platform,row,mode=mode)
    assert result['value']['lower']-1e-8<=exact['value']['mean']<=result['value']['upper']+1e-8
    assert result['counters']['states']<=limit
    if not result['value']['complete']:
        assert result['value']['mean'] is None
