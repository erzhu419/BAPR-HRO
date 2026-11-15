"""Causal minima, observation-only information, and retained budget bounds."""
import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v46.small_oracle import evaluate
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v52.reference import CausalReference
from src.evolving_trip_belief import EvolvingTripBelief, _ChainLaw
from src.transit_graph import TransitGraph, Stop, Connection
from test_v46_shared import shared_case


def frozen_chain(times, mass):
    times = [np.array(row, dtype=float) for row in times]
    count = len(mass)
    transition = np.full((count, count), -np.inf)
    np.fill_diagonal(transition, 0.)
    return _ChainLaw(times, np.log(mass), lambda slot:transition,
                     [np.zeros(count) for _ in times],
                     [np.zeros(count, dtype=bool) for _ in times],
                     tuple(True for _ in times[1:]))


def information_case(*, observed=True, mct=0, max_states=10000, implementation='chronological'):
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'origin', min_transfer_time=mct))
    graph.add_stop(Stop(1, 'destination', min_transfer_time=0))
    for cid, trip, origin, target, dep, arr in (
            (0, 'A', 0, 1, 10, 30), (1, 'B', 0, 0, 5, 6), (2, 'B', 0, 1, 20, 25)):
        graph.add_connection(Connection(cid, 'R', trip, origin, target, dep, arr))
    model = EvolvingTripBelief(graph, public_occurrences=False)
    parameters = {'R':dict(mean=0., std=.1, cancel_rate=0.)}
    fit = dict(route_kind={}, pooled={k:dict(bias_seconds=0.,sd_seconds=5.) for k in ('departure','arrival')})
    history = PhysicalHistory(model, parameters, fit=fit)
    history.message_clock = 0.
    platform = Platform(0, {0:Ride(0,(0,),{0:None}), 2:Ride(2,(2,),{2:None})})
    contract = ObservationContract({0:(0,1,2) if observed else (0,2)})
    reference = CausalReference(graph, model, parameters, 60, history, contract,
                                max_states=max_states, max_process_cpu=None, keep_decisions=True,
                                implementation=implementation)
    reference.root_law.roots.update(A=frozen_chain([[10],[30]],[1.]),
        B=frozen_chain([[5,15],[6,16],[20,40],[25,55]],[.5,.5]))
    return reference, platform, history


def test_causal_optimum_cannot_choose_with_unseen_future_outcome():
    reference, platform, _ = information_case(observed=False)
    result = reference.solve(platform, dict(kind='platform_start',clock=0.,entry=0.))
    # At A's offer, B's unobserved cost is equally 25 or55. Choose A at30.
    assert result['value']['complete']
    assert result['value']['mean'] == pytest.approx(30.)
    assert result['value']['mean'] > .5*25+.5*30  # private-world clairvoyance


def test_observation_only_vehicle_changes_later_legal_choices():
    reference, platform, _ = information_case(observed=True)
    result = reference.solve(platform, dict(kind='platform_start',clock=0.,entry=0.))
    assert result['value']['mean'] == pytest.approx(27.5)
    decisions = [d for d in result['decisions'] if d['clock']==10.]
    assert {d['selected'] for d in decisions} == {'wait',0}
    assert all(1 not in d['visible'] for d in decisions)
    assert result['future_public_frames']==0


def test_same_trip_simultaneous_events_are_joint_not_independent():
    reference, _, history = information_case()
    reference.root_law.roots['B'] = frozen_chain([[5,15],[5,15],[5,20],[25,55]],[.5,.5])
    outcomes = reference._joint_trip_outcomes('B', (1,2), 5., history)
    assert dict((visible,p) for p,visible in outcomes) == {():.5,(1,2):.5}
    assert reference.stats['same_trip_observation_groups']==1


def test_long_supported_clock_grid_does_not_recurse_on_absence():
    reference,platform,_=information_case(observed=False)
    platform.rides.pop(2)
    reference.observations=ObservationContract({0:(0,)})
    reference.deadline=400.
    times=np.arange(1,251,dtype=float)
    reference.root_law.roots['A']=frozen_chain([times,times+1],np.full(len(times),1/len(times)))
    result=reference.solve(platform,dict(kind='platform_start',clock=0.,entry=0.))
    assert result['value']['complete']
    assert result['value']['mean']==pytest.approx(float(times.mean()+1),abs=1e-8)
    assert result['counters']['iterative_no_event_steps']>=249


def test_ready_is_not_reset_and_pre_ready_offer_is_not_boardable():
    reference, platform, _ = information_case(observed=True,mct=11)
    result = reference.solve(platform, dict(kind='platform_start',clock=0.,entry=0.))
    assert result['value']['mean'] == pytest.approx(40.)
    assert all(d['clock'] >= 11 for d in result['decisions'])


def test_budget_remainder_keeps_optimum_inside_interval():
    reference, platform, _ = information_case(observed=True,max_states=2)
    result = reference.solve(platform, dict(kind='platform_start',clock=0.,entry=0.))
    assert not result['value']['complete']
    assert result['value']['lower'] <= 27.5 <= result['value']['upper']
    assert result['counters']['states'] <= 2
    assert result['value']['mean'] is None


def test_p0_cheap_uses_the_same_conditional_physical_chain():
    reference, platform, _ = information_case(observed=True)
    result = reference.solve(platform, dict(kind='platform_start',clock=0.,entry=0.),mode='cheap_p0')
    assert result['value']['complete']
    assert result['value']['mean'] >= 27.5-1e-8
    assert result['scoring_law']=='Consistent root P0'


def test_finite_improvement_depth_counts_actual_choices_not_observation_epochs():
    one, platform, _ = information_case(observed=True)
    first = one.solve(platform,dict(kind='platform_start',clock=0.,entry=0.),mode='one_step_p0')
    assert all(d['optimized'] for d in first['decisions'] if d['clock']==10.)
    assert all(not d['optimized'] for d in first['decisions'] if d['clock']>10.)
    two, platform, _ = information_case(observed=True)
    second = two.solve(platform,dict(kind='platform_start',clock=0.,entry=0.),mode='two_step_p0')
    assert any(d['optimized'] for d in second['decisions'] if d['clock']>10.)


def test_forced_onboard_actions_start_after_current_arrival():
    reference, platform, history = information_case(observed=False)
    graph = reference.graph
    graph.add_stop(Stop(2,'before',min_transfer_time=0))
    graph.add_connection(Connection(3,'R','H',2,0,0,0))
    graph.add_connection(Connection(4,'R','H',0,1,1,50))
    model=EvolvingTripBelief(graph,public_occurrences=False)
    history=PhysicalHistory(model,reference.parameters,fit=history.fit)
    history.message_clock=0.
    history.observe(3,'departure',0.)
    history.observe(3,'arrival',0.)
    root=Ride(3,(3,4),{3:platform,4:None})
    oracle=CausalReference(graph,model,reference.parameters,60,history,
        reference.observations,max_states=10000,max_process_cpu=None)
    oracle.root_law.roots.update(reference.root_law.roots)
    oracle.root_law.roots['H']=frozen_chain([[0],[0],[1],[50]],[1.])
    result=oracle.solve(root,dict(kind='onboard',cid=3,clock=0.))
    assert result['actions']['alight']['mean']==pytest.approx(30.)
    assert result['actions']['continue']['mean']==pytest.approx(50.)
    assert result['certified_action']=='alight'


@pytest.mark.parametrize('reverse', [False,True])
def test_working_policy_matches_independent_private_event_enumeration(monkeypatch,reverse):
    graph,model,params,history,platform,entry=shared_case(monkeypatch,reverse=reverse,forecasts=True)
    independent=evaluate(graph,model,params,60,platform,
        dict(kind='platform_start',entry=entry,clock=entry),history,action='policy',max_evaluations=100000)
    reference=CausalReference(graph,model,params,60,history,
        ObservationContract.from_menu(platform),max_states=100000,max_process_cpu=None)
    result=reference.solve(platform,dict(kind='platform_start',entry=entry,clock=entry),mode='cheap_working')
    assert result['value']['complete']
    assert result['value']['mean']==pytest.approx(independent['mean'],abs=1e-8)


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('observed', [False,True])
def test_mass_terminal_preserves_causal_values_and_choices(mode, observed):
    results=[]
    for implementation in CausalReference.IMPLEMENTATIONS:
        reference,platform,_=information_case(observed=observed,implementation=implementation)
        results.append(reference.solve(platform,dict(kind='platform_start',clock=0.,entry=0.),mode=mode))
    old,new=results
    assert all(r['value']['complete'] for r in results)
    assert new['value']['mean']==pytest.approx(old['value']['mean'],abs=1e-8)
    assert {a:v['mean'] for a,v in new['actions'].items()}==pytest.approx(
        {a:v['mean'] for a,v in old['actions'].items()},abs=1e-8)
    assert sorted((d['clock'],str(d['selected'])) for d in new['decisions'])==sorted(
        (d['clock'],str(d['selected'])) for d in old['decisions'])
    assert new['counters']['terminal_integrations']>0
    assert old['counters']['terminal_integrations']==0
    assert new['counters']['chance_outcomes']<old['counters']['chance_outcomes']


def test_terminal_integral_preserves_cancellation_and_truncation():
    values=[]
    for implementation in CausalReference.IMPLEMENTATIONS:
        reference,platform,history=information_case(observed=False,implementation=implementation)
        law=frozen_chain([[10]*5,[5,20,60,65,25]],[.1,.2,.3,.25,.15])
        law.canceled[1][-1]=True
        reference.root_law.roots['A']=law
        values.append(reference._ride(platform.rides[0],0,history,0.,-1))
    # Only the second endpoint is a legal arrival; all other mass costs H.
    assert values[0]==pytest.approx((52.,52.,True),abs=1e-8)
    assert values[1]==pytest.approx(values[0],abs=1e-8)


@pytest.mark.parametrize('distinct_departures', [False,True])
def test_mass_order_retains_bounds_and_covers_heavy_branch_with_equal_state_budget(distinct_departures):
    bounds=[]
    for implementation in CausalReference.IMPLEMENTATIONS:
        old,_,history=information_case(observed=False)
        graph=old.graph
        graph.add_stop(Stop(2,'last',min_transfer_time=0))
        graph.add_connection(Connection(3,'R','A',1,2,31,35))
        model=EvolvingTripBelief(graph,public_occurrences=False)
        history=PhysicalHistory(model,old.parameters,fit=history.fit)
        history.message_clock=0.
        reference=CausalReference(graph,model,old.parameters,60,history,
            ObservationContract({1:()}),max_states=4,max_process_cpu=None,
            implementation=implementation)
        departures=[10,11,12] if distinct_departures else [10]*3
        reference.root_law.roots['A']=frozen_chain(
            [departures,[20,30,40],[21,31,41],[25,35,45]],[.01,.01,.98])
        ride=Ride(0,(0,3),{0:Platform(1,{}),3:None})
        bound=reference._ride(ride,0,history,0.,-1)
        assert not bound[2]
        assert bound[0] <= 44.7 <= bound[1]
        assert reference.stats['states']==4
        bounds.append(bound)
    assert bounds[1][1]-bounds[1][0] < bounds[0][1]-bounds[0][0]


def test_terminal_integral_stops_before_work_when_budget_is_spent():
    reference,_,_=information_case(implementation='mass_terminal',max_states=0)
    law=reference.root_law.roots['A']
    bound=reference._terminal_arrival(law,1,law.marginal(1),10.)
    assert bound==(10.,60.,False)
    assert reference.stats['terminal_integrations']==0
