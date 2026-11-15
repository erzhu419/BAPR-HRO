"""Initial bounds must include causal values and preserve trip correlation."""
import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v53.bounds import (
    commitment_upper_bound, destination_endpoints, destination_lower_bound,
)
from test_v52_reference import frozen_chain, information_case
from test_v51_baselines import deterministic_case


def lower(reference, node, history, clock=0., position=None):
    return destination_lower_bound(reference.graph, reference.root_law,
        reference.deadline, node, history, clock, position)


def test_independent_destination_minimum_is_lower_than_causal_optimum():
    reference, node, history = information_case(observed=False)
    assert lower(reference, node, history) == pytest.approx(27.5)
    result = reference.solve(node, dict(kind='platform_start',clock=0.,entry=0.))
    assert result['value']['mean'] == pytest.approx(30.)


def test_same_trip_destination_slots_use_union_bound_not_independence():
    reference, _, history = information_case(observed=False)
    reference.deadline = 100.
    reference.root_law.roots['B'] = frozen_chain(
        [[5,15],[25,55],[26,56],[30,60]],[.5,.5])
    node = Platform(0, {1:Ride(1,(1,),{1:None}),2:Ride(2,(2,),{2:None})})
    # Actual min under the perfectly shared type has expectation40. A
    # product of slot marginals would give32.5; the valid union bound27.5.
    assert lower(reference, node, history) == pytest.approx(27.5)


def test_past_cancelled_and_late_endpoints_are_deadline_mass():
    reference, node, history = information_case(observed=False)
    node.rides.pop(0)
    assert lower(reference, node, history, clock=40.) == pytest.approx(57.5)
    law = reference.root_law.roots['B']
    law.canceled[3][1] = True
    assert lower(reference, node, history, clock=40.) == pytest.approx(60.)
    assert lower(reference, node, history, clock=60.) == 60.


def test_reachable_endpoint_collection_respects_suffix_and_first_destination():
    a = Platform(0,{0:Ride(0,(0,),{0:None})})
    b = Platform(0,{2:Ride(2,(2,),{2:None})})
    ride = Ride(10,(10,11,12,13),{10:a,11:b,12:None,13:a})
    assert destination_endpoints(ride) == (0,2,12)
    assert destination_endpoints(ride,1) == (2,12)
    assert destination_endpoints(ride,2) == (12,)
    assert destination_endpoints(ride,4) == ()


def test_current_physical_evidence_conditions_endpoint_bounds():
    reference, node, history = information_case(observed=False)
    node.rides.pop(0)
    history.observe(1,'departure',5.)
    assert lower(reference,node,history,clock=5.) == pytest.approx(25.)
    # A future publication attached to history is not consumed by a bound.
    history.frames = [dict(receipt_seconds=99., endpoint_updates=[])]
    assert lower(reference,node,history,clock=5.) == pytest.approx(25.)
    assert history.frame_index == 0


def test_empty_remaining_menu_cannot_arrive_before_deadline():
    reference, node, history = information_case(observed=False)
    assert lower(reference,Platform(0,{}),history) == reference.deadline


def upper(reference, node, history, clock=0., **kwargs):
    return commitment_upper_bound(reference.graph,reference.root_law,
        reference.deadline,node,history,clock,**kwargs)


def test_fixed_route_upper_cannot_claim_adaptive_information_gain():
    reference,node,history = information_case(observed=True)
    bound = upper(reference,node,history)
    assert bound['upper'] == pytest.approx(30.)
    result = reference.solve(node,dict(kind='platform_start',clock=0.,entry=0.))
    assert result['value']['mean'] == pytest.approx(27.5)
    assert bound['upper'] > result['value']['mean']


def test_fixed_route_wait_excludes_current_offer_and_preserves_ready():
    reference,node,history = information_case(observed=False)
    assert upper(reference,node,history,clock=10.,ready=0.)['upper'] == pytest.approx(30.)
    assert upper(reference,node,history,clock=10.,ready=0.,after=10.)['upper'] == pytest.approx(40.)
    assert upper(reference,node,history,clock=10.,ready=21.)['upper'] == pytest.approx(57.5)


@pytest.mark.parametrize('mct,expected',[(0,30.),(6,35.)])
def test_independent_fixed_route_transfer_integrates_legality(monkeypatch,mct,expected):
    from experiments.reconstruction_v43.reference import RootConditionalLaw
    graph,model,parameters,history,node = deterministic_case(monkeypatch,transfer=mct)
    root_law = RootConditionalLaw(history,node.trips(model))
    receipt = commitment_upper_bound(graph,root_law,60.,node,history,9.)
    assert receipt['upper'] == pytest.approx(expected)


def test_commitment_repeated_trip_is_rejected_without_shrinking_menu(monkeypatch):
    from experiments.reconstruction_v43.reference import RootConditionalLaw
    graph,model,parameters,history,_ = deterministic_case(monkeypatch)
    tail = Platform(1,{1:Ride(1,(1,),{1:None})})
    node = Platform(0,{0:Ride(0,(0,),{0:tail})})
    receipt = commitment_upper_bound(graph,RootConditionalLaw(history,node.trips(model)),
        60.,node,history,9.)
    assert receipt['upper'] == 60. and receipt['repeated_trip_rejections'] == 1
    assert tuple(node.rides) == (0,) and tuple(tail.rides) == (1,)


def test_commitment_masks_intermediate_failure_even_if_terminal_operates(monkeypatch):
    from experiments.reconstruction_v43.reference import RootConditionalLaw
    graph,model,parameters,history,_ = deterministic_case(monkeypatch)
    node = Ride(0,(0,1),{1:None})
    root_law = RootConditionalLaw(history,())
    root_law.roots['A'] = frozen_chain([[10,10],[20,20],[22,22],[40,40]],[.5,.5])
    root_law.roots['A'].canceled[1][1] = True
    receipt = commitment_upper_bound(graph,root_law,60.,node,history,9.)
    assert receipt['upper'] == pytest.approx(50.)


def test_random_transfer_upper_matches_four_independent_physical_worlds(monkeypatch):
    from experiments.reconstruction_v43.reference import RootConditionalLaw
    graph,model,parameters,history,_ = deterministic_case(monkeypatch)
    tail = Platform(1,{3:Ride(3,(3,),{3:None})})
    node = Platform(0,{0:Ride(0,(0,),{0:tail})})
    root_law = RootConditionalLaw(history,())
    root_law.roots['A'] = frozen_chain([[10,10],[20,30],[22,32],[40,50]],[.5,.5])
    root_law.roots['C'] = frozen_chain([[25,35],[40,50]],[.5,.5])
    # Arrival20 catches either C; arrival30 misses C25 and waits to H.
    expected = (40.+50.+60.+50.)/4
    receipt = commitment_upper_bound(graph,root_law,60.,node,history,9.)
    assert receipt['upper'] == pytest.approx(expected)


def test_commitment_limit_keeps_a_valid_upper_and_reports_unfinished_search():
    reference,node,history = information_case(observed=False)
    receipt = upper(reference,node,history,max_routes=0)
    assert receipt['upper'] == 60. and receipt['stopped'] == 'route_limit'
    receipt = upper(reference,node,history,available=lambda:False)
    assert receipt['upper'] == 60. and receipt['stopped'] == 'compute_limit'
