"""Scientific attribution must distinguish proven gains from unknown ranges."""
import pytest
from copy import deepcopy

from experiments.reconstruction_v53.summarize import (
    check_working_root_action, compatible, gap, joint_attribution, observed_action_gaps,
    valid_certificate, valid_value,
)


def value(low,high=None):
    exact=high is None
    return dict(lower=low,upper=low if exact else high,complete=exact,mean=low if exact else None)


def test_gap_classifies_positive_zero_and_unresolved_without_completion_assumption():
    assert gap(value(20,30),value(5,10))['status']=='certified_positive_loss'
    assert gap(value(20),value(20))['status']=='certified_zero_within_tolerance'
    assert gap(value(10,30),value(5,20))['status']=='computationally_unresolved'
    with pytest.raises(ValueError,match='policy classes'):
        gap(value(10),value(20))


def test_working_belief_comparison_keeps_both_signs():
    assert gap(value(10),value(20),False)['status']=='certified_working_better'
    assert gap(value(20),value(10),False)['status']=='certified_working_worse'
    assert gap(value(10,30),value(15,25),False)['status']=='computationally_unresolved'


def test_exact_point_must_lie_inside_its_own_and_old_ranges():
    with pytest.raises(ValueError,match='contained point'):
        valid_value(dict(lower=10,upper=10,mean=12,complete=True),60)
    a=dict(value=value(5,15),actions={'a':value(5,15)},certified_action=None)
    b=dict(value=value(12),actions={'a':value(12)},certified_action=None)
    assert compatible(a,b)['complete_point_checks']==2
    b['actions']['a']=value(18)
    with pytest.raises(ValueError,match='disjoint'):
        compatible(a,b)


def test_certificate_requires_all_competitors_and_strict_separation():
    bank=dict(actions={'a':value(10,12),'b':value(12,20)},certified_action='a')
    with pytest.raises(ValueError,match='strict'):
        valid_certificate(bank)
    bank['actions']['b']=value(13,20)
    valid_certificate(bank)


def test_observed_choices_use_their_own_forced_optimal_continuation_only():
    observations=[dict(policy='cheap',selected_action=None,state_index=2,source='roots/000_cheap.json.gz#/states/2')]
    optimal=dict(value=value(10,12),actions={'wait':value(20,24),'5':value(10,12)})
    rows=observed_action_gaps(observations,optimal)
    assert len(rows)==1 and rows[0]['policy']=='cheap' and rows[0]['reference_action']=='wait'
    assert (rows[0]['lower'],rows[0]['upper'])==(8,14)
    assert rows[0]['status']=='certified_positive_loss'
    observations[0]['selected_action']=9
    with pytest.raises(ValueError,match='no matching'):
        observed_action_gaps(observations,optimal)


def test_restored_working_choice_matches_only_actual_cheap_occurrences():
    occurrences=[dict(policy='cheap',selected_action=None,source='cheap source'),
                 dict(policy='full128',selected_action=5,source='full source')]
    bank=dict(cheap_action='wait',selected_action='wait',certified_action=5)
    assert check_working_root_action(occurrences,bank)==1
    # A different certificate is not the original controller's actual action.
    assert check_working_root_action(occurrences[1:],dict(cheap_action=9,selected_action=9))==0
    bank['cheap_action']=5
    with pytest.raises(ValueError,match='root choice changed'):
        check_working_root_action(occurrences,bank)
    bank.update(cheap_action='wait',selected_action=5)
    with pytest.raises(ValueError,match='root choice changed'):
        check_working_root_action(occurrences,bank)


def test_certified_optimal_first_action_has_zero_loss_before_values_converge():
    observations=[dict(policy='cheap',selected_action=None,state_index=2,source='cheap source'),
                  dict(policy='full128',selected_action=5,state_index=2,source='full source')]
    optimal=dict(value=value(10,20),actions={'wait':value(10,20),'5':value(21,30)},certified_action='wait')
    valid_certificate(optimal)
    rows=observed_action_gaps(observations,optimal)
    assert (rows[0]['lower'],rows[0]['upper'])==(0.,0.)
    assert rows[0]['status']=='certified_zero_loss'
    assert rows[0]['proof_basis']=='strict_optimal_action_certificate'
    assert rows[0]['certified_optimal_action']=='wait'
    assert (rows[1]['lower'],rows[1]['upper'])==(1,20)
    # Knowing the first action is optimal does not identify the subsequent
    # fixed cheap policy value, which can still exceed the optimum.
    assert gap(value(15,25),optimal['value'])['status']=='computationally_unresolved'
    assert optimal['value']==value(10,20)


def joint_results():
    ranges={'optimal':(0,50),'two_step_p0':(0,70),'one_step_p0':(10,80),
            'cheap_p0':(20,90),'cheap_working':(5,100)}
    return {name:dict(value=value(*limits),cheap_action='alight',certified_action=None,
        actions={'alight':value(*limits),'continue':value(100,110)},cost=dict(process_cpu=i+1.))
        for i,(name,limits) in enumerate(ranges.items())}


def test_joint_policy_order_tightens_ranges_and_charges_all_five_calls():
    rows=joint_results()
    rows['cheap_working']['value']=value(5,40)
    before=deepcopy(rows)
    joint=joint_attribution(rows)
    assert joint['values']['optimal']['upper']==40
    assert joint['values']['cheap_p0']['lower']==20
    assert joint['values']['one_step_p0']['upper']==80
    assert joint['cost']==dict(process_cpu=15.,complete_queries=5)
    assert joint['equalities']==[] and rows==before


@pytest.mark.parametrize('certifying_mode',['cheap_p0','one_step_p0'])
def test_joint_shared_continuation_certificate_proves_only_one_step_equality(certifying_mode):
    rows=joint_results();rows[certifying_mode]['certified_action']='alight'
    valid_certificate(rows[certifying_mode])
    before=deepcopy(rows)
    joint=joint_attribution(rows)
    assert joint['values']['cheap_p0']['upper']==joint['values']['one_step_p0']['upper']==80
    assert joint['values']['cheap_p0']['lower']==joint['values']['one_step_p0']['lower']==20
    proof=joint['gaps']['cheap_p0_minus_one_step']
    assert (proof['lower'],proof['upper'])==(0.,0.)
    assert proof['status']=='certified_zero_loss' and proof['certificate_modes']==[certifying_mode]
    assert joint['gaps']['cheap_p0_minus_optimal']['status']=='computationally_unresolved'
    assert rows==before


def test_different_certified_action_does_not_prove_cheap_one_step_equality():
    rows=joint_results()
    rows['cheap_p0'].update(value=value(80,105),
        actions={'continue':value(10,70),'alight':value(80,105)})
    rows['one_step_p0'].update(value=value(10,70),certified_action='continue',
        actions={'continue':value(10,70),'alight':value(80,110)})
    valid_certificate(rows['one_step_p0'])
    joint=joint_attribution(rows)
    assert joint['equalities']==[]
    assert joint['gaps']['cheap_p0_minus_one_step']['status']=='certified_positive_loss'
