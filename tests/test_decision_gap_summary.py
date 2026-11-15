"""Regression facts, unresolved denominators, and fixed-policy comparisons."""
from copy import deepcopy

import pytest

from experiments.decision_gap_v1.summarize import (
    LABELS, NEW_LABELS, aggregate_rows, audit_gates, comparison, paired_rows,
    residual_diagnostic, validate_old_replay, validate_rows,
)


def receipt(label, root=0, loss=100.):
    return dict(root_index=root, root_id=f'root{root}', day='day', origin=0, goal=2,
        loss_cap=300., label=label, observation={'0':[1,2]}, menu_source='new menu',
        outcome=dict(loss=loss, status='complete' if loss is not None else 'missing_endpoint',
            trace=[dict(kind='onboard', cid=1, clock=10., choice='continue',
                        public_history={'seen':[]}, scoring={'cpu':1.})]),
        actions=[dict(kind='onboard', cid=1, clock=10., choice='continue')],
        states=[], gates=[], cost=dict(process_cpu=2., wall=3.,
            shared_menu=dict(process_cpu=.5, wall=.6),
            process_cpu_including_menu=2.5, wall_including_menu=3.6))


def bank(root=0, **losses):
    return {label:receipt(label, root, losses.get(label,100.)) for label in LABELS}


def test_old_replay_compares_public_facts_not_computational_receipts():
    old = receipt('cheap')
    new = receipt('old_old_cheap')
    new['outcome']['trace'][0]['scoring']['cpu'] = 9.
    assert all(validate_old_replay(new,old).values())
    new['outcome']['trace'][0]['public_history']['seen'].append(7)
    with pytest.raises(AssertionError,match='public_physical_trace_equal'):
        validate_old_replay(new,old)


def test_missing_endpoint_keeps_full_denominator_and_paired_loss_range():
    banks = [bank(0,new_full128=90.), bank(1,new_full128=None)]
    totals = aggregate_rows(banks,'new_full128')
    assert totals['complete'] == totals['unresolved'] == 1
    assert totals['all_root_mean_loss_bounds'] == [45.,195.]
    assert totals['complete_mean_loss'] == 90.
    compared = comparison(banks,'new_cheap','new_full128')
    assert compared['all_root_mean_gain_bounds'] == [-95.,55.]
    assert compared['wins'] == compared['undecided'] == 1
    assert compared['positive_roots'] == [0]
    assert compared['unresolved_roots'] == [1]
    pair = paired_rows(banks,'new_cheap','new_full128')[1]
    assert (pair['gain_lower'],pair['gain_upper']) == (-200.,100.)
    assert not pair['complete_pair']


def test_observation_and_preparation_cost_are_part_of_the_comparison():
    rows = bank()
    rows['old_old_cheap']['observation'] = {'old':[1]}
    validate_rows(rows,0)
    rows['new_mean']['observation'] = {'new':[1]}
    with pytest.raises(AssertionError,match='observation contract'):
        validate_rows(rows,0)
    rows['new_mean']['observation'] = rows['new_cheap']['observation']
    rows['new_mean']['cost']['process_cpu_including_menu'] = 2.
    with pytest.raises(AssertionError,match='Menu preparation'):
        validate_rows(rows,0)


def test_different_roots_explained_by_different_cheap_rules_are_not_one_policy():
    banks = [bank(4,new_full128=90.,new_pair=80.),
             bank(62,new_full128=90.,new_snapshot_csa=80.),
             bank(63,new_full128=90.)]
    result = residual_diagnostic(banks)['new_full128']
    assert result['strictly_better_than_every_cheap_policy_roots'] == [63]
    assert result['versus_each_fixed_cheap_policy']['new_pair']['negative_roots'] == [4]
    assert result['versus_each_fixed_cheap_policy']['new_snapshot_csa']['negative_roots'] == [62]
    assert result['versus_each_fixed_cheap_policy']['new_cheap']['positive_roots'] == [4,62,63]


def test_cpu_interruption_retains_every_world_and_falls_back_to_base_action():
    rows = bank()
    state = dict(kind='onboard',root_index=0,cid=1,clock=10.,node={},public_history={},
                 selected_action='alight',base_action='alight',gate_index=0)
    gate = dict(future_messages=0,count=2,rows={'alight':[{},{}],'continue':[{},{}]},
        paired={'full_bank_mean_gain_interval':[-290.,290.]},selected_action='alight',
        base_action='alight',state_index=0,used_kernel='full_common_observation_pi',
        fallback=None,completed_pairs=0)
    rows['new_full128'].update(states=[state],gates=[gate])
    assert audit_gates([rows],2)['cpu_unfinished_by_policy']['new_full128'] == 1
    gate['rows']['alight'].pop()
    with pytest.raises(AssertionError,match='Unresolved worlds'):
        audit_gates([rows],2)


def test_same_state_completed_full_rollout_uses_identical_pair_banks():
    rows = bank()
    for label in ('new_matched128','new_full128'):
        rows[label]['states'] = [dict(kind='onboard',root_index=0,cid=1,clock=10.,
            node={},public_history={},selected_action='alight',base_action='alight',gate_index=0)]
        rows[label]['gates'] = [dict(future_messages=0,count=2,
            rows={'alight':[{'loss':10.},{'loss':20.}],'continue':[{'loss':12.},{'loss':22.}]},
            paired={'full_bank_mean_gain_interval':[-2.,-2.]},selected_action='alight',
            base_action='alight',state_index=0,used_kernel='full_common_observation_pi',
            fallback=None,completed_pairs=2,common_tape={'seed':77})]
    assert audit_gates([rows],2)['same_state_full128_comparisons'] == 1
    rows['new_full128']['gates'][0]['rows']['alight'][0]['loss'] += 1.
    with pytest.raises(AssertionError,match='bank differs'):
        audit_gates([rows],2)
