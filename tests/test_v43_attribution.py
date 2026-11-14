import pytest

from experiments.reconstruction_v43.attribution import attribution_summary, decompose, menu_counts


def test_absolute_clock_conversion_and_signed_error_decomposition():
    rows={'alight':[dict(loss=8.),dict(loss=12.)], 'continue':[dict(loss=12.),dict(loss=16.)]}
    result=decompose(dict(alight=120.,continuation=130.),rows,100.,200.,frozen_alight=40.)
    assert result['cheap_alight']==20. and result['cheap_continue']==30.
    assert result['mc_gap']==-4.
    assert result['alight_error']==10. and result['continue_error']==16.
    assert result['gap_error']==result['alight_error']-result['continue_error']==-6.
    # Identical paired differences have zero SE despite nonzero arm variance.
    assert result['mc_paired_se']==0.
    assert result['hybrid_changes_cheap_rank'] and not result['hybrid_agrees_mc_rank']
    aggregate=attribution_summary([result])
    assert aggregate['hybrid_introduces_new_crossing']==1


def test_current_alight_correction_can_resolve_a_score_boundary_without_policy_claim():
    rows={'alight':[dict(loss=10.)], 'continue':[dict(loss=30.)]}
    result=decompose(dict(alight=140.,continuation=130.),rows,100.,200.,frozen_alight=10.)
    assert result['mc_paired_se'] is None and result['mc_ci95'] is None
    assert result['cheap_mc_boundary_crossing']
    assert result['hybrid_agrees_mc_rank'] and result['hybrid_gap']==-20.
    assert attribution_summary([result])['hybrid_resolves_existing_crossing']==1
    rows['alight'][0]['loss']=None
    with pytest.raises(ValueError,match='do not discard'):
        decompose(dict(alight=140.,continuation=130.),rows,100.,200.)


def test_zero_mc_gap_keeps_v41_base_action_instead_of_creating_a_boundary_crossing():
    rows={'alight':[dict(loss=20.)], 'continue':[dict(loss=20.)]}
    result=decompose(dict(alight=140.,continuation=130.),rows,100.,200.)
    assert result['mc_gap_zero'] and result['mc_rank']=='continue'
    assert not result['cheap_mc_boundary_crossing']


def test_static_menu_counts_remove_past_exits_but_keep_all_future_branches():
    direct=dict(kind='ride',first_cid=8,path=[8,9],exits=[dict(cid=9,next=None)])
    platform=dict(kind='platform',stop=4,rides=[dict(cid=8,next=direct)])
    node=dict(kind='ride',first_cid=1,path=[1,2,3],exits=[dict(cid=1,next=platform),
        dict(cid=2,next=platform),dict(cid=3,next=None)])
    counts=menu_counts(node,2)
    assert counts['current_and_future_exit_count']==2 and counts['future_exit_count']==1
    assert counts['ride_occurrences']==2 and counts['ride_exit_occurrences']==3
    assert counts['platform_occurrences']==1 and counts['max_transfer_depth']==1
    assert counts['max_ride_exits']==2
