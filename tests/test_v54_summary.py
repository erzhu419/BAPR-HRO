"""V54 pairing preserves regressions and charges incremental refinement."""
from copy import deepcopy

import pytest

from experiments.reconstruction_v54.summarize import (
    gap_aggregate, matched_action_attribution, matched_supplement,
    paired_aggregate, paired_cell, refinement_row, require_panel,
)


def result(low,high,cpu,certificate=None):
    return dict(value=dict(lower=low,upper=high,complete=False),
        cost=dict(process_cpu=cpu),certified_action=certificate)


def test_paired_comparison_retains_wider_slower_and_lost_certificate_cells():
    old=result(5,10,1.,'alight');new=result(4,12,1.5)
    before=deepcopy((old,new))
    rows=[paired_cell(1,1,'b2','optimal','impact',old,new),
          paired_cell(2,2,'b2','optimal','impact',new,old)]
    summary=paired_aggregate(rows)
    assert summary['cells']==2 and summary['wider']==summary['narrower']==1
    assert summary['lost_certificate_queries']==[1]
    assert summary['gained_certificate_queries']==[2]
    assert rows[0]['cpu_change']==.5 and rows[0]['width_reduction']==-3
    assert (old,new)==before


def test_signed_and_unresolved_gaps_remain_in_the_full_denominator():
    rows=[dict(query_index=0,status='certified_working_better',lower=-5.,upper=-1.),
          dict(query_index=1,status='certified_zero_loss',lower=0.,upper=0.),
          dict(query_index=2,status='computationally_unresolved',lower=0.,upper=100.)]
    output=gap_aggregate(rows)
    assert output['states']==3 and output['negative_queries']==[0]
    assert output['zero_queries']==[1] and output['positive_queries']==[]
    assert output['statuses']['computationally_unresolved']==1


def panel():
    return dict(rows=[dict(query_index=i,state=dict(root_index=i%72)) for i in range(88)])


def test_full_panel_and_original_budget_are_required():
    protocol=dict(modes=['a'],budgets=[dict(name='b2',states=2000,process_cpu_seconds=2)])
    require_panel(panel(),protocol,deepcopy(protocol))
    smaller=panel();smaller['rows'].pop()
    with pytest.raises(ValueError,match='88-state'):
        require_panel(smaller,protocol,protocol)
    changed=deepcopy(protocol);changed['budgets'][0]['process_cpu_seconds']=3
    with pytest.raises(ValueError,match='original budgets'):
        require_panel(panel(),changed,protocol)


def receipt():
    return dict(query_index=1,root_index=1,mode='optimal',clock=100.,budget=dict(name='b2',states=2000),
        counters=dict(states=1800),value=dict(lower=20.,upper=35.),complete_cost=dict(process_cpu=1.8),
        cost=dict(process_cpu=1.75),
        root_refinement=dict(initial_cutoff=.01,final_cutoff=.001,reflow_calls=1,threshold_refinements=1,
            first_pass_value=[120.,170.,False],first_pass_cpu=.8,first_pass_states=1000,final_stop='budget',
            passes=[dict(phase='initial',states_start=0,states_end=1000,state_limit=1000,
                         cpu_start=0.,cpu_end=.8,unsupported_histories=0,before=[100.,200.,False],after=[120.,170.,False]),
                    dict(phase='reflow',states_start=1000,states_end=1800,state_limit=2000,
                         cpu_start=.8,cpu_end=1.7,unsupported_histories=0,before=[120.,170.,False],after=[120.,135.,False])]))


def test_refinement_charges_cumulative_work_and_keeps_first_pass_scope():
    output=refinement_row(receipt())
    assert output['states']==1800 and output['complete_cpu']==1.8
    assert output['reflow_pass_cpu']==pytest.approx(.9)
    assert output['after_first_pass_solver_cpu']==pytest.approx(.95)
    assert output['width_reduction_after_first_pass']==35.


def test_initial_pass_has_no_previous_action_interval():
    bank=receipt();bank['root_refinement']['passes'][0]['before']=None
    row=refinement_row(bank)
    assert row['passes']==2 and row['reflow_calls']==1
    bank['root_refinement']['passes'][1]['after']=[119.,135.,False]
    with pytest.raises(ValueError,match='root action interval widened'):
        refinement_row(bank)


@pytest.mark.parametrize('change,pattern',[
    (lambda b:b['root_refinement']['passes'][1].update(states_start=0),'cumulative state'),
    (lambda b:b['root_refinement']['passes'][1].update(states_end=2001),'cumulative state'),
    (lambda b:b['root_refinement']['passes'][1].update(cpu_start=.1),'CPU clock'),
    (lambda b:b['complete_cost'].update(process_cpu=1.5),'omitted'),
    (lambda b:b['root_refinement'].update(final_cutoff=.1),'raised'),
    (lambda b:b['root_refinement']['passes'][1].update(after=[119.,135.,False]),'root action interval widened'),
    (lambda b:b['value'].update(upper=71.),'policy interval widened'),
])
def test_refinement_rejects_budget_reset_or_uncharged_work(change,pattern):
    bank=receipt();change(bank)
    with pytest.raises(ValueError,match=pattern):
        refinement_row(bank)


def interval(low,high):
    return dict(lower=low,upper=high,complete=False,mean=None)


def matching_results():
    limits={'optimal':(0,8),'two_step_p0':(3,8),'one_step_p0':(6,8),
            'cheap_p0':(10,40),'cheap_working':(10,40)}
    results={mode:dict(value=interval(*bounds),cheap_action='wait',certified_action=None,
                      selected_action='wait',cost=dict(process_cpu=i+1.))
             for i,(mode,bounds) in enumerate(limits.items())}
    results['cheap_p0']['actions']={'wait':interval(10,40),'133':interval(5,15)}
    results['one_step_p0']['actions']={'wait':interval(20,30),'133':interval(6,8)}
    return results


def test_matched_actions_rebuild_different_root_choices_without_intersecting_root_values():
    results=matching_results();before=deepcopy(results)
    derived=matched_action_attribution(results,'platform')
    # C0 and V1 ranges are disjoint legitimately: they choose different actions.
    assert derived['values']['cheap_p0']['lower']==20
    assert derived['values']['cheap_p0']['upper']==30
    assert derived['values']['one_step_p0']['upper']==8
    assert derived['gaps']['cheap_p0_minus_one_step']['status']=='certified_positive_loss'
    assert derived['matched_minimizer_certificate']==133
    assert derived['equalities']==[] and derived['cost']['process_cpu']==15.
    assert results==before


def test_narrow_one_step_value_does_not_replace_wide_cost_of_cheaps_different_action():
    results=matching_results()
    results['cheap_p0']['actions']={'wait':interval(1265,1419),'133':interval(1265,1280)}
    results['one_step_p0']['actions']={'wait':interval(1265,1421),'133':interval(1265,1267)}
    for mode in results:
        results[mode]['value']=interval(1200,1500)
    results['optimal']['value']=interval(1200,1267)
    results['two_step_p0']['value']=interval(1265,1267)
    derived=matched_action_attribution(results,'platform')
    assert derived['values']['cheap_p0']['upper']==1419
    assert derived['values']['one_step_p0']['upper']==1267
    assert derived['gaps']['cheap_p0_minus_one_step']['status']=='computationally_unresolved'


def test_matched_root_certificate_proves_one_step_equality_not_full_optimality():
    results=matching_results()
    results['cheap_p0']['actions']={'alight':interval(20,35),'continue':interval(40,60)}
    results['one_step_p0']['actions']={'alight':interval(15,30),'continue':interval(45,50)}
    results['cheap_p0'].update(cheap_action='alight',value=interval(20,35))
    results['one_step_p0'].update(cheap_action='alight',value=interval(15,30))
    results['optimal']['value']=interval(0,15)
    results['two_step_p0']['value']=interval(10,20)
    derived=matched_action_attribution(results,'onboard')
    assert derived['gaps']['cheap_p0_minus_one_step']['status']=='certified_zero_loss'
    assert derived['gaps']['cheap_p0_minus_optimal']['lower']==5
    assert derived['gaps']['cheap_p0_minus_optimal']['status']=='certified_positive_loss'


def test_platform_start_is_not_admissible_for_matched_root_continuations():
    with pytest.raises(ValueError,match='onboard or platform'):
        matched_action_attribution(matching_results(),'platform_start')


def test_disjoint_same_action_bounds_fail_instead_of_becoming_a_new_claim():
    results=matching_results();results['one_step_p0']['actions']['wait']=interval(41,50)
    with pytest.raises(ValueError,match='disjoint'):
        matched_action_attribution(results,'platform')


def test_matched_supplement_keeps_versions_separate_and_reports_gap_upper_tiers():
    first=matching_results();second=matching_results()
    second['cheap_p0']['actions']['wait']=interval(10,35)
    second['one_step_p0']['actions']['wait']=interval(20,35)
    records=lambda data:[dict(query_index=0,root_index=0,kind='platform',results={'b2':data})]
    output=matched_supplement({'a':records(first),'b':records(second)},[dict(name='b2')])
    a,b=(output['aggregate'][version]['b2'] for version in ('a','b'))
    assert a['gaps']['cheap_p0_minus_optimal']['mean_upper']==30
    assert b['gaps']['cheap_p0_minus_optimal']['mean_upper']==35
    assert a['gaps']['cheap_p0_minus_optimal']['upper_bound_at_most_seconds']['10']['states']==0
    assert a['gaps']['cheap_p0_minus_optimal']['upper_bound_at_most_seconds']['60']['states']==1
    assert a['mean_cpu_per_joint_state']==b['mean_cpu_per_joint_state']==15.


def test_complete_pipeline_keeps_all_modes_budgets_and_five_query_costs(tmp_path,monkeypatch):
    import experiments.reconstruction_v54.summarize as module
    modes=['optimal','cheap_p0','cheap_working','one_step_p0','two_step_p0']
    budgets=[dict(name='b2',states=2000,process_cpu_seconds=2),
             dict(name='b10',states=20000,process_cpu_seconds=10)]
    protocol=dict(modes=modes,budgets=budgets)
    frozen=panel();frozen.update(missing_first_nontrivial_roots=[],all_state_count=507,same_history_disagreement_states=36)
    directory=tmp_path/'reconstruction_v54';(directory/'natural').mkdir(parents=True)
    bank=receipt();bank.update(day='test',implementation='refining_conditional',deadline=200.,future_public_frames=0,
        selected_action='alight',cheap_action='alight',certified_action='alight',
        value=dict(lower=20.,upper=35.,complete=False,mean=None),
        actions={'alight':dict(lower=20.,upper=35.,complete=False,mean=None),
                 'continue':dict(lower=80.,upper=90.,complete=False,mean=None)},
        reasons={},root_bounds={},prefix_cost=dict(process_cpu=.05),limits={},screening_workers=4)
    bank['counters'].update({name:0 for name in ('cache_hits','whole_tail_integrations','whole_tail_hits',
        'destination_bounds','destination_bound_hits','budget_unknowns','unsupported_histories')})
    bank['complete_cost']['wall']=1.9
    documents={};old_records=[]
    for item in frozen['rows']:
        q=item['query_index'];root=item['state']['root_index']
        item.update(source_state_index=q,selection_reasons=[],selected_actions=[],occurrences=[dict(
            policy='cheap',selected_action='alight',state_index=0,source=f'roots/{root:03}_cheap.json.gz')])
        item['state'].update(day='test',clock=100.,deadline=200.,kind='onboard')
        old=dict(query_index=q,source_state_index=q,root_index=root,day='test',clock=100.,kind='onboard',
            selection_reasons=[],observed_baseline_actions=[],source='frozen',actual_v51='original',results={})
        for b in budgets:
            old['results'][b['name']]={}
            for mode in modes:
                current=deepcopy(bank);current.update(query_index=q,root_index=root,source_state_index=q,mode=mode,budget=b)
                path=directory/'natural'/f"{q:03}_{b['name']}_{mode}.json.gz";path.touch();documents[path]=current
                previous=module.normalized(current,'old')
                old['results'][b['name']][mode]=previous
        old_records.append(old)
    previous=dict(protocol,records=old_records,all_root_context=[dict(root_index=i) for i in range(72)],
        observed_v51_action_gaps=dict(aggregate={b['name']:{'cheap':{}} for b in budgets},scope='first action'),
        joint_attribution=dict(scope='five charged queries'),
        optimal_certificate_root_choice_alignment=dict(scope='same first action'),gap_scope='conditional')
    documents[directory/'contract.json']=protocol
    documents[tmp_path/'reconstruction_v52/panel.json.gz']=frozen
    documents[tmp_path/'reconstruction_v53/summary.json']=previous
    documents[tmp_path/'reconstruction_v53/summary_compressed.json']=previous
    monkeypatch.setattr(module,'HERE',directory)
    monkeypatch.setattr(module,'read',lambda path:documents[path])
    monkeypatch.setattr(module,'plot',lambda *args:None)
    result=module.summarize()
    assert result['calls']==880 and result['queries']==88
    assert result['audit']['cross_version_cells']==1760
    assert result['joint_attribution']['aggregate']['b2']['mean_cpu_per_joint_state']==pytest.approx(9.)
    assert result['joint_attribution']['aggregate']['b2']['gaps']['cheap_p0_minus_optimal']['statuses']=={
        'computationally_unresolved':88}
    assert result['joint_attribution']['aggregate']['b2']['gaps']['cheap_p0_minus_one_step']['zero_queries']==list(range(88))
    assert len(result['all_root_context'])==72
    supplemental=result['matched_action_attribution']['aggregate']
    assert set(supplemental)=={'v54','v53_impact','v53_compressed'}
    assert supplemental['v54']['b2']['states']==88
    assert supplemental['v54']['b2']['mean_cpu_per_joint_state']==pytest.approx(9.)
