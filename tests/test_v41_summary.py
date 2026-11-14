import pytest

from experiments.reconstruction_v41 import summarize


def root(index, loss, cpu=1., day='a'):
    return dict(root_index=index, root_id=str(index), day=day, origin='A', goal='B', loss_cap=100.,
        outcome=dict(loss=loss, status='arrived' if loss is not None else 'unsupported_public_history:example',
                     trace=[], decisions={}),
        cost=dict(total_process_cpu=cpu, total_wall=cpu, cold=dict(process_cpu=.1,wall=.1),
                  decision_cpu_sum=.75*cpu, final_execution_cpu=.25*cpu),
        decision_costs=[], gates=[], publications=[], saved_answers_reused=0)


def test_full_cohort_gain_and_single_observation_uncertainty_do_not_hide_missing_mass():
    values=summarize.action_value([dict(loss=20.,status='arrived'),dict(loss=None,status='projection_incompatible:clock')],100.)
    assert values['full_bank_mean'] is None
    assert values['full_bank_mean_interval'] == [10.,60.]
    assert values['resolved_sample'] == dict(n=1,mean=20.,se=None)
    comparison=summarize.pair_summary([summarize.pair(root(0,20.),root(0,30.)),
                                      summarize.pair(root(1,None),root(1,40.))])
    assert comparison['full_cohort_mean_gain'] is None
    assert comparison['full_cohort_mean_gain_interval'] == [-25.,25.]
    assert comparison['denominator'] == 2 and comparison['unresolved_pairs'] == 1


def test_signed_day_concentration_keeps_harms_and_no_net_share_when_cohort_missing():
    rows=[summarize.pair(root(0,10.,day='a'),root(0,40.,day='a')),
          summarize.pair(root(1,30.,day='b'),root(1,20.,day='b'))]
    result=summarize.grouped_comparison(rows)
    assert result['wins'] == result['harms'] == 1
    assert result['concentration']['day']['a']['share_of_full_cohort_net_gain'] == 1.5
    assert result['concentration']['day']['b']['share_of_full_cohort_net_gain'] == -.5
    rows.append(summarize.pair(root(2,None),root(2,20.)))
    assert summarize.grouped_comparison(rows)['concentration']['day']['a']['share_of_full_cohort_net_gain'] is None


def test_nested_phase_and_gate_costs_are_subsets_of_whole_journey_total():
    row=root(0,20.,cpu=2.)
    row['gates']=[dict(count=16,generated_count=128,selected_action='alight',base_action='continue',source=None,
        paired=dict(unresolved_pairs=1),rows=dict(alight=[dict(status='arrived')],
            continue_=[dict(status='projection_incompatible:clock')]),
        cost=dict(total_process_cpu=.8,conditioning=dict(process_cpu=.1),generation=dict(process_cpu=.2),
            planner_setup=dict(process_cpu=.05),continuations=dict(alight=dict(process_cpu=.2),
                continue_=dict(process_cpu=.2))),
        meeting_stats=dict(checks=2,hits=1,avoided_decisions=1,check_cpu=.03))]
    gate=row['gates'][0]
    gate['cost']['continuations']['continue']=gate['cost']['continuations'].pop('continue_')
    compact=summarize.compact_root(row,'root')
    result=summarize.control_summary([compact])
    assert result['process_cpu']['total'] == 2.
    assert result['cold_cpu_subset']['total'] == .1
    assert result['gates']['gate_cpu_subset'] == .8
    assert result['gate_phases_cpu']['meeting_check'] == .03
    assert result['gates']['evaluated_pairs'] == 16 and result['gates']['generated_pairs'] == 128
    assert result['all_cost_partitions_exact']


def test_incomplete_declared_receipts_do_not_emit_formal_summary_or_figure(tmp_path,monkeypatch):
    monkeypatch.setattr(summarize,'HERE',tmp_path)
    (tmp_path/'protocol.json').write_text('{"online_configurations":["physical_cheap"]}')
    with pytest.raises(SystemExit,match='Awaiting'):
        summarize.main()
    assert not (tmp_path/'summary.json').exists()
    assert not (tmp_path/'quality_cpu.png').exists()
