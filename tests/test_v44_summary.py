from copy import deepcopy

import pytest

from experiments.reconstruction_v44 import summarize


def row(loss=10.,cpu=.2,wall=.3):
    return dict(root_index=0,label='matched128',outcome=dict(loss=loss),saved_answers_reused=0,
        decision_costs=[dict(process_cpu=cpu*.75,wall=wall*.75)],
        cost=dict(total_process_cpu=cpu,total_wall=wall,decision_cpu_sum=cpu*.75,
                  final_execution_cpu=cpu*.25,final_execution_wall=wall*.25))


def test_budget_counts_whole_journeys_and_cpu_wall_separately():
    subjects=[row(cpu=.2,wall=.4),row(cpu=.05,wall=.05)]
    baseline=[row(cpu=.05,wall=.05),row(cpu=.4,wall=.4)]
    cpu=summarize.decision_budget(subjects,'process_cpu',.25)
    wall=summarize.decision_budget(subjects,'wall',.25)
    assert cpu['all_decisions_within_roots']==2 and wall['all_decisions_within_roots']==1
    joint=summarize.joint_budgets(subjects,baseline)
    assert joint['process_cpu']['0.25']['both_within']==1
    assert joint['wall']['0.25']['subject_only']==1
    assert joint['wall']['0.25']['baseline_only']==1


def test_compute_gate_cannot_pass_by_trading_a_root_harm_for_lower_runtime():
    subject=[row(loss=11.,cpu=.05,wall=.05)]
    baseline=[row(loss=10.,cpu=.2,wall=.3)]
    result=summarize.net_compute(subject,baseline)
    assert result['lower_aggregate_cpu_and_wall'] and not result['passed']
    subject[0]['outcome']['loss']=10.
    assert summarize.net_compute(subject,baseline)['passed']
    assert summarize.validate_partition(subject[0])['exact']
    subject[0]['cost']['final_execution_wall']+=.1
    with pytest.raises(ValueError,match='cost partition'):
        summarize.validate_partition(subject[0])


def gates():
    fields=dict(cid=2,clock=10,public_history=[dict(observed=3)],seed=7,source=None,count=2,generated_count=2,
                base_action='continue',selected_action='continue')
    matched=dict(fields,rows={arm:[dict(world_index=i,value=summarize.point(loss)) for i,loss in enumerate(values)]
                              for arm,values in (('alight',[10.,12.]),('continue',[8.,10.]))})
    full=dict(fields,rows={arm:[dict(world_index=i,loss=loss) for i,loss in enumerate(values)]
                           for arm,values in (('alight',[12.,14.]),('continue',[8.,10.]))})
    return dict(gates=[matched]),dict(gates=[full])


def test_common_gate_matching_uses_public_history_and_retains_paired_gap_covariance():
    matched,full=gates()
    result=summarize.common_gate_comparison(matched,full,0)
    bridge=result['common'][0]['matched_minus_full_gap']
    assert bridge['mean']==-2. and bridge['paired_se']==0.
    altered=deepcopy(full);altered['gates'][0]['public_history']=[dict(observed=4)]
    result=summarize.common_gate_comparison(matched,altered,0)
    assert not result['common'] and result['unmatched_candidate_gate_indices']==[0]
    altered=deepcopy(full);altered['gates'][0]['seed']=8
    with pytest.raises(ValueError,match='paired sample identity'):
        summarize.common_gate_comparison(matched,altered,0)


def test_single_level_target_validation_includes_unresolved_probability():
    saved=summarize.point(10.)
    fresh=dict(saved,mean=None,unresolved_probability=.1)
    assert not summarize.local_value_equal(fresh,saved)
    assert summarize.local_value_equal(dict(saved,counters=dict(events=10)),saved)


def test_missing_any_declared_reference_or_root_prevents_formal_summary(tmp_path,monkeypatch):
    monkeypatch.setattr(summarize,'HERE',tmp_path)
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(SystemExit,match='Awaiting 508 required V44 receipts'):
        summarize.main()
    assert not (tmp_path/'summary.json').exists()
