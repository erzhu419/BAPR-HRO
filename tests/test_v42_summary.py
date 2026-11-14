from copy import deepcopy

import pytest

from experiments.reconstruction_v42 import summarize


def bank():
    return dict(seed=1,count=128,generated_count=128,public_history=[],future_messages=0,
        paired=dict(unresolved_pairs=0,full_bank_mean_gain_interval=[0.,0.]),
        rows={arm:[dict(loss=20,status='arrived',trace=[dict(kind='onboard',cid=3,choice='continue')],
            decisions=dict(onboard=1,platform=0),world_index=0)] for arm in ('alight','continue')},
        base_action='continue',selected_action='continue')


def test_science_comparison_rejects_changed_trace_even_when_loss_matches():
    original=bank()
    subject=deepcopy(original)
    subject['rows']['alight'][0]['trace'][0]['choice']='alight'
    with pytest.raises(ValueError,match='science changed'):
        summarize.require_same_bank(subject,original,'case')


def test_science_comparison_rejects_changed_generated_world_count_and_selection():
    original=bank()
    subject=deepcopy(original)
    subject['generated_count']=16
    with pytest.raises(ValueError,match='science changed'):
        summarize.require_same_bank(subject,original,'case')
    subject=deepcopy(original)
    subject['selected_action']='alight'
    with pytest.raises(ValueError,match='action changed'):
        summarize.require_same_bank(subject,original,'case')


def test_timing_difference_cannot_change_science_and_ratio_uses_total_costs():
    original=bank()
    subject=deepcopy(original)
    subject['rows']['alight'][0]['total_process_cpu']=100.
    summarize.require_same_bank(subject,original,'case')
    result=summarize.paired_cost([dict(process_cpu=2.,wall=2.),dict(process_cpu=3.,wall=4.)],
        [dict(process_cpu=1.,wall=1.),dict(process_cpu=9.,wall=10.)])
    assert result['aggregate_cpu_ratio']==.5
    assert result['faster_states']==1
    assert result['process_cpu_difference']['total']==-5.
    stats=summarize.total_stats([dict(checks=3,check_cpu=.2,context_units=30),
                                dict(checks=2,check_cpu=.1,context_units=20)])
    assert stats['checks']==5 and stats['context_units']==50
    assert stats['check_cpu']==pytest.approx(.3)


def test_missing_frozen_receipts_cannot_emit_partial_formal_results(tmp_path,monkeypatch):
    monkeypatch.setattr(summarize,'HERE',tmp_path)
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(SystemExit,match='Awaiting 1472 declared receipts'):
        summarize.main()
    assert not (tmp_path/'summary.json').exists()
    assert not (tmp_path/'controls.csv').exists()
