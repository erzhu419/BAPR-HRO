from experiments.reconstruction_v40.summarize import (
    budget_feasibility, cohort_quality, compact_root, pair_row, paired_summary, prefix_record,
)


def root(index, loss, cap=100., cpu=1.):
    return dict(root_index=index, root_id=str(index), day='day', loss_cap=cap,
        outcome=dict(loss=loss, status='arrived' if loss is not None else 'unresolved', trace=[]),
        cost=dict(total_process_cpu=cpu, total_wall=cpu), decision_costs=[], gates=[], saved_answers_reused=0)


def test_missing_outcome_keeps_full_cohort_loss_and_paired_gain_denominators():
    subjects = [root(0, 20.), root(1, None)]
    baselines = [root(0, 30.), root(1, 40.)]
    quality = cohort_quality(subjects)
    assert quality['denominator'] == 2 and quality['unresolved'] == 1
    assert quality['full_cohort_mean_loss'] is None
    assert quality['full_cohort_mean_loss_interval'] == [10., 60.]
    assert quality['resolved_only_descriptive']['mean'] == 20.
    paired = paired_summary([pair_row(a, b) for a, b in zip(subjects, baselines)])
    assert paired['full_cohort_mean_gain'] is None
    assert paired['full_cohort_mean_gain_interval'] == [-25., 25.]
    assert paired['wins'] == 1 and paired['unresolved_pairs'] == 1


def test_quota_feasibility_charges_late_work_and_includes_nongate_decisions():
    row = root(0, 20.)
    row['decision_costs'] = [dict(process_cpu=.04, gate=False), dict(process_cpu=.07, gate=True)]
    result = budget_feasibility([row], .05)
    assert result['within_fraction'] == .5
    assert result['exceeded'] == 1 and result['gate_within'] == 0
    assert result['roots_all_decisions_within'] == 0
    assert abs(result['overrun_seconds']['total']-.02) < 1e-12


def test_full_bank_prefix_does_not_admit_a_late_pair_into_a_quota_snapshot():
    bank = dict(actions=dict(base='continue'),
        prefixes=[dict(count=1, elapsed_cpu=.04, selected_action='continue', full_bank_mean_gain_interval=[1,1]),
                  dict(count=2, elapsed_cpu=.06, selected_action='alight', full_bank_mean_gain_interval=[-1,-1])],
        rows=dict(alternative=[dict(loss=1),dict(loss=0)], base=[dict(loss=0),dict(loss=3)]))
    at_quota = prefix_record(bank, quota=.05)
    assert at_quota['count'] == 1 and at_quota['selected_action'] == 'continue'
    assert prefix_record(bank, quota=.03)['count'] == 0
    assert prefix_record(bank, count=2)['selected_action'] == 'alight'


def test_compaction_never_adds_overlapping_gate_cpu_to_root_cost():
    row = root(0, 20., cpu=2.)
    row['gates'] = [dict(count=1, executed_count=2, late_pair_count=1, blocks=[dict(count=128)],
        max_count=512, selected_action='continue', actions=dict(base='continue', alternative='alight'),
        source=None, seed_origin='new_public_event', stop_reason='late_pair',
        paired=dict(unresolved_pairs=0), cost=dict(total_process_cpu=100.))]
    compact = compact_root(row, 'receipt')
    assert compact['cost']['total_process_cpu'] == 2.
    assert compact['gates'][0]['count'] == 1
    assert compact['gates'][0]['generated_count'] == 128
