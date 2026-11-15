"""Structural accounting must retain adverse and unresolved reference results."""
from copy import deepcopy

import pytest

import experiments.reconstruction_v55.summarize as summary


def interval(low, high):
    return dict(lower=low, upper=high, complete=False, mean=None)


def receipt():
    counters = dict(states=1800, cache_hits=0, whole_tail_integrations=0, whole_tail_hits=0,
        destination_bounds=0, destination_bound_hits=0, budget_unknowns=1, unsupported_histories=0)
    counters.update(dict.fromkeys(summary.STRUCTURAL_COUNTERS, 0))
    counters.update(terminal_dp_calls=2, terminal_dp_admitted=1, terminal_dp_complete=1, terminal_dp_states=7)
    return dict(query_index=0, source_state_index=0, root_index=0, day='test', mode='cheap_working',
        budget=dict(name='b2', states=2000, process_cpu_seconds=2), clock=100., deadline=200.,
        future_public_frames=0, implementation='structural_conditional',
        value=interval(20, 25), actions={'alight': interval(20, 25), 'continue': interval(80, 90)},
        selected_action='alight', cheap_action='alight', certified_action='alight', counters=counters,
        reasons={}, root_bounds={}, cost=dict(process_cpu=1.8, wall=1.85),
        complete_cost=dict(process_cpu=1.9, wall=1.95), prefix_cost=dict(process_cpu=.1, wall=.1),
        limits={}, screening_workers=4, semantic=dict(dropped_views=3),
        observation_contract=dict(station_to_cids=[]))


def item(query=0):
    root = query % 72
    return dict(query_index=query, source_state_index=query, selection_reasons=[], selected_actions=[],
        context=dict(observation_contract=dict(station_to_cids=[])),
        state=dict(root_index=root, day='test', kind='onboard', clock=100., deadline=200.),
        occurrences=[dict(policy='cheap', selected_action='alight', state_index=0,
                          source=f'roots/{root:03}_cheap.json.gz#/states/0')])


def test_paired_summary_keeps_worse_width_cpu_and_lost_certificates():
    old = summary.normalized(receipt(), 'old'); new = deepcopy(old)
    new.update(value=interval(19, 30), certified_action=None, cost=dict(process_cpu=2.5))
    rows = [summary.paired_cell(0, 0, 'b2', 'optimal', 'impact', old, new),
            summary.paired_cell(1, 1, 'b2', 'optimal', 'impact', new, old)]
    result = summary.paired_aggregate(rows)
    assert result['narrower'] == result['wider'] == 1
    assert result['lost_certificate_queries'] == [0]
    assert result['gained_certificate_queries'] == [1]
    assert rows[0]['width_reduction'] == -6 and rows[0]['cpu_change'] == pytest.approx(.6)
    assert 'root_refinement' not in old and old['semantic']['dropped_views'] == 3


@pytest.mark.parametrize('change,pattern', [
    (lambda b: b['counters'].update(states=2001), 'state budget'),
    (lambda b: b.update(future_public_frames=1), 'information contract'),
    (lambda b: b.update(observation_contract={}), 'observation contract'),
    (lambda b: b.update(implementation='refining_conditional'), 'wrong implementation'),
    (lambda b: b['complete_cost'].update(process_cpu=1.7), 'omitted complete'),
    (lambda b: b.update(cheap_action='continue'), 'root choice changed'),
])
def test_receipt_rejects_changed_budget_information_cost_or_actual_working_choice(change, pattern):
    bank = receipt(); change(bank)
    with pytest.raises(ValueError, match=pattern):
        summary.validate_receipt(bank, item(), receipt()['budget'], 'cheap_working')


def full_panel(tmp_path, monkeypatch):
    directory = tmp_path/'reconstruction_v55'; (directory/'natural').mkdir(parents=True)
    modes = ['optimal', 'cheap_p0', 'cheap_working', 'one_step_p0', 'two_step_p0']
    budgets = [dict(name='b2', states=2000, process_cpu_seconds=2),
               dict(name='b10', states=20000, process_cpu_seconds=10)]
    policies = ['cheap', 'mean', 'single', 'pair', 'one_step', 'snapshot_csa', 'matched128', 'full128']
    protocol = dict(modes=modes, budgets=budgets)
    panel = dict(rows=[item(q) for q in range(88)], missing_first_nontrivial_roots=[],
                 all_state_count=507, same_history_disagreement_states=36)
    docs = {directory/'contract.json': protocol, tmp_path/'reconstruction_v52/panel.json.gz': panel}
    old_records = []
    for entry in panel['rows']:
        q, root = entry['query_index'], entry['state']['root_index']
        old = dict(query_index=q, source_state_index=q, root_index=root, day='test', kind='onboard', clock=100.,
            selection_reasons=[], observed_baseline_actions=[], source='frozen', actual_v51=f'root/{root}', results={})
        for budget in budgets:
            level = budget['name']; old['results'][level] = {}
            for mode in modes:
                bounds = {'optimal': (10, 20), 'two_step_p0': (15, 30), 'one_step_p0': (30, 50),
                          'cheap_p0': (30, 50), 'cheap_working': ((20, 25), (60, 70), (20, 60))[q % 3]}
                bank = receipt(); bank.update(query_index=q, source_state_index=q, root_index=root,
                    mode=mode, budget=budget, value=interval(*bounds[mode]))
                bank['actions']['alight'] = interval(*bounds[mode])
                path = directory/'natural'/f'{q:03}_{level}_{mode}.json.gz'; path.touch(); docs[path] = bank
                old['results'][level][mode] = summary.normalized(bank, 'old')
        old_records.append(old)
    context = [dict(root_index=root, actual_v51={policy: dict(loss=100+root, status='arrived')
               for policy in policies}) for root in range(72)]
    previous = dict(protocol, records=old_records, all_root_context=context,
        observed_v51_action_gaps=dict(aggregate={b['name']: dict.fromkeys(policies, {}) for b in budgets}, scope='actual first action'),
        joint_attribution=dict(scope='five paid queries'),
        optimal_certificate_root_choice_alignment=dict(scope='first action only'), gap_scope='conditional')
    docs[tmp_path/'reconstruction_v53/summary.json'] = previous
    docs[tmp_path/'reconstruction_v53/summary_compressed.json'] = previous
    monkeypatch.setattr(summary, 'HERE', directory)
    monkeypatch.setattr(summary, 'read', lambda path: docs[path])
    monkeypatch.setattr(summary, 'plot', lambda *args: None)
    return docs, directory, previous


def test_all_880_receipts_keep_context_signed_cases_matched_arms_and_full_costs(tmp_path, monkeypatch):
    _, directory, previous = full_panel(tmp_path, monkeypatch)
    result = summary.summarize()
    assert result['queries'] == 88 and result['calls'] == 880
    assert result['audit']['cross_version_cells'] == 1760
    assert result['audit']['cross_budget_cells'] == 440
    assert result['audit']['original_working_root_action_checks'] == 176
    assert result['audit']['observation_contract_checks'] == 880
    assert result['audit']['working_root_action_checks_impact'] == 176
    assert result['audit']['working_root_action_checks_compressed'] == 176
    assert result['all_root_context'] == previous['all_root_context']
    assert not result['g2_passed'] and not result['new_method_contribution_established']
    assert result['aggregate']['b2']['optimal']['counters_total']['terminal_dp_states'] == 88*7
    signed = result['gap_summary']['b2']['working_minus_consistent_p0']
    assert signed['negative_queries'] == list(range(0, 88, 3))
    assert signed['positive_queries'] == list(range(1, 88, 3))
    assert signed['statuses']['computationally_unresolved'] == 29
    joint = result['joint_attribution']['aggregate']['b2']
    assert joint['mean_cpu_per_joint_state'] == pytest.approx(9.5)
    assert joint['gaps']['cheap_p0_minus_one_step']['zero_queries'] == list(range(88))
    assert joint['gaps']['cheap_p0_minus_optimal']['positive_queries'] == list(range(88))
    matched = result['matched_action_attribution']['aggregate']
    assert set(matched) == {'v55', 'v53_impact', 'v53_compressed'}
    assert matched['v55']['b2']['states'] == 88
    assert matched['v55']['b2']['mean_cpu_per_joint_state'] == pytest.approx(9.5)
    assert result['observed_v51_action_gaps']['aggregate']['b2']['cheap']['certified_zero_queries'] == list(range(88))
    assert (directory/'same_budget_paired.csv').read_text().count('\n') == 1761


def test_incomplete_panel_is_not_summarized_as_success(tmp_path, monkeypatch):
    _, directory, _ = full_panel(tmp_path, monkeypatch)
    (directory/'natural/087_b10_two_step_p0.json.gz').unlink()
    with pytest.raises(ValueError, match='1 missing receipts'):
        summary.summarize()


def test_disjoint_frozen_reference_fails_before_attribution(tmp_path, monkeypatch):
    docs, directory, _ = full_panel(tmp_path, monkeypatch)
    bank = deepcopy(docs[directory/'natural/000_b2_optimal.json.gz'])
    bank['value'] = interval(25, 28); bank['actions']['alight'] = interval(25, 28)
    docs[directory/'natural/000_b2_optimal.json.gz'] = bank
    with pytest.raises(ValueError, match='disjoint'):
        summary.summarize()


def priority_panel(tmp_path, monkeypatch):
    docs, directory, previous = full_panel(tmp_path, monkeypatch)
    target = directory/'priority'; (target/'natural').mkdir(parents=True)
    docs[directory/'contract_priority.json'] = docs[directory/'contract.json']
    docs[directory/'summary.json'] = previous
    (directory/'summary.json').write_text('ordinary evidence retained')
    for path, bank in list(docs.items()):
        if path.parent == directory/'natural':
            newpath = target/'natural'/path.name; newpath.touch()
            current = deepcopy(bank); current['implementation'] = 'structural_impact_conditional'
            docs[newpath] = current
    # One state was visited only by another V51 controller. The 176 restored
    # W choices must still match V53; actual cheap occurrences number 174.
    docs[tmp_path/'reconstruction_v52/panel.json.gz']['rows'][-1]['occurrences'] = []
    return docs, directory, target


def test_priority_outputs_stay_separate_and_charge_all_three_comparisons(tmp_path, monkeypatch):
    _, directory, target = priority_panel(tmp_path, monkeypatch)
    plots = []
    monkeypatch.setattr(summary, 'plot', lambda *args: plots.append(args[3]))
    result = summary.summarize(priority=True)
    assert result['implementation'] == 'structural_impact_conditional'
    assert result['audit']['cross_version_cells'] == 2640
    assert result['audit']['original_working_root_action_checks'] == 174
    for name in ('impact', 'compressed', 'ordinary'):
        assert result['audit']['working_root_action_checks_'+name] == 176
    assert result['audit']['observation_contract_checks'] == 880
    assert result['ordinary_comparison']['aggregate']['b2']['optimal']['tied'] == 88
    assert set(result['matched_action_attribution']['aggregate']) == {
        'v55_priority', 'v55_ordinary', 'v53_impact', 'v53_compressed'}
    assert result['joint_attribution']['aggregate']['b2']['mean_cpu_per_joint_state'] == pytest.approx(9.5)
    assert (target/'same_budget_paired.csv').read_text().count('\n') == 2641
    assert (target/'ordinary_paired.csv').read_text().count('\n') == 881
    assert '../../reconstruction_v51/' in (target/'observed_action_gaps.csv').read_text()
    assert (directory/'summary.json').read_text() == 'ordinary evidence retained'
    assert plots == [target]
    assert summary.summarize(priority=True) == result
    assert (directory/'summary.json').read_text() == 'ordinary evidence retained'
    assert plots == [target, target]


def test_priority_checks_working_choice_even_without_an_actual_cheap_occurrence(tmp_path, monkeypatch):
    docs, _, target = priority_panel(tmp_path, monkeypatch)
    bank = docs[target/'natural/087_b2_cheap_working.json.gz']
    bank.update(cheap_action='continue', selected_action='continue')
    with pytest.raises(ValueError, match='Working root action differs'):
        summary.summarize(priority=True)
