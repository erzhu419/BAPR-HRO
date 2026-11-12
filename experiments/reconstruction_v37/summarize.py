"""Saved-bank accounting and fixed-policy comparisons for the V37 replay."""

from collections import Counter
from functools import lru_cache
from math import isclose
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.reconstruction_v34.summarize import PUBLIC, LOCAL, arm_summary, first_divergence, numeric, read, write
from experiments.reconstruction_v36.summarize import compare


HERE = Path(__file__).resolve().parent
V34, V36 = HERE.parent/'reconstruction_v34', HERE.parent/'reconstruction_v36'
NAME = 'onboard_complete_continuation_rollout'
OWN = 'fixed_own_only_nominal'


@lru_cache(maxsize=None)
def receipt(path):
    return read(Path(path))


def source_path(value):
    path = Path(value)
    return path if path.is_absolute() else HERE/path


def source_name(path):
    return str(Path(path).resolve())


def conditional_summary(bank):
    pairs = [(a['loss'], b['loss']) for a, b in zip(bank['rows']['alternative'], bank['rows']['base'])
             if a['loss'] is not None and b['loss'] is not None]
    stats, bound = bank['paired']['complete_pair_statistics'], bank['paired']['full_bank_mean_gain_interval']
    expected = bank['actions']['alternative'] if bound[1] < 0 else bank['actions']['base']
    return dict(base_action=bank['actions']['base'], alternative_action=bank['actions']['alternative'],
        nominal_action=bank['actions']['nominal_diagnostic'], selected_action=bank['selected_action'],
        cheap_agreed=bank['cheap_agreed'], switch=bank['selected_action'] != bank['actions']['base'],
        rule_consistent=bank['selected_action'] == expected,
        full_bank_mean_D_interval=bound, mean_D_complete=None if stats is None else stats['mean_gain'],
        complete_pair_SE=None if stats is None else stats['se'],
        complete_pair_CI95_D=None if stats is None else stats['ci95'],
        complete_pairs=len(pairs), unresolved_pairs=bank['paired']['unresolved_pairs'],
        alternative_wins=sum(a < b for a, b in pairs), base_wins=sum(a > b for a, b in pairs),
        complete_pair_ties=sum(a == b for a, b in pairs), worlds=bank['count'], seed=bank['seed'],
        retained_action_worlds={name: len(values) for name, values in bank['rows'].items()},
        cost_cpu=bank['cost']['total_process_cpu'], cost_wall=bank['cost']['total_wall'])


def first_difference(subject, baseline, new_file, old_file, old_location):
    result = first_divergence(subject['outcome'], baseline['outcome'])
    if result['first'] is not None:
        index = result['first']['trace_index']
        result['first'].update(subject_source=dict(file=new_file, location=f'outcome.trace[{index}]'),
            baseline_source=dict(file=old_file, location=f'{old_location}.trace[{index}]'))
    return result


def gate_records(roots, pairings):
    gates, uses, root_accounting = [], {}, []
    for root in roots:
        index, costs = root['root_index'], root['cost']
        hits_cpu, hits_wall = 0., 0.
        for gate in root['gates']:
            path = source_path(gate['receipt_source'])
            bank = receipt(str(path))
            relative = source_name(path)
            state = gate['context']
            event_index = next(i for i, event in enumerate(root['outcome']['trace'])
                if event['kind'] == 'onboard' and event['cid'] == state['cid'] and event['clock'] == state['clock'])
            event = root['outcome']['trace'][event_index]
            identical = all(gate[key] == bank[key] for key in (
                'state', 'public_history', 'actions', 'count', 'seed', 'selected_action', 'paired'))
            state_matches = bank['state'] == state and event['public_history'] == state['public_history']
            if gate['reused']:
                hits_cpu += bank['cost']['total_process_cpu']
                hits_wall += bank['cost']['total_wall']
            row = dict(root_index=index, root_id=root['root_id'], day=root['day'],
                actual_call_index=gate['gate_call_index'], source_call_index=gate['source_evaluation_call_index'],
                current_cid=state['cid'], clock=state['clock'], trace_index=event_index,
                source=relative, reused=gate['reused'], proposed_seed=gate['proposed_seed'],
                preserved_seed_differs_from_current_ordinal=gate['seed'] != gate['proposed_seed'],
                exact_source_receipt_match=identical, exact_state_and_actual_history_match=state_matches,
                actual_action_matches_selected=event['choice'] == gate['selected_action'],
                **conditional_summary(bank),
                actual_trace=dict(file=str(HERE/'roots'/f'{index:03}.json.gz'), location=f'outcome.trace[{event_index}]'),
                root_paired_gains={method: pairings[method]['pairs'][index]['gain_seconds'] for method in pairings},
                actual_scope='Gains refer to the whole root policy replay; no isolated actual action counterfactual is claimed.')
            gates.append(row)
            uses.setdefault(relative, []).append(dict(root_index=index, call=gate['gate_call_index'], trace_index=event_index))
        root_accounting.append(dict(root_index=index, actual_cpu=costs['actual_process_cpu'],
            charged_cpu=costs['total_process_cpu'], added_reuse_cpu=costs['reused_evaluation_cpu'],
            actual_wall=costs['actual_wall'], charged_wall=costs['total_wall'],
            reuse_fee_matches_sources=isclose(hits_cpu, costs['reused_evaluation_cpu'], abs_tol=1e-9)
                and isclose(hits_wall, costs['reused_evaluation_wall'], abs_tol=1e-9),
            charged_equals_actual_plus_hits=isclose(costs['total_process_cpu'], costs['actual_process_cpu']+hits_cpu, abs_tol=1e-9)
                and isclose(costs['total_wall'], costs['actual_wall']+hits_wall, abs_tol=1e-9)))
    return gates, uses, root_accounting


def gate_counts(rows):
    return dict(eligible_calls=len(rows), reused=sum(r['reused'] for r in rows),
        newly_evaluated=sum(not r['reused'] for r in rows), cheap_agreed=sum(r['cheap_agreed'] for r in rows),
        switches=sum(r['switch'] for r in rows),
        switches_while_cheap_agreed=sum(r['switch'] and r['cheap_agreed'] for r in rows),
        consumed_worlds=sum(r['worlds'] for r in rows), complete_pairs=sum(r['complete_pairs'] for r in rows),
        unresolved_pairs=sum(r['unresolved_pairs'] for r in rows),
        alternative_wins=sum(r['alternative_wins'] for r in rows), base_wins=sum(r['base_wins'] for r in rows),
        complete_pair_ties=sum(r['complete_pair_ties'] for r in rows),
        all_rules_consistent=all(r['rule_consistent'] for r in rows),
        all_exact_receipts=all(r['exact_source_receipt_match'] for r in rows),
        all_actual_histories_match=all(r['exact_state_and_actual_history_match'] for r in rows),
        all_actual_actions_match=all(r['actual_action_matches_selected'] for r in rows),
        source_seed_retained_after_different_ordinal=sum(r['preserved_seed_differs_from_current_ordinal'] for r in rows))


def draw(current, baselines):
    groups = {**baselines, NAME: current}
    figure, ax = plt.subplots(figsize=(9, 5))
    for (method, rows), color in zip(groups.items(), plt.get_cmap('tab10').colors):
        stats = arm_summary(rows)
        ax.scatter(stats['cold_total_process_cpu_seconds']['mean'], stats['loss_seconds']['mean'],
            color='black' if method == OWN else color, marker='*' if method == NAME else ('D' if method == OWN else 'o'),
            s=180 if method == NAME else 75, label=method)
    ax.set_xscale('log')
    ax.set(xlabel='Mean fully charged cold control CPU (seconds, log scale)',
           ylabel='Mean truncated journey loss (seconds)', title='72 fixed roots: complete-continuation rollout and fixed controls')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='best')
    figure.text(.5, .015, 'Saved conditional work is charged on every policy use. '
        'Own-only nominal uses a different information set. Logical replay clocks remain unchanged.',
        ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .05, 1, 1))
    figure.savefig(HERE/'quality_cpu.png', dpi=180)
    plt.close(figure)


def main():
    selection, reference = read(V34/'selection.json')['selected'], read(HERE/'reference_summary.json')
    if reference['completed_states'] != reference['expected_states']:
        raise SystemExit('Awaiting all frozen reference banks')
    methods = list(read(V36/'protocol.json')['paired_replay']['methods'])
    roots, previous, own, missing = [], [], [], []
    for index, selected in enumerate(selection):
        path = HERE/'roots'/f'{index:03}.json.gz'
        if not path.exists():
            missing.append(index)
            continue
        root, old, local = read(path), read(V36/'roots'/path.name), read(V34/'roots'/path.name)
        if root['root_index'] != index or any(r['root_id'] != selected['root_id'] for r in (root, old, local)):
            raise ValueError(f'Root {index} differs from the frozen cohort')
        roots.append(root)
        previous.append(old)
        own.append(local)
    if missing:
        raise SystemExit(f'Awaiting {len(missing)}/{len(selection)} fixed policy roots')
    indices = list(range(len(roots)))
    current = [dict(outcome=r['outcome'], cost=r['cost']) for r in roots]
    baselines = {m: [r['rows'][PUBLIC][m] for r in previous] for m in methods}
    baselines[OWN] = [r['rows'][LOCAL]['rolling_nominal'] for r in own]
    pairs = {method: compare(current, values, indices) for method, values in baselines.items()}
    gates, uses, accounting = gate_records(roots, pairs)
    unique_sources = sorted(set(uses))
    references, new_policy = [], []
    for case in reference['cases']:
        source = source_name(HERE/case['source'])
        bank = receipt(str(HERE/source))
        references.append(dict(state_id=case['state_id'], root_index=case['root_index'], day=case['day'],
            clock=case['clock'], source=source, **conditional_summary(bank),
            original_alight_minus_continue_seconds=case['original_alight_minus_continue_seconds'],
            policy_uses=uses.get(source, []), actual_gain_scope='Only linked whole-policy root gains are available.'))
    for path in sorted((HERE/'policy_cases').glob('*.json.gz')):
        bank = receipt(str(path))
        new_policy.append(dict(source=source_name(path), root_index=bank['state']['root_index'],
            day=bank['state']['day'], **conditional_summary(bank)))
    first_differences = []
    for index, root in enumerate(roots):
        for method in baselines:
            oldfile = str((V34 if method == OWN else V36)/'roots'/f'{index:03}.json.gz')
            location = f'rows.{LOCAL}.rolling_nominal.outcome' if method == OWN else f'rows.{PUBLIC}.{method}.outcome'
            difference = first_difference(current[index], baselines[method][index], str(HERE/'roots'/f'{index:03}.json.gz'), oldfile, location)
            first_differences.append(dict(root_index=index, day=root['day'], baseline=method,
                information='different: own/local only' if method == OWN else 'same complete public information and same model',
                **difference, paired_result=pairs[method]['pairs'][index]))
    daily = {}
    for day in sorted({r['day'] for r in roots}):
        mask = [i for i, r in enumerate(roots) if r['day'] == day]
        daily[day] = dict(denominator=len(mask), control=arm_summary([current[i] for i in mask]),
            actual_control_CPU_seconds=numeric([roots[i]['cost']['actual_process_cpu'] for i in mask]),
            comparisons={m: compare([current[i] for i in mask], [values[i] for i in mask], mask)
                         for m, values in baselines.items()},
            gates=gate_counts([g for g in gates if g['day'] == day]),
            reference_states=sum(r['day'] == day for r in references),
            reference_alternative_selected=sum(r['day'] == day and r['switch'] for r in references))
    unique_used = [conditional_summary(receipt(str(HERE/path))) for path in unique_sources]
    result = dict(method=NAME, development_only=True, denominator=len(roots),
        attribution='Standard complete-continuation rollout baseline; no new algorithm claim.',
        actual_gain_orientation='Fixed control loss minus new policy loss: positive favors new policy.',
        conditional_orientation='D=alternative minus base loss: negative favors alternative. '
            'Alternative/base win counts are recomputed from saved rows, not copied from the legacy wins field.',
        control=arm_summary(current), baselines={m: arm_summary(v) for m, v in baselines.items()},
        comparisons=pairs, by_day=daily,
        timing=dict(charged_control_cpu_seconds=numeric([r['cost']['total_process_cpu'] for r in roots]),
            actual_control_cpu_seconds=numeric([r['cost']['actual_process_cpu'] for r in roots]),
            charged_control_wall_seconds=numeric([r['cost']['total_wall'] for r in roots]),
            actual_control_wall_seconds=numeric([r['cost']['actual_wall'] for r in roots]),
            added_saved_evaluation_cpu_seconds=sum(r['cost']['reused_evaluation_cpu'] for r in roots),
            pipeline_actual_cpu_seconds=numeric([r['pipeline_actual_process_cpu'] for r in roots]),
            all_reuse_fees_match_sources=all(r['reuse_fee_matches_sources'] for r in accounting),
            all_charged_equal_actual_plus_hits=all(r['charged_equals_actual_plus_hits'] for r in accounting),
            accounting='New-bank cost is already in actual policy CPU. Only reused-bank original evaluation '
                'CPU is added to obtain charged cost. Reference preparation is separate; reference-bank '
                'totals are not added a second time to the policy.'),
        gate_accounting=gate_counts(gates),
        unique_bank_accounting=dict(reference_banks=len(references), reference_worlds=sum(r['worlds'] for r in references),
            new_policy_banks=len(new_policy), new_policy_worlds=sum(r['worlds'] for r in new_policy),
            policy_unique_used_banks=len(unique_sources), policy_unique_used_worlds=sum(r['worlds'] for r in unique_used),
            entire_stage_unique_banks=len(references)+len(new_policy),
            entire_stage_unique_worlds=sum(r['worlds'] for r in references+new_policy),
            entire_stage_unique_action_continuations=sum(sum(r['retained_action_worlds'].values()) for r in references+new_policy),
            entire_stage_unique_complete_pairs=sum(r['complete_pairs'] for r in references+new_policy),
            entire_stage_unique_unresolved_pairs=sum(r['unresolved_pairs'] for r in references+new_policy)),
        reference=dict(source=str(HERE/'reference_summary.json'), states=len(references),
            alternative_selected=sum(r['switch'] for r in references),
            all_rules_consistent=all(r['rule_consistent'] for r in references)),
        root_cost_accounting=accounting, diagnostics=str(HERE/'state_diagnostics.json'),
        limitations=['The full-bank bounds account for missing outcomes in one empirical bank; they are not population significance or regret certificates.',
            'A reference ranking reversal is model-conditional. Actual gains belong to whole journeys and cannot be assigned independently to every encountered state.',
            'All roots and negative outcomes are retained. Own-only nominal has a different information set.',
            'Physical quality uses logical clocks; fully charged CPU does not establish real-time boarding feasibility.'])
    write(HERE/'summary.json', result)
    write(HERE/'state_diagnostics.json', dict(reference_states=references, policy_gates=gates,
        full_action_comparisons=first_differences, source_reuse=uses))
    draw(current, baselines)
    print(dict(roots=len(roots), policy_gate_accounting=result['gate_accounting'],
               unique_banks=result['unique_bank_accounting']))


if __name__ == '__main__':
    main()
