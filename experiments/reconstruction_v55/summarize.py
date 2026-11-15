"""All-state structural-reference comparison, with paid policy attribution."""
from collections import Counter
import gzip
import json
from math import isfinite
from pathlib import Path

from experiments.reconstruction_v34.discover import read
from experiments.reconstruction_v33.run import jsonable
from experiments.reconstruction_v38.run import save
from experiments.reconstruction_v53.summarize import (
    TOL, GAPS, aggregate, check_working_root_action, compatible, gap,
    joint_attribution, mean, observed_action_gaps, valid_certificate, valid_value,
    width, write_csv,
)
from experiments.reconstruction_v54.summarize import (
    gap_aggregate, matched_supplement, require_panel,
)

HERE = Path(__file__).resolve().parent
STRUCTURAL_COUNTERS = ('terminal_dp_calls', 'terminal_dp_admitted', 'terminal_dp_complete',
    'terminal_dp_states', 'terminal_dp_budget_stops', 'terminal_dp_hits')


def normalized(bank, source):
    result = {k: bank[k] for k in ('value', 'actions', 'selected_action', 'cheap_action',
        'certified_action', 'counters', 'reasons', 'root_bounds', 'prefix_cost',
        'limits', 'screening_workers')}
    return dict(result, cost=bank['complete_cost'], source=source,
                semantic=bank.get('semantic', {}))


def validate_receipt(bank, item, budget, mode, implementation='structural_conditional'):
    state = item['state']
    if (bank['query_index'], bank['source_state_index'], bank['root_index'], bank['day'], bank['mode']) != (
            item['query_index'], item['source_state_index'], state['root_index'], state['day'], mode):
        raise ValueError('Receipt differs from the frozen state or policy')
    if bank['budget'] != budget or bank['counters']['states'] > budget['states']:
        raise ValueError('Receipt violates the original state budget')
    if bank['implementation'] != implementation:
        raise ValueError('Receipt uses the wrong implementation')
    if (bank['clock'], bank['deadline'], bank['future_public_frames']) != (state['clock'], state['deadline'], 0):
        raise ValueError('Physical clock or information contract changed')
    if bank['observation_contract'] != item['context']['observation_contract']:
        raise ValueError('Actual observation contract differs from the frozen panel')
    for value in (bank['value'], *bank['actions'].values()):
        valid_value(value, state['deadline']-state['clock'])
    valid_certificate(bank)
    for name in ('process_cpu', 'wall'):
        total = bank['complete_cost'][name]
        if not isfinite(total) or total < 0 or total+TOL < bank['cost'][name]:
            raise ValueError('Invalid or omitted complete query cost')
    return check_working_root_action(item['occurrences'], bank) if mode == 'cheap_working' else 0


def working_choice_unchanged(old, new):
    normalize = lambda action: 'wait' if action is None else str(action)
    if any(normalize(old[key]) != normalize(new[key]) for key in ('cheap_action', 'selected_action')):
        raise ValueError('Working root action differs from the frozen reference')


def nested_sources(value):
    """The priority result lives one directory below the ordinary result."""
    if isinstance(value, dict):
        return {key: nested_sources(item) for key, item in value.items()}
    if isinstance(value, list):
        return [nested_sources(item) for item in value]
    if isinstance(value, str) and value.startswith('../reconstruction_'):
        return '../'+value
    return value


def paired_cell(query, root, budget, mode, baseline, old, new):
    return dict(query_index=query, root_index=root, budget=budget, mode=mode, baseline=baseline,
        baseline_width=width(old['value']), v55_width=width(new['value']),
        width_reduction=width(old['value'])-width(new['value']),
        baseline_cpu=old['cost']['process_cpu'], v55_cpu=new['cost']['process_cpu'],
        cpu_change=new['cost']['process_cpu']-old['cost']['process_cpu'],
        baseline_complete=old['value']['complete'], v55_complete=new['value']['complete'],
        baseline_certified=old['certified_action'], v55_certified=new['certified_action'])


def paired_aggregate(rows):
    return dict(cells=len(rows), narrower=sum(r['width_reduction'] > TOL for r in rows),
        wider=sum(r['width_reduction'] < -TOL for r in rows),
        tied=sum(abs(r['width_reduction']) <= TOL for r in rows),
        mean_width_reduction=mean(rows, lambda r: r['width_reduction']),
        mean_cpu_change=mean(rows, lambda r: r['cpu_change']),
        baseline_complete=sum(r['baseline_complete'] for r in rows),
        v55_complete=sum(r['v55_complete'] for r in rows),
        baseline_certificates=sum(r['baseline_certified'] is not None for r in rows),
        v55_certificates=sum(r['v55_certified'] is not None for r in rows),
        gained_certificate_queries=[r['query_index'] for r in rows
            if r['baseline_certified'] is None and r['v55_certified'] is not None],
        lost_certificate_queries=[r['query_index'] for r in rows
            if r['baseline_certified'] is not None and r['v55_certified'] is None])


def attribution(record, item, level, gaps, choices, joints):
    results = record['results'][level]
    record['gaps'][level] = {name: gap(results[left]['value'], results[right]['value'], positive)
                            for name, left, right, positive in GAPS}
    base = dict(query_index=record['query_index'], root_index=record['root_index'], budget=level)
    gaps.extend(dict(base, comparison=name, **entry) for name, entry in record['gaps'][level].items())
    observed = observed_action_gaps(item['occurrences'], results['optimal'])
    record['observed_v51_action_gaps'][level] = observed
    for choice in observed:
        choices.append(dict(base, **{k: v for k, v in choice.items() if k not in ('action_value', 'optimal_value')},
            action_lower=choice['action_value']['lower'], action_upper=choice['action_value']['upper'],
            optimal_lower=choice['optimal_value']['lower'], optimal_upper=choice['optimal_value']['upper']))
    joint = record['joint_attribution'][level] = joint_attribution(results)
    for name, entry in joint['gaps'].items():
        joints.append(dict(base, comparison=name,
            **{k: v for k, v in entry.items() if k != 'certificate_modes'},
            certificate_modes=','.join(entry['certificate_modes']),
            joint_process_cpu=joint['cost']['process_cpu'], charged_queries=joint['cost']['complete_queries']))


def attribution_summary(records, budgets, policies, choices, joints):
    observed, joint, alignment = {}, {}, {}
    for budget in budgets:
        level = budget['name']; observed[level] = {}
        for policy in policies:
            cells = [r for r in choices if (r['budget'], r['policy']) == (level, policy)]
            observed[level][policy] = dict(occurrences=len(cells), queries=len({r['query_index'] for r in cells}),
                statuses=dict(Counter(r['status'] for r in cells)),
                certified_positive_queries=sorted({r['query_index'] for r in cells if r['lower'] > TOL}),
                certified_positive_roots=sorted({r['root_index'] for r in cells if r['lower'] > TOL}),
                certified_zero_queries=sorted({r['query_index'] for r in cells if r['upper'] <= TOL}),
                certificate_implied_zero_occurrences=sum(r['proof_basis'] == 'strict_optimal_action_certificate' for r in cells),
                max_proven_loss_lower=max((r['lower'] for r in cells), default=None))
        joint[level] = dict(states=len(records), queries_per_state=5,
            mean_cpu_per_joint_state=mean(records, lambda r: r['joint_attribution'][level]['cost']['process_cpu']),
            total_charged_cpu=sum(r['joint_attribution'][level]['cost']['process_cpu'] for r in records),
            certified_cheap_one_step_equalities=sum(bool(r['joint_attribution'][level]['equalities']) for r in records),
            gaps={name: gap_aggregate([r for r in joints if (r['budget'], r['comparison']) == (level, name)])
                  for name, *_ in GAPS})
        certified = [r for r in records if r['results'][level]['optimal']['certified_action'] is not None]
        alignment[level] = {}
        for mode in ('cheap_p0', 'cheap_working'):
            same = [r['query_index'] for r in certified if str(r['results'][level]['optimal']['certified_action'])
                    == str(r['results'][level][mode]['cheap_action'])]
            alignment[level][mode] = dict(optimal_certified_queries=len(certified), same_root_choice=len(same),
                different_root_choice=len(certified)-len(same), same_query_indices=same)
    return observed, joint, alignment


def summarize(*, priority=False):
    directory = HERE/'priority' if priority else HERE
    implementation = 'structural_impact_conditional' if priority else 'structural_conditional'
    protocol = read(HERE/('contract_priority.json' if priority else 'contract.json'))
    panel = read(HERE.parent/'reconstruction_v52/panel.json.gz')
    baselines = {name: read(HERE.parent/'reconstruction_v53'/file) for name, file in
                 (('impact', 'summary.json'), ('compressed', 'summary_compressed.json'))}
    if priority:
        baselines['ordinary'] = read(HERE/'summary.json')
    for baseline in baselines.values():
        require_panel(panel, protocol, baseline)
    modes, budgets = protocol['modes'], protocol['budgets']
    paths = [directory/'natural'/f"{r['query_index']:03}_{b['name']}_{m}.json.gz"
             for r in panel['rows'] for b in budgets for m in modes]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise ValueError(f'Complete panel required: {len(missing)} missing receipts; first={missing[0].name}')
    previous = {name: {r['query_index']: r for r in data['records']} for name, data in baselines.items()}
    context = baselines['impact']['all_root_context']
    policies = baselines['impact']['observed_v51_action_gaps']['aggregate'][budgets[0]['name']]
    if (len(context) != 72 or {r['root_index'] for r in context} != set(range(72)) or len(policies) != 8
            or any(set(r['actual_v51']) != set(policies) for r in context)):
        raise ValueError('The V51 all72-root/eight-policy context is incomplete')
    records, flat, pairs, gaps, choices, joints, budget_pairs = [], [], [], [], [], [], []
    audit = Counter()
    for item in panel['rows']:
        query = item['query_index']; root = item['state']['root_index']
        record = {k: previous['impact'][query][k] for k in ('query_index', 'source_state_index',
            'root_index', 'day', 'kind', 'clock', 'selection_reasons', 'observed_baseline_actions', 'source', 'actual_v51')}
        record.update(results={}, gaps={}, observed_v51_action_gaps={}, joint_attribution={})
        for budget in budgets:
            level = budget['name']; results = record['results'][level] = {}
            for mode in modes:
                source = f'natural/{query:03}_{level}_{mode}.json.gz'; bank = read(directory/source)
                audit['original_working_root_action_checks'] += validate_receipt(bank, item, budget, mode, implementation)
                audit['observation_contract_checks'] += 1
                result = results[mode] = normalized(bank, source)
                for name, old in previous.items():
                    prior = old[query]['results'][level][mode]
                    audit.update(compatible(prior, result)); audit['cross_version_cells'] += 1
                    if mode == 'cheap_working':
                        working_choice_unchanged(prior, result)
                        audit['working_root_action_checks_'+name] += 1
                    pairs.append(paired_cell(query, root, level, mode, name, prior, result))
                flat.append(dict(query_index=query, root_index=root, budget=level, mode=mode,
                    lower=bank['value']['lower'], upper=bank['value']['upper'], width=width(bank['value']),
                    complete=bank['value']['complete'], selected_action=bank['selected_action'],
                    certified_action=bank['certified_action'], states=bank['counters']['states'],
                    cpu=bank['complete_cost']['process_cpu'], wall=bank['complete_cost']['wall'],
                    cpu_threshold=budget['process_cpu_seconds'],
                    cpu_overshoot=max(0., bank['complete_cost']['process_cpu']-budget['process_cpu_seconds']),
                    **{k: bank['counters'].get(k, 0) for k in STRUCTURAL_COUNTERS}, source=source))
            attribution(record, item, level, gaps, choices, joints)
        for mode in modes:
            small, large = (record['results'][b['name']][mode] for b in budgets)
            audit.update(compatible(small, large)); audit['cross_budget_cells'] += 1
            budget_pairs.append(dict(query_index=query, root_index=root, mode=mode,
                b2_width=width(small['value']), b10_width=width(large['value']),
                width_reduction=width(small['value'])-width(large['value']),
                b2_cpu=small['cost']['process_cpu'], b10_cpu=large['cost']['process_cpu'],
                b2_certified=small['certified_action'], b10_certified=large['certified_action']))
        records.append(record)
    aggregates = {b['name']: {m: aggregate([r['results'][b['name']][m] for r in records], b)
                             for m in modes} for b in budgets}
    comparisons = {name: {b['name']: {m: paired_aggregate([r for r in pairs
        if (r['baseline'], r['budget'], r['mode']) == (name, b['name'], m)]) for m in modes}
        for b in budgets} for name in baselines}
    observed, joint, alignment = attribution_summary(records, budgets, policies, choices, joints)
    versions = {('v55_priority' if priority else 'v55'): records}
    versions.update({('v55_ordinary' if name == 'ordinary' else 'v53_'+name): data['records']
                     for name, data in baselines.items()})
    matched = matched_supplement(versions, budgets)
    result = dict(stage='V55', implementation=implementation, queries=len(records), calls=len(flat),
        modes=modes, budgets=budgets, original_root_denominator=72, represented_roots=len({r['root_index'] for r in records}),
        missing_roots=panel['missing_first_nontrivial_roots'], all_v51_state_count=panel['all_state_count'],
        all_same_history_disagreements=panel['same_history_disagreement_states'],
        aggregate=aggregates, records=records, all_root_context=context,
        gap_summary={b['name']: {name: gap_aggregate([r for r in gaps
            if (r['budget'], r['comparison']) == (b['name'], name)]) for name, *_ in GAPS} for b in budgets},
        same_budget_comparison=dict(primary_baseline='V53 impact', aggregate=comparisons,
            records='same_budget_paired.csv',
            scope='Every V55 query is separately charged against the listed frozen references at the identical original budget. All narrower, wider, tied and unresolved cases remain.'),
        budget_comparison=dict(records='budget_paired.csv', scope='Separate original b2/b10 queries; no monotonic-width assumption.'),
        observed_v51_action_gaps=dict(aggregate=observed, records='observed_action_gaps.csv',
            scope=baselines['impact']['observed_v51_action_gaps']['scope']),
        joint_attribution=dict(aggregate=joint, records='joint_attribution.csv',
            scope=baselines['impact']['joint_attribution']['scope']),
        matched_action_attribution=dict(aggregate=matched['aggregate'], records='matched_action_attribution.json.gz',
            table='matched_action_attribution.csv', scope='Supplemental matching of same forced cheap-P0/one-step arms within each version and budget; all five query costs charged. No cross-version pooling.'),
        optimal_certificate_root_choice_alignment=dict(aggregate=alignment,
            scope=baselines['impact']['optimal_certificate_root_choice_alignment']['scope']),
        audit=dict(audit, all_value_and_action_ranges_intersect=True, conflicting_certificates=0,
            state_caps_respected=True, cpu_is_atomic_operation_threshold=True),
        gap_scope=baselines['impact']['gap_scope'], g2_passed=False,
        new_method_contribution_established=False, ordinary_method_intrinsic_barrier_established=False,
        whole_journey_optimality_claim=False, server_transfer_bytes=0)
    if priority:
        result['ordinary_comparison'] = dict(aggregate=comparisons['ordinary'], records='ordinary_paired.csv',
            scope='Strong impact scheduling versus the retained ordinary V55 arm; each complete query has its own original budget.')
        result = nested_sources(result)
        choices = nested_sources(choices)
        write_csv(directory/'ordinary_paired.csv', [r for r in pairs if r['baseline'] == 'ordinary'])
    save(directory/'summary.json', result)
    # This is a derived appendix, so regenerating a report replaces it.
    # Raw paid receipts continue to use the exclusive writer in the runner.
    with gzip.open(directory/'matched_action_attribution.json.gz', 'wt') as handle:
        json.dump(matched['records'], handle, default=jsonable, allow_nan=False)
    for name, rows in (('paired', flat), ('same_budget_paired', pairs), ('gaps', gaps),
        ('observed_action_gaps', choices), ('joint_attribution', joints), ('budget_paired', budget_pairs),
        ('matched_action_attribution', matched['flat'])):
        write_csv(directory/f'{name}.csv', rows)
    plot(pairs, modes, budgets, directory, 'V55 priority' if priority else 'V55')
    print(dict(queries=len(records), calls=len(flat), audit=result['audit'],
        optimal={b['name']: {k: aggregates[b['name']]['optimal'][k]
            for k in ('mean_width', 'mean_cpu', 'certified_actions', 'complete')} for b in budgets}), flush=True)
    return result


def plot(rows, modes, budgets, directory=None, label='V55'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = HERE if directory is None else directory
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for column, baseline in enumerate(('impact', 'compressed')):
        for mode, color in zip(modes, colors):
            cells = [r for r in rows if r['baseline'] == baseline and r['mode'] == mode]
            for row, metric in enumerate(('width', 'cpu')):
                axes[row, column].scatter([r['baseline_'+metric] for r in cells],
                    [r['v55_'+metric] for r in cells], s=13, alpha=.45, color=color, label=mode)
        for row, metric in enumerate(('width', 'cpu')):
            ax = axes[row, column]; cells = [r for r in rows if r['baseline'] == baseline]
            top = max(max(r['baseline_'+metric], r['v55_'+metric]) for r in cells)*1.03
            ax.plot([0, top], [0, top], '--', color='gray', linewidth=1)
            ax.set(xlabel=f'V53 {baseline} {metric} (s)', ylabel=f'{label} {metric} (s)',
                   title='All 880 paired queries; b2 and b10'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(directory/'paired_quality_time.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, budget in zip(axes, budgets):
        for mode, color in zip(modes, colors):
            groups = [[r for r in rows if (r['budget'], r['mode'], r['baseline']) ==
                       (budget['name'], mode, baseline)] for baseline in ('compressed', 'impact')]
            x = [mean(g, lambda r: r['baseline_cpu']) for g in groups]+[mean(groups[0], lambda r: r['v55_cpu'])]
            y = [mean(g, lambda r: r['baseline_width']) for g in groups]+[mean(groups[0], lambda r: r['v55_width'])]
            ax.plot(x, y, color=color, linewidth=1, label=mode)
            for i, marker in enumerate(('x', '+', 'o')):
                ax.scatter(x[i], y[i], color=color, marker=marker, s=45)
        ax.set(xlabel='Mean total process CPU per query (s)', ylabel='Mean conditional interval width (s)')
        ax.set_title(f"{budget['name']}: all 88 states\ncompressed x, impact +, {label} o", fontsize=11)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(directory/'quality_time.png', dpi=170); plt.close(fig)


if __name__ == '__main__':
    summarize()
