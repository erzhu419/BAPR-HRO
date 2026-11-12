"""Compare the added standard rollout baseline with every fixed public control."""

from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.reconstruction_v34.summarize import (
    HERE, PUBLIC, action, arm_summary, first_divergence, mean, numeric, paired, read, write,
)


NAME = 'standard_conditional_rollout'


def control_row(root):
    """Match the original model-construction-through-execution cost scope."""
    cost = root['cost']
    if 'control_total_process_cpu' not in cost:
        if root['root_index'] != 28:
            raise ValueError('Only the preserved root28 pilot lacks a separate control timer')
        return dict(outcome=root['outcome'], cost=dict(total_process_cpu=cost['total_process_cpu'],
            total_wall=cost['total_wall']), cost_scope='Conservative upper bound including common data preparation')
    return dict(outcome=root['outcome'], cost=dict(
        total_process_cpu=cost['control_total_process_cpu'], total_wall=cost['control_total_wall']),
        cost_scope='Matched model-construction-through-execution timer')


def gate_summary(root):
    rows = []
    for gate in root['gates']:
        nominal, combined = gate['actions']['nominal'], gate['actions']['combined']
        bound = gate['paired']['full_bank_mean_gain_interval']
        expected = nominal if bound[1] < 0 else combined
        index = gate['gate_call_index']
        trigger = gate['trigger']
        actual = next((event for event in root['outcome']['trace']
            if event['kind'] == trigger['kind'] and event['clock'] == trigger['clock']
            and event.get('cid', event.get('stop')) == trigger.get('cid', trigger.get('stop'))), None)
        rows.append(dict(root_index=root['root_index'], root_id=root['root_id'], day=root['day'],
            gate_call_index=index, trigger=trigger, seed=gate['seed'], worlds=gate['count'],
            nominal_action=nominal, combined_action=combined, selected_action=gate['selected_action'],
            alternative_selected=gate['selected_action'] == nominal,
            gate_rule_consistent=gate['selected_action'] == expected,
            unchanged_public_history_at_gate=actual is not None and actual['public_history'] == gate['public_history'],
            empirical_nominal_minus_combined_mean_interval=bound,
            complete_world_pairs=gate['paired']['complete_pairs'],
            unresolved_world_pairs=gate['paired']['unresolved_pairs'],
            complete_pair_statistics=gate['paired']['complete_pair_statistics'],
            action_status_counts={name: dict(Counter(row['status'] for row in action_rows))
                for name, action_rows in gate['rows'].items()},
            retained_action_world_counts={name: len(action_rows) for name, action_rows in gate['rows'].items()},
            total_process_cpu=gate['cost']['total_process_cpu'], total_wall=gate['cost']['total_wall'],
            source=dict(file=f'gated_roots/{root["root_index"]:03}.json.gz',
                        json_location=f'gates[{index}]')))
    return rows


def comparisons(gated, original, methods):
    return {method: paired(dict(indices=[r['root_index'] for r in gated],
        rows=[control_row(r) for r in gated]), [r['rows'][PUBLIC][method] for r in original])
        for method in methods}


def public_prefixes(gated, original, methods):
    records = []
    for root, old in zip(gated, original):
        rows = {}
        for method in methods:
            comparison = first_divergence(root['outcome'], old['rows'][PUBLIC][method]['outcome'])
            first = comparison['first']
            if first is not None:
                first['subject_source'] = dict(file=f'gated_roots/{root["root_index"]:03}.json.gz',
                    json_location=f'outcome.trace[{first["trace_index"]}]')
                first['baseline_source'] = dict(file=f'roots/{root["root_index"]:03}.json.gz',
                    json_location=f'rows.{PUBLIC}.{method}.outcome.trace[{first["trace_index"]}]')
            rows[method] = comparison
        records.append(dict(root_index=root['root_index'], root_id=root['root_id'], day=root['day'],
            actual_public_actions=[action(row) for row in root['outcome']['trace']],
            full_history_receipts=f'gated_roots/{root["root_index"]:03}.json.gz', comparisons=rows))
    return records


def draw(gated, original, methods):
    names = methods+[NAME]
    colors = plt.get_cmap('tab10').colors
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for index, method in enumerate(names):
        rows = [control_row(r) for r in gated] if method == NAME else [r['rows'][PUBLIC][method] for r in original]
        rows = [r for r in rows if r['outcome']['loss'] is not None]
        x = [r['cost']['total_process_cpu'] for r in rows]
        y = [r['outcome']['loss'] for r in rows]
        marker, size = ('*', 55) if method == NAME else ('o', 17)
        axes[0].scatter(x, y, marker=marker, s=size, color=colors[index], alpha=.55, label=method)
        if rows:
            axes[1].scatter([mean(x)], [mean(y)], marker=marker, s=180 if method == NAME else 80,
                color=colors[index], edgecolor='black', linewidth=.5)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('Complete cold control CPU (seconds, log scale)')
        ax.set_ylabel('Truncated journey loss (seconds)')
        ax.grid(alpha=.2)
    axes[0].set_title(f'Public messages: all {len(gated)} fixed roots')
    axes[1].set_title('Fixed-policy means across the same roots')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .995), ncol=3, frameon=False)
    figure.text(.5, .012, 'Standard conditional rollout is a strengthened baseline. All model, posterior, '
        'sampling and continuation costs are charged; root28 includes common preparation as a CPU upper bound.',
        ha='center', fontsize=8.6)
    figure.tight_layout(rect=(0, .065, 1, .82))
    figure.savefig(HERE/'quality_cpu_gated.png', dpi=180)
    plt.close(figure)


def main():
    selection, protocol = read(HERE/'selection.json'), read(HERE/'rollout_protocol.json')
    methods = list(read(HERE/'protocol.json')['methods'])
    original, gated, missing = [], [], []
    for index, selected in enumerate(selection['selected']):
        path = HERE/'gated_roots'/f'{index:03}.json.gz'
        if not path.exists():
            missing.append(index)
            continue
        row, prior = read(path), read(HERE/'roots'/f'{index:03}.json.gz')
        if row['root_index'] != index or row['root_id'] != selected['root_id'] or row['root_id'] != prior['root_id']:
            raise ValueError(f'Rollout root differs from frozen selection or original receipt: {path}')
        gated.append(row)
        original.append(prior)
    if missing:
        raise SystemExit(f'Awaiting {len(missing)}/{len(selection["selected"])} rollout roots: {missing}')
    gates = [row for root in gated for row in gate_summary(root)]
    prefixes = public_prefixes(gated, original, methods)
    exact_indices = {r['root_index'] for r in gated if 'control_total_process_cpu' in r['cost']}
    exact_gated = [r for r in gated if r['root_index'] in exact_indices]
    exact_original = [r for r in original if r['root_index'] in exact_indices]
    by_day = {}
    for day in read(HERE/'protocol.json')['main_dates']:
        day_gated = [r for r in gated if r['day'] == day]
        day_original = [r for r in original if r['day'] == day]
        day_gates = [g for g in gates if g['day'] == day]
        by_day[day] = dict(denominator=len(day_gated), control=arm_summary([control_row(r) for r in day_gated]),
            comparisons=comparisons(day_gated, day_original, methods), gate_calls=len(day_gates),
            changed_actions=sum(g['alternative_selected'] for g in day_gates),
            unresolved_world_pairs=sum(g['unresolved_world_pairs'] for g in day_gates))
    output = dict(method=NAME, attribution='Standard disagreement-triggered executable-continuation '
            'rollout, a strengthened baseline; not a claim of a new OR algorithm.',
        protocol='rollout_protocol.json', development_only=True, denominator=len(selection['selected']),
        control=arm_summary([control_row(r) for r in gated]),
        fixed_public_controls={method: arm_summary([r['rows'][PUBLIC][method] for r in original]) for method in methods},
        comparisons=comparisons(gated, original, methods), by_day=by_day,
        sign_convention='Actual comparison gain = fixed baseline loss minus rollout loss; positive is beneficial. '
            'Conditional gate intervals instead use nominal-action minus combined-action loss; negative favors nominal.',
        gate_accounting=dict(calls=len(gates), roots_triggered=sum(bool(r['gates']) for r in gated),
            changed_actions=sum(g['alternative_selected'] for g in gates),
            roots_with_changed_action=len({g['root_index'] for g in gates if g['alternative_selected']}),
            worlds_per_trigger=protocol['worlds_per_trigger'],
            shared_conditional_worlds=sum(g['worlds'] for g in gates),
            action_continuations=sum(sum(g['retained_action_world_counts'].values()) for g in gates),
            complete_world_pairs=sum(g['complete_world_pairs'] for g in gates),
            unresolved_world_pairs=sum(g['unresolved_world_pairs'] for g in gates),
            gate_cpu_seconds=numeric([g['total_process_cpu'] for g in gates]),
            all_rules_consistent=all(g['gate_rule_consistent'] for g in gates),
            all_public_gate_histories_unchanged=all(g['unchanged_public_history_at_gate'] for g in gates),
            action_world_counts_match_budget=all(all(n == protocol['worlds_per_trigger']
                for n in g['retained_action_world_counts'].values()) for g in gates)),
        full_pipeline_total_process_cpu_seconds=numeric([r['cost']['total_process_cpu'] for r in gated]),
        full_pipeline_total_wall_seconds=numeric([r['cost']['total_wall'] for r in gated]),
        control_timing_scope_by_root=[dict(root_index=r['root_index'], scope=control_row(r)['cost_scope']) for r in gated],
        exact_same_scope_timing_sensitivity=dict(denominator=len(exact_gated),
            excluded_from_this_timing_sensitivity=[r['root_index'] for r in gated if r['root_index'] not in exact_indices],
            control=arm_summary([control_row(r) for r in exact_gated]),
            comparisons=comparisons(exact_gated, exact_original, methods)),
        timing_scope='The matched control CPU comparison starts before model construction and includes '
            'menu/message setup, both cheap scores, all conditional sampling and simulated continuations, '
            'actual decisions, posterior updates and trace. Full-pipeline totals additionally include common '
            'data loading and graph preparation. The preserved root28 pilot has only the full-pipeline timer, '
            'used as a conservative control-cost upper bound; the other 71 roots have the exact matching scope. '
            'A separate 71-root timing sensitivity is provided. Gate costs are already included in root costs, not added twice.',
        gates=gates, actual_public_prefixes=prefixes,
        limitations=['The finite-bank missing-outcome interval is not a population confidence or regret certificate.',
            'All selected roots and all conditional worlds remain represented, including unresolved outcomes.',
            'CPU is charged in logical-clock replay; computation does not delay a physical decision.',
            'The original six controls and preceding four-state 128-world reference results remain unchanged.'])
    write(HERE/'gated_summary.json', output)
    draw(gated, original, methods)
    print(dict(roots=len(gated), gate_calls=len(gates), changed_actions=output['gate_accounting']['changed_actions'],
               unresolved_world_pairs=output['gate_accounting']['unresolved_world_pairs']))


if __name__ == '__main__':
    main()
