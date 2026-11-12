"""Compare each fixed control and retain the complete positive/negative panel."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.reconstruction_v34.summarize import (
    PUBLIC, LOCAL, arm_summary, first_divergence, paired, read, write,
)

HERE = Path(__file__).resolve().parent
GATED = 'standard_conditional_rollout'


def observations_only(row):
    """Remove only declared model metadata, retaining every raw public fact."""
    outcome = dict(row['outcome'])
    outcome['trace'] = [dict(event, public_history=[
        {k: v for k, v in record.items() if k not in (
            'emission_bias_seconds', 'emission_sd_seconds', 'emission_selected')}
        for record in event.get('public_history', [])]) for event in outcome['trace']]
    return dict(row, outcome=outcome)


def compare(subject, baseline, indices, *, cross_model=False):
    if cross_model:
        subject, baseline = ([observations_only(r) for r in rows] for rows in (subject, baseline))
    return paired(dict(indices=indices, rows=subject), baseline)


def main():
    protocol = read(HERE/'protocol.json')
    selected = read(HERE.parent/'reconstruction_v34/selection.json')['selected']
    methods = list(protocol['paired_replay']['methods'])
    roots, previous, own = [], [], []
    for index, root in enumerate(selected):
        new = read(HERE/'roots'/f'{index:03}.json.gz')
        old = read(HERE.parent/'reconstruction_v35/roots'/f'{index:03}.json.gz')
        local = read(HERE.parent/'reconstruction_v34/roots'/f'{index:03}.json.gz')
        if new['root_index'] != index or any(r['root_id'] != root['root_id'] for r in (new, old, local)):
            raise ValueError(f'Root {index} differs from the fixed cohort')
        roots.append(new)
        previous.append(old)
        own.append(local)
    indices = list(range(len(roots)))
    current = {m: [r['rows'][PUBLIC][m] for r in roots] for m in methods}
    all_messages = {m: [r['rows'][PUBLIC][m] for r in previous] for m in methods}
    model_pairs = {m: compare(current[m], all_messages[m], indices, cross_model=True) for m in methods}
    control_pairs = {m: compare(current[GATED], current[m], indices) for m in methods if m != GATED}
    own_pairs = {m: compare(current[m], [r['rows'][LOCAL][m] for r in own], indices)
                 for m in methods if m != GATED}
    fixed_own_nominal = [r['rows'][LOCAL]['rolling_nominal'] for r in own]
    fixed_baseline_pairs = {m: compare(current[m], fixed_own_nominal, indices) for m in methods}
    divergences, gates = [], []
    for index, root in enumerate(roots):
        for method in methods:
            a, b = (observations_only(rows[method][index]) for rows in (current, all_messages))
            difference = first_divergence(a['outcome'], b['outcome'])
            if not difference['actions_identical']:
                divergences.append(dict(root_index=index, day=root['day'], method=method, **difference,
                    pair=model_pairs[method]['pairs'][index], source=f'roots/{index:03}.json.gz',
                    baseline=f'../reconstruction_v35/roots/{index:03}.json.gz'))
        for gate in current[GATED][index]['gates']:
            bound = gate['paired']['full_bank_mean_gain_interval']
            expected = gate['actions']['nominal'] if bound[1] < 0 else gate['actions']['combined']
            gates.append(dict(root_index=index, day=root['day'], call=gate['gate_call_index'],
                seed=gate['seed'], count=gate['count'], trigger=gate['trigger'], actions=gate['actions'],
                selected=gate['selected_action'], paired=gate['paired'],
                switch=gate['selected_action'] == gate['actions']['nominal'],
                rule_consistent=gate['selected_action'] == expected,
                retained_action_counts={a: len(v) for a, v in gate['rows'].items()},
                cpu=gate['cost']['total_process_cpu']))
    daily = {}
    for day in sorted({r['day'] for r in roots}):
        mask = [i for i, r in enumerate(roots) if r['day'] == day]
        daily[day] = dict(denominator=len(mask),
            controls={m: arm_summary([current[m][i] for i in mask]) for m in methods},
            model_change_pairs={m: compare([current[m][i] for i in mask],
                [all_messages[m][i] for i in mask], mask, cross_model=True) for m in methods},
            rollout_pairs={m: compare([current[GATED][i] for i in mask],
                [current[m][i] for i in mask], mask) for m in methods if m != GATED})
    result = dict(denominator=len(roots), development_only=True,
        gain_orientation='baseline loss minus subject loss; positive improves travel',
        controls={m: arm_summary(current[m]) for m in methods},
        all_message_controls={m: arm_summary(all_messages[m]) for m in methods},
        model_change_pairs=model_pairs, same_model_rollout_pairs=control_pairs,
        single_message_vs_own_only=own_pairs, by_day=daily,
        fixed_own_nominal=arm_summary(fixed_own_nominal),
        all_controls_vs_fixed_own_nominal=fixed_baseline_pairs,
        gate_accounting=dict(calls=len(gates), shared_worlds=sum(g['count'] for g in gates),
            action_continuations=sum(sum(g['retained_action_counts'].values()) for g in gates),
            unresolved_pairs=sum(g['paired']['unresolved_pairs'] for g in gates),
            switches=sum(g['switch'] for g in gates), all_rules_consistent=all(g['rule_consistent'] for g in gates)),
        gates=gates, source='protocol.json',
        limitations=['All policy dates were previously developed; old measurements are reused.',
            'Common raw public information remains available; only the model compression rule changes.',
            'Quality uses logical clocks; all current control CPU is charged separately.',
            'A single-message compression gain would not by itself identify conditional error correlation or establish a novel algorithm.'])
    write(HERE/'summary.json', result)
    write(HERE/'divergences.json', dict(rows=divergences))
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for method, color in zip(methods, ['#4863a0', '#c76b28', '#358559']):
        for group, marker, label in ((current, 'o', 'single'), (all_messages, 'x', 'all fitted')):
            s = arm_summary(group[method])
            ax.scatter(s['cold_total_process_cpu_seconds']['mean'], s['loss_seconds']['mean'],
                       c=color, marker=marker, s=85, label=f'{label}: {method}')
    baseline = result['fixed_own_nominal']
    ax.scatter(baseline['cold_total_process_cpu_seconds']['mean'], baseline['loss_seconds']['mean'],
               c='black', marker='D', s=60, label='fixed own-only rolling_nominal')
    ax.set(xlabel='Complete cold control CPU (seconds)', ylabel='Mean journey loss (seconds)',
           title='72 fixed roots: single-message and all-message controls')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE/'quality_cpu.png', dpi=180)
    plt.close(fig)
    print({m: result['controls'][m]['loss_seconds'] for m in methods})
    print(result['gate_accounting'])


if __name__ == '__main__':
    main()
