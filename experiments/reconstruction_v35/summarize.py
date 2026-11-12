"""Keep model changes separate from fixed-controller improvements."""

from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.reconstruction_v34.summarize import (
    PUBLIC, LOCAL, arm_summary, first_divergence, paired, read, write,
)
from experiments.reconstruction_v34.summarize_gated import control_row


HERE = Path(__file__).resolve().parent
V34 = HERE.parent/'reconstruction_v34'
GATED = 'standard_conditional_rollout'


def observations_only(row):
    """Model-emission parameters are not additional received information."""
    result = deepcopy(row)
    for event in result['outcome']['trace']:
        for record in event.get('public_history', []):
            for key in ('emission_bias_seconds', 'emission_sd_seconds'):
                record.pop(key, None)
    return result


def compare(subject, baseline, indices, *, cross_model=False):
    if cross_model:
        subject = [observations_only(r) for r in subject]
        baseline = [observations_only(r) for r in baseline]
    return paired(dict(indices=indices, rows=subject), baseline)


def main():
    protocol = read(HERE/'protocol.json')
    selection = read(V34/'selection.json')['selected']
    methods = list(protocol['paired_replay']['methods'])
    roots, old, old_gated = [], [], []
    for index, selected in enumerate(selection):
        row = read(HERE/'roots'/f'{index:03}.json.gz')
        if row['root_index'] != index or row['root_id'] != selected['root_id']:
            raise ValueError(f'Root {index} differs from frozen cohort')
        roots.append(row)
        old.append(read(V34/'roots'/f'{index:03}.json.gz'))
        old_gated.append(read(V34/'gated_roots'/f'{index:03}.json.gz'))
    indices = list(range(len(roots)))
    current = {m: [r['rows'][PUBLIC][m] for r in roots] for m in methods}
    legacy = {m: ([control_row(r) for r in old_gated] if m == GATED else
                 [r['rows'][PUBLIC][m] for r in old]) for m in methods}
    model_pairs = {m: compare(current[m], legacy[m], indices, cross_model=True) for m in methods}
    control_pairs = {m: compare(current[GATED], current[m], indices) for m in methods if m != GATED}
    information_pairs = {m: compare(current[m], [r['rows'][LOCAL][m] for r in old], indices)
                         for m in methods if m != GATED}
    divergences, gates = [], []
    for index, root in enumerate(roots):
        for name in methods:
            a, b = observations_only(current[name][index]), observations_only(legacy[name][index])
            comparison = first_divergence(a['outcome'], b['outcome'])
            if not comparison['actions_identical']:
                divergences.append(dict(root_index=index, day=root['day'], method=name,
                    **comparison, pair=model_pairs[name]['pairs'][index],
                    source=f'roots/{index:03}.json.gz', baseline_source=(
                        f'../reconstruction_v34/gated_roots/{index:03}.json.gz' if name == GATED else
                        f'../reconstruction_v34/roots/{index:03}.json.gz')))
        for gate in current[GATED][index]['gates']:
            gates.append(dict(root_index=index, day=root['day'], call=gate['gate_call_index'],
                seed=gate['seed'], count=gate['count'], trigger=gate['trigger'],
                actions=gate['actions'], selected=gate['selected_action'], paired=gate['paired'],
                alternative_selected=gate['selected_action'] == gate['actions']['nominal'],
                cost=gate['cost']['total_process_cpu']))
    by_day = {}
    for day in sorted({r['day'] for r in roots}):
        mask = [i for i, r in enumerate(roots) if r['day'] == day]
        by_day[day] = dict(denominator=len(mask),
            controls={m: arm_summary([current[m][i] for i in mask]) for m in methods},
            model_pairs={m: compare([current[m][i] for i in mask], [legacy[m][i] for i in mask], mask, cross_model=True)
                         for m in methods},
            control_pairs={m: compare([current[GATED][i] for i in mask], [current[m][i] for i in mask], mask)
                           for m in methods if m != GATED})
    result = dict(denominator=len(roots), development_only=True,
        validation='validation_summary.json', candidate_model='fitted_emission.json',
        gain_orientation='baseline loss minus subject loss; positive improves travel',
        controls={m: arm_summary(current[m]) for m in methods},
        original_controls={m: arm_summary(legacy[m]) for m in methods},
        model_change_pairs=model_pairs, same_model_rollout_pairs=control_pairs,
        same_control_fitted_messages_vs_own_only=information_pairs, by_day=by_day,
        gate_accounting=dict(calls=len(gates), shared_worlds=sum(g['count'] for g in gates),
            action_continuations=2*sum(g['count'] for g in gates),
            unresolved_pairs=sum(g['paired']['unresolved_pairs'] for g in gates),
            switches=sum(g['alternative_selected'] for g in gates)), gates=gates,
        limitations=['Original measurements are reused rather than rerun; old rollout root28 CPU is an upper bound including preparation.',
            'Same-model policy comparisons use identical information; cross-model comparisons remove only emission-parameter metadata from history equality.',
            'Quality is logical-clock replay and CPU is separately charged; no realtime feasibility claim.',
            'Prediction scores and standard rollout gains do not establish a new algorithm or G2.'])
    write(HERE/'summary.json', result)
    write(HERE/'divergences.json', dict(rows=divergences))
    fig, ax = plt.subplots(figsize=(8, 4.7))
    colors = ['#4863a0', '#c76b28', '#358559']
    for name, color in zip(methods, colors):
        for source, marker, label in ((current, 'o', 'fitted'), (legacy, 'x', 'legacy')):
            summary = arm_summary(source[name])
            ax.scatter(summary['cold_total_process_cpu_seconds']['mean'], summary['loss_seconds']['mean'],
                       c=color, marker=marker, s=85, label=f'{label}: {name}')
    ax.set(xlabel='Complete cold control CPU (seconds)', ylabel='Mean journey loss (seconds)',
           title='Same 72 roots: fixed model and fixed controller comparisons')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE/'quality_cpu.png', dpi=180)
    plt.close(fig)
    print({m: result['controls'][m]['loss_seconds'] for m in methods})
    print(result['gate_accounting'])


if __name__ == '__main__':
    main()
