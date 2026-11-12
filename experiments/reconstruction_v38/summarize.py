"""Paired speed and physical quality, including every positive and negative root."""

from collections import Counter
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reconstruction_v34.discover import read
from experiments.reconstruction_v38.run import VARIANTS, HERE


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(count=len(values), total=float(values.sum()), mean=float(values.mean()),
        median=float(np.median(values)), minimum=float(values.min()), maximum=float(values.max()))


def paired_cost(subject, baseline):
    subject, baseline = np.asarray(subject), np.asarray(baseline)
    return dict(total_reduction_fraction=float(1-subject.sum()/baseline.sum()),
        subject_over_baseline_ratio=stats(subject/baseline),
        baseline_minus_subject_seconds=stats(baseline-subject),
        faster=int((subject < baseline).sum()), slower=int((subject > baseline).sum()))


def main():
    protocol, manifest = read(HERE/'protocol.json'), read(HERE/'seed_manifest.json')
    bank_rows, bank_costs, bank_stats = [], {v: [] for v in VARIANTS}, {v: Counter() for v in VARIANTS}
    bank_wall = {v: [] for v in VARIANTS}
    for entry in manifest:
        name = entry['source'].replace('/', '_').removesuffix('.json.gz')
        rows = {v: read(HERE/'banks'/f'{name}_{v}.json.gz') for v in VARIANTS}
        for variant, row in rows.items():
            bank_costs[variant].append(row['complete_cost']['process_cpu'])
            bank_wall[variant].append(row['complete_cost']['wall'])
            bank_stats[variant].update(row['cost']['planner_stats_total'])
        bank_rows.append(dict(source=entry['source'], seed=entry['seed'],
            all_exact=all(r['exact_v37'] for r in rows.values()),
            costs={v:r['complete_cost'] for v,r in rows.items()},
            complete_pairs=rows['fast']['paired']['complete_pairs'],
            unresolved_pairs=rows['fast']['paired']['unresolved_pairs'],
            selected_action=rows['fast']['selected_action'],
            full_bank_mean_D_interval=rows['fast']['paired']['full_bank_mean_gain_interval']))
    methods = [*VARIANTS, *protocol['cheap_methods']]
    roots = {name: [read(HERE/'roots'/f'{i:03}_{name}.json.gz') for i in range(72)] for name in methods}
    controls = {name: dict(loss_seconds=stats([r['outcome']['loss'] for r in rows]),
        process_cpu_seconds=stats([r['cost']['total_process_cpu'] for r in rows]),
        wall_seconds=stats([r['cost']['total_wall'] for r in rows]),
        statuses=dict(Counter(r['outcome']['status'] for r in rows)),
        all_exact_original=all(r['exact_original'] for r in rows)) for name, rows in roots.items()}
    comparisons = {}
    for name in methods:
        gains = [b['outcome']['loss']-a['outcome']['loss'] for a,b in zip(roots['fast'], roots[name])]
        comparisons[name] = dict(baseline_loss_minus_fast=stats(gains),
            wins=sum(g > 0 for g in gains), harms=sum(g < 0 for g in gains), ties=sum(g == 0 for g in gains),
            cost=paired_cost([r['cost']['total_process_cpu'] for r in roots['fast']],
                             [r['cost']['total_process_cpu'] for r in roots[name]]))
    root_rows = []
    for index in range(72):
        row = roots['fast'][index]
        root_rows.append(dict(root_index=index, root_id=row['root_id'], day=row['day'],
            losses={name: roots[name][index]['outcome']['loss'] for name in methods},
            process_cpu={name: roots[name][index]['cost']['total_process_cpu'] for name in methods},
            gain_vs_combined=roots['combined_rolling'][index]['outcome']['loss']-row['outcome']['loss'],
            gates=len(row['gates']), switches=sum(g['selected_action'] != g['actions']['base'] for g in row['gates'])))
    daily = {day: dict(roots=sum(r['day'] == day for r in root_rows),
        gain_seconds=stats([r['gain_vs_combined'] for r in root_rows if r['day'] == day]),
        fast_cpu_seconds=stats([r['process_cpu']['fast'] for r in root_rows if r['day'] == day]))
        for day in sorted({r['day'] for r in root_rows})}
    result = dict(stage='V38', development_only=True,
        reference=dict(states=len(manifest), variants=3, measurements=len(bank_rows)*3,
            unique_existing_worlds=sum(e['count'] for e in manifest), new_scientific_worlds=0,
            world_executions=sum(e['count'] for e in manifest)*3,
            action_continuations=sum(e['count'] for e in manifest)*6,
            all_exact=all(r['all_exact'] for r in bank_rows),
            unresolved_pairs_per_variant=sum(r['unresolved_pairs'] for r in bank_rows),
            cpu={v:stats(bank_costs[v]) for v in VARIANTS}, wall={v:stats(bank_wall[v]) for v in VARIANTS},
            fast_vs_legacy=paired_cost(bank_costs['fast'], bank_costs['legacy']),
            fast_vs_shared=paired_cost(bank_costs['fast'], bank_costs['shared']),
            shared_vs_legacy=paired_cost(bank_costs['shared'], bank_costs['legacy']),
            planner_statistics=bank_stats),
        online=dict(roots=72, methods=len(methods), replays=72*len(methods),
            rollout_calls_per_variant={v:sum(len(r['gates']) for r in roots[v]) for v in VARIANTS},
            original_streams_reexecuted=True, saved_answers_reused=0,
            all_banks_exact=all(r['all_banks_exact_v37'] for v in VARIANTS for r in roots[v]),
            all_trajectories_exact=all(c['all_exact_original'] for c in controls.values()),
            controls=controls, comparisons=comparisons, by_day=daily),
        all_nonzero_cases=[r for r in root_rows if r['gain_vs_combined'] != 0],
        root_pairs=root_rows, state_pairs=bank_rows,
        timing='Current single-process, single-BLAS-thread measurements; all online computation executed and charged. '
               'State/seed manifest loading is charged; common archives/graph preparation and output writing are separate. '
               'The old V37 cached-run charged timing is not the speedup denominator.',
        limitations=['Same six development dates, fixed pruned menus and original 128-world streams; implementation repeats are not new scientific samples.',
            'Full-bank empirical missing-outcome bounds are not population confidence or regret bounds.',
            'Logical replay clocks are unchanged; CPU savings do not establish delayed real-time boarding feasibility.',
            'This preserves standard rollout and supplies exact algebraic implementation savings; it does not alone establish an original OR algorithm.'])
    (HERE/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    labels = dict(legacy='Full rollout', shared='Shared-cache rollout', fast='Accelerated rollout',
        nominal='Nominal', combined_rolling='Combined', single_rolling='Single target',
        pair_rolling='Two target', pair_rollout='Two target + 1 step', pair_stopping='Two target + stopping')
    for index, name in enumerate(methods):
        row = controls[name]
        axes[0].scatter(row['process_cpu_seconds']['mean'], row['loss_seconds']['mean'],
            s=100 if name in VARIANTS else 50, marker='*' if name == 'fast' else 'o',
            label=labels[name], color=plt.get_cmap('tab10')(index))
    axes[0].set(xscale='log', xlabel='Mean full online CPU (s, log scale)', ylabel='Mean truncated travel loss (s)')
    axes[0].legend(fontsize=7)
    high = max(max(bank_costs['shared']), max(bank_costs['fast']))
    axes[1].plot([0, high], [0, high], '--', color='gray', lw=1)
    axes[1].scatter(bank_costs['shared'], bank_costs['fast'], s=14, alpha=.65)
    axes[1].set(xlabel='Shared-cache bank CPU (s)', ylabel='Accelerated bank CPU (s)',
        title='All 148 frozen states; identical world outcomes')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(HERE/'quality_cpu.png', dpi=170)
    plt.close(fig)
    print(json.dumps(dict(controls=controls,
        fast_vs_legacy=result['online']['comparisons']['legacy'],
        fast_vs_shared=result['online']['comparisons']['shared'],
        fast_vs_combined=result['online']['comparisons']['combined_rolling']), indent=2))


if __name__ == '__main__':
    main()
