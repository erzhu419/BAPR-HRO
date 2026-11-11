"""Aggregate the common-publisher policy, budget and scheduling ablations."""

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent


def summarize(rows):
    by_method = defaultdict(list)
    for row in rows:
        by_method[row['method']].append(row)
    result = {}
    for method, group in by_method.items():
        assert len(group) == 4
        decisions = [d for row in group for d in row['decisions']]
        gates = [d for d in decisions if d['gate'] is not None]
        result[method] = dict(
            cases=len(group), status=dict(Counter(row['status'] for row in group)),
            mean_loss_lower=mean(row['loss_lower'] for row in group),
            mean_loss_upper=mean(row['loss_upper'] for row in group),
            mean_pipeline_wall_seconds=mean(row['actual_seconds'] + row['graph_seconds'] for row in group),
            mean_pipeline_cpu_seconds=mean(row['actual_cpu_seconds'] for row in group),
            mean_worker_seconds=mean(row['worker_seconds'] for row in group),
            mean_setup_seconds=mean(row['setup_seconds'] for row in group),
            search_calls=sum(d['search_seconds'] > 0 for d in decisions),
            search_transitions=sum(d['transitions'] for d in decisions),
            search_wall_seconds=sum(d['search_seconds'] for d in decisions),
            gate_wall_seconds=sum(d.get('gate_seconds', 0.) for d in decisions),
            candidate_wall_seconds=sum(d['candidate_seconds'] for d in decisions),
            gate_checks=dict(Counter(d['gate']['kind'] for d in gates)),
            gate_accepts=dict(Counter(d['gate']['kind'] for d in gates if d['gate']['accepted'])),
            gate_retained_publications=dict(Counter(d['gate']['kind'] for d in gates
                if d['gate']['accepted'] and d['publication'] == 'retained')),
            publication=dict(Counter(d['publication'] for d in decisions)),
            scheduling_skips=sum(d.get('scheduled_skip', False) for d in decisions),
            attempts_during_worker=sum(a['while_computing'] for row in group for a in row['actions']),
        )
    return result


def pair(rows, method, reference):
    by_case = {(r['date'], r['query'], r['method']): r for r in rows}
    result = []
    for date, query, arm in sorted(by_case):
        if arm != method:
            continue
        a, b = by_case[date, query, arm], by_case[date, query, reference]
        result.append(dict(date=date, query=query, loss_difference_lower=a['loss_lower']-b['loss_upper'],
                           loss_difference_upper=a['loss_upper']-b['loss_lower'],
                           worker_wall_difference=a['worker_seconds']-b['worker_seconds'],
                           pipeline_wall_difference=a['actual_seconds']-b['actual_seconds'],
                           same_realized_attempts=[(r['connection'], r['caught'], r['next_clock']) for r in a['actions']]
                           == [(r['connection'], r['caught'], r['next_clock']) for r in b['actions']]))
    return result


def plot(summary):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    labels = [('off', (64, 128, 512), 'Gate off', 'o'), ('terminal', (64, 128), 'Terminal gate', 's'),
              ('middle', (64, 128), 'Middle gate', '^')]
    for prefix, budgets, name, marker in labels:
        rows = [summary['charged'][f'{prefix}{n}'] for n in budgets]
        xs = [r['mean_pipeline_wall_seconds'] for r in rows]
        ys = [r['mean_loss_upper'] for r in rows]
        ax.plot(xs, ys, marker=marker, label=name)
        for n, x, y in zip(budgets, xs, ys):
            if prefix == 'off':
                ax.annotate(str(n), (x, y), xytext=(0, 8), ha='center', textcoords='offset points', fontsize=8)
    for method, name in [('compiled_nominal', 'Compiled nominal'), ('analytic', 'Analytic recourse')]:
        r = summary['charged'][method]
        ax.scatter(r['mean_pipeline_wall_seconds'], r['mean_loss_upper'], marker='x', label=name)
    for method, r in summary['charged'].items():
        if r['mean_loss_lower'] < r['mean_loss_upper']:
            ax.vlines(r['mean_pipeline_wall_seconds'], r['mean_loss_lower'], r['mean_loss_upper'], color='grey')
    ax.set(xscale='log', xlabel='Mean wall time per journey (s)', ylabel='Mean capped travel loss (s)')
    ax.set_xticks([5, 10, 20, 40, 80])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend(fontsize=8, frameon=False)
    pairs = [('charged', 'off64', 'Every decision'), ('scheduled', 'scheduled_off64', 'Due commitments skipped')]
    for i, (mode, method, name) in enumerate(pairs):
        prefix = method.removesuffix('off64')
        vals = [summary[mode][prefix + m]['mean_worker_seconds'] for m in ('off64', 'terminal64', 'middle64')]
        bx.bar([j+(i-.5)*.32 for j in range(3)], vals, width=.32, label=name)
    bx.set(ylabel='Mean worker wall time (s)', xticks=[0, 1, 2], xticklabels=['Off', 'Terminal', 'Middle'])
    bx.set_ylim(0, 1.22 * max(summary['charged'][m]['mean_worker_seconds']
                             for m in ('off64', 'terminal64', 'middle64')))
    bx.legend(fontsize=8, frameon=False, loc='upper left')
    fig.savefig(HERE / 'quality_time.png', dpi=190)
    plt.close(fig)


def main():
    rows = {mode: json.loads((HERE / f'issued_{mode}_results.json').read_text())['rows']
            for mode in ('charged', 'logical', 'scheduled')}
    for mode in ('charged', 'logical'):
        replacement = json.loads((HERE / f'issued_{mode}_middle_results.json').read_text())['rows']
        updated = {(r['date'], r['query'], r['method']): r for r in replacement}
        expected = {(r['date'], r['query'], r['method']) for r in rows[mode] if r['method'].startswith('middle')}
        if set(updated) != expected:
            raise ValueError(f'Incomplete middle-gate rerun for {mode}')
        rows[mode] = [updated.get((r['date'], r['query'], r['method']), r) for r in rows[mode]]
    results = {mode: summarize(group) for mode, group in rows.items()}
    comparisons = {mode: {method: pair(group, method, base) for method, base in comparison}
                   for (mode, group), comparison in zip(rows.items(), (
                       [('terminal64', 'off64'), ('middle64', 'terminal64'),
                        ('terminal128', 'off128'), ('middle128', 'terminal128')],
                       [('terminal64', 'off64'), ('middle64', 'terminal64')],
                       [('scheduled_terminal64', 'scheduled_off64'), ('scheduled_middle64', 'scheduled_terminal64')],
                   ))}
    comparisons['scheduling_vs_charged'] = {'scheduled_off64': pair(
        rows['charged'] + rows['scheduled'], 'scheduled_off64', 'off64')}
    (HERE / 'issued_summary.json').write_text(json.dumps(dict(summary=results, paired=comparisons), indent=2) + '\n')
    plot(results)


if __name__ == '__main__':
    main()
