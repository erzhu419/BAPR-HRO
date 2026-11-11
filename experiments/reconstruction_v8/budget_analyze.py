"""Summarize route quality, computation and search coverage on paired cases."""

import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def read(path):
    return json.loads(path.read_text())


def compare_logical(rows):
    fields = ('clock_seconds', 'stop', 'current_trip', 'action', 'outcome', 'end_seconds',
              'model_transitions', 'root_actions')
    archives = {
        'uct64': ('reconstruction_v5/city_budget64_logical_results.json', 'evolving_uct'),
        'uct128': ('reconstruction_v5/city_budget128_logical_results.json', 'evolving_uct'),
        'uct512': ('reconstruction_v5/city_frontier_logical_results.json', 'evolving_uct'),
        'compiled_nominal': ('reconstruction_v6/city_itinerary_logical.json', 'compiled_nominal'),
        'analytic_recourse': ('reconstruction_v6/city_itinerary_logical.json', 'analytic_recourse'),
    }
    matches = []
    for method, (path, old_method) in archives.items():
        old = {(r['date'], r['query']): r for r in read(EXPERIMENTS / path)['rows']
               if r['method'] == old_method}
        for row in rows:
            if row['method'] != method or row['clock'] != 'logical':
                continue
            reference = old[row['date'], row['query']]
            trace = lambda r: [tuple(d[key] for key in fields) for d in r['decisions']]
            matched = all(row[key] == reference[key] for key in ('status', 'loss_seconds', 'events'))
            matches.append(dict(method=method, date=row['date'], query=row['query'], archive=path,
                matched=matched and trace(row) == trace(reference)))
    return matches


def summarize(rows, values):
    result = {}
    for clock, methods in values.items():
        points = {}
        for method, old in methods.items():
            group = [r for r in rows if r['clock'] == clock and r['method'] == method]
            decisions = [d for row in group for d in row['decisions']]
            root = [row['decisions'][0] for row in group]
            days = sorted({r['date'] for r in group})
            points[method] = dict(date_queries=len(group),
                losses=[r['loss_seconds'] for r in group],
                day_loss_ranges={day: [
                    mean(r['elapsed_observed_seconds'] for r in group if r['date'] == day),
                    mean(r['loss_seconds'] if r['loss_seconds'] is not None else
                         r['od']['deadline_seconds'] - r['od']['ready_seconds']
                         for r in group if r['date'] == day)] for day in days},
                status=old['statuses'], loss_lower_seconds=old['mean_loss_lower_seconds'],
                loss_upper_seconds=old['mean_loss_upper_seconds'],
                mean_wall_seconds=old['mean_end_to_end_ms'] / 1000,
                mean_cpu_seconds=mean(r['cpu_seconds'] for r in group),
                mean_graph_seconds=old['mean_graph_build_ms'] / 1000,
                mean_setup_seconds=old['mean_policy_setup_ms'] / 1000,
                mean_charged_seconds=mean(sum(d['computation_clock_ticks'] for d in r['decisions']) for r in group),
                mean_transitions=mean(r['planning_transitions'] for r in group),
                p50_p95_p99_decision_seconds=[v / 1000 for v in old['decision_p50_p95_p99_ms']],
                first_decisions=[dict(date=r['date'], query=r['query'],
                    root_actions=d.get('root_actions'), scored_root_actions=d.get('scored_root_actions'),
                    wall_seconds=d['decision_total_ms'] / 1000,
                    full_model_gap=d.get('full_model_gap')) for r, d in zip(group, root)],
                mean_index_candidates=(mean(d['nominal_index']['candidate_edges'] for d in decisions)
                                       if method.startswith('uct') else None))
        result[clock] = points
    return result


def plot(panels):
    labels = {'compiled_nominal': 'Compiled nominal', 'analytic_recourse': 'Analytic recourse',
              'uct64': 'UCT 64', 'uct128': 'UCT 128', 'uct512': 'UCT 512'}
    colors = {'compiled_nominal': '#6B7280', 'analytic_recourse': '#127D75', 'uct64': '#2F5597',
              'uct128': '#2F5597', 'uct512': '#2F5597'}
    y_offsets = {'compiled_nominal': 7, 'analytic_recourse': 7, 'uct64': -14,
                 'uct128': 7, 'uct512': 7}
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, layout='constrained')
    for ax, (clock, methods) in zip(axes, panels.items()):
        curve = [methods[m] for m in ('uct64', 'uct128', 'uct512')]
        ax.plot([m['mean_wall_seconds'] for m in curve], [m['loss_upper_seconds'] for m in curve],
                '-', color=colors['uct64'], alpha=.45, linewidth=1)
        for method, point in methods.items():
            x = point['mean_wall_seconds']
            lo, hi = point['loss_lower_seconds'], point['loss_upper_seconds']
            ax.plot(x, hi, 'o', color=colors[method])
            if lo != hi:
                ax.vlines(x, lo, hi, color=colors[method], lw=2)
            ax.annotate(labels[method], (x, hi), xytext=(3, y_offsets[method]),
                        textcoords='offset points', fontsize=8)
        ax.set(xscale='log', xlabel='Mean end-to-end wall time (s)', title=f'{clock.capitalize()} clock')
        ax.grid(alpha=.18)
        ax.margins(x=.18, y=.20)
    axes[0].set_ylabel('Mean capped journey loss (s)')
    for suffix in ('png', 'pdf'):
        figure.savefig(HERE / f'budget_quality_time.{suffix}', dpi=160)


def main():
    archive = read(HERE / 'budget_results.json')
    rows = archive['rows']
    panels = summarize(rows, archive['summaries'])
    matches = compare_logical(rows)
    if not all(m['matched'] for m in matches):
        raise AssertionError([m for m in matches if not m['matched']])
    out = dict(panels=panels, archived_logical_parity=matches,
               pairing='Two known service dates × two fixed OD; no new confirmation.',
               pipeline='Index charged as part of UCT; measured on common CPU affinity and thread settings.')
    (HERE / 'budget_summary.json').write_text(json.dumps(out, indent=2) + '\n')
    plot(panels)


if __name__ == '__main__':
    main()
