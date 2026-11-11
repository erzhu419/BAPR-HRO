"""Keep development, independent dates, and censored new OD evidence separate."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ('trained_nominal', 'compiled_nominal', 'evolving_uct', 'analytic_recourse')
LABELS = ('Nominal', 'Compiled nominal', 'UCT 64', 'Analytic recourse')
COLORS = ('#707070', '#d99a28', '#377eb8', '#23936b')


def load(name):
    return json.loads((HERE / f'{name}.json').read_text())


def limits(row):
    if row['loss_seconds'] is not None:
        return row['loss_seconds'], row['loss_seconds']
    return row['elapsed_observed_seconds'], row['od']['deadline_seconds'] - row['od']['ready_seconds']


def main():
    panels = [('Development: July 26–27', 'city_itinerary_charged'),
              ('Independent dates: July 28–29', 'city_confirmation_charged'),
              ('New OD: original source join', 'city_new_od_charged'),
              ('New OD: repaired stop identity', 'city_source_completed_paired_charged')]
    result = {}
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 3.8), sharey=True)
    for number, (ax, (title, filename)) in enumerate(zip(axes, panels)):
        source = load(filename)
        rows = [r for r in source['rows'] if filename != 'city_source_completed_paired_charged' or r['query'] >= 2]
        by_method = {method: [r for r in rows if r['method'] == method] for method in METHODS}
        keys = [{(r['date'], r['query']) for r in by_method[m]} for m in METHODS]
        if not all(key == keys[0] and len(by_method[m]) == 4 for key, m in zip(keys, METHODS)):
            raise ValueError('Methods do not contain the same four date/query cells')
        aggregate = {}
        for method, label, color in zip(METHODS, LABELS, COLORS):
            group = by_method[method]
            low, high = np.mean([limits(r) for r in group], axis=0)
            seconds = np.mean([r['end_to_end_ms'] / 1000 for r in group])
            aggregate[method] = dict(n=4, mean_loss_lower_seconds=low,
                mean_loss_upper_seconds=high, mean_end_to_end_seconds=seconds,
                mean_compute_seconds=float(np.mean([r['compute_ms'] / 1000 for r in group])),
                n_arrived=sum(r['status'] == 'arrived' for r in group),
                n_censored=sum(r['status'] == 'censored' for r in group))
            center = (low + high) / 2
            ax.errorbar(seconds, center, yerr=[[center - low], [high - center]],
                        fmt='o', color=color, capsize=4, label=label, markersize=7)
        paired = defaultdict(list)
        lookup = {(r['date'], r['query'], r['method']): r for r in rows}
        for day, query in sorted(keys[0]):
            al, au = limits(lookup[day, query, 'analytic_recourse'])
            bl, bu = limits(lookup[day, query, 'evolving_uct'])
            paired[day].append((al - bu, au - bl))
        result[filename] = dict(methods=aggregate,
            analytic_minus_uct_by_date={day: np.mean(values, axis=0).tolist() for day, values in paired.items()},
            analytic_minus_uct_mean_bounds=np.mean([pair for values in paired.values() for pair in values], axis=0).tolist())
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Mean total pipeline wall time (s)')
        ax.set_xscale('log')
        ax.set_xticks(((5, 10, 20, 40), (4, 6, 10, 20, 30),
                      (3, 4, 6, 8), (3, 4, 6, 8, 12))[number])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.2)
        ax.set_ylim(0, 3000)
    axes[0].set_ylabel('Mean capped journey loss (s)')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.24), ncol=4, frameon=False)
    fig.suptitle('Charged execution clock; 4 date/query cells per panel', fontsize=11)
    fig.tight_layout(rect=(0, .08, 1, .96))
    fig.savefig(HERE / 'city_quality_compute.png', dpi=180, bbox_inches='tight')
    fig.savefig(HERE / 'city_quality_compute.pdf', bbox_inches='tight')
    # A shadow-clock estimate does not become a runtime certificate merely
    # because its final policy is revalued once at that estimate.
    result['latency_secondary_shadow_error_seconds'] = {
        name: [d['execution_clock_seconds'] - d['estimated_execution_clock']
               for r in load(name)['rows'] for d in r['decisions'] if 'estimated_execution_clock' in d]
        for name in ('city_latency_recourse_charged', 'city_october_latency_recourse_charged',
                     'city_confirmation_latency_secondary')}
    logical = {(r['date'], r['query'], r['method']): (r['status'], r['loss_seconds'])
               for r in load('city_confirmation_logical')['rows']}
    charged = {(r['date'], r['query'], r['method']): (r['status'], r['loss_seconds'])
               for r in load('city_confirmation_charged')['rows']}
    result['confirmation_logical_and_charged_losses_identical'] = logical == charged
    (HERE / 'city_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: value.get('analytic_minus_uct_mean_bounds')
                      for name, value in result.items() if isinstance(value, dict)}, indent=2))


if __name__ == '__main__':
    main()
