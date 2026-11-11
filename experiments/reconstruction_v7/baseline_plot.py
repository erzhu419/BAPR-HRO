"""Observed legacy-contract quality and actual pipeline costs, not new data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V6 = HERE.parent / 'reconstruction_v6'


def main():
    current = json.loads((HERE / 'baseline_native_logical.json').read_text())
    old = json.loads((V6 / 'native_city_interned_results.json').read_text())
    references = json.loads((V6 / 'city_itinerary_logical.json').read_text())
    points = []
    for name, rows in [('Strict native: old port', [r for r in old['rows'] if r['clock'] == 'logical']),
                       ('Strict native: optimized', current['rows'])]:
        points.append((name, np.mean([r['end_to_end_ms'] / 1000 for r in rows]),
                       np.mean([r['loss_seconds'] for r in rows])))
    for method, name in [('compiled_nominal', 'Compiled nominal'), ('evolving_uct', 'UCT64'),
                         ('analytic_recourse', 'Analytic recourse')]:
        rows = [r for r in references['rows'] if r['method'] == method]
        points.append((name, np.mean([r['end_to_end_ms'] / 1000 for r in rows]),
                       np.mean([r['loss_seconds'] for r in rows])))
    fig, ax = plt.subplots(figsize=(7., 3.7))
    for (name, wall, loss), color, marker in zip(points,
            ['#8a8a8a', '#bd5d13', '#9c8c30', '#377eb8', '#23936b'], ['x', 'o', 'o', 'o', 'o']):
        ax.scatter(wall, loss, label=name, color=color, marker=marker, s=55)
    ax.plot([points[0][1], points[1][1]], [points[0][2], points[1][2]], color='#bd5d13', alpha=.4)
    ax.set(xscale='log', xlabel='Mean complete pipeline wall time (s)',
           ylabel='Mean capped journey loss (s)',
           title='July26/27 development, 4 cells; logical legacy contract')
    ax.grid(alpha=.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / 'baseline_quality_compute.png', dpi=170, bbox_inches='tight')
    fig.savefig(HERE / 'baseline_quality_compute.pdf', bbox_inches='tight')
    (HERE / 'baseline_quality_compute.json').write_text(json.dumps(
        {'scope': 'Descriptive measured cross-run pipeline costs; no causal novelty or C++ author-speed claim.',
         'points': [dict(name=name, pipeline_seconds=float(wall), loss_seconds=float(loss)) for name, wall, loss in points]}, indent=2) + '\n')


if __name__ == '__main__':
    main()
