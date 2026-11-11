"""Summarize the matched city budget panels without dropping censored trips."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ('evolving_rollout', 'evolving_uct', 'certified_rollout')


def read(stem):
    return json.loads((HERE / f'{stem}_results.json').read_text())


def metrics(rows):
    decisions = [d for r in rows for d in r['decisions']]
    return {
        'n': len(rows),
        'arrived': sum(r['status'] == 'arrived' for r in rows),
        'timeout': sum(r['status'] == 'timeout' for r in rows),
        'censored': sum(r['status'] == 'censored' for r in rows),
        'mean_loss_lower_seconds': float(np.mean([r['elapsed_observed_seconds'] for r in rows])),
        'mean_loss_upper_seconds': float(np.mean([
            r['loss_seconds'] if r['loss_seconds'] is not None else
            r['od']['deadline_seconds'] - r['od']['ready_seconds'] for r in rows])),
        'mean_end_to_end_seconds': float(np.mean([r['end_to_end_ms'] / 1000 for r in rows])),
        'decision_p50_p95_p99_seconds': np.quantile(
            [d['decision_total_ms'] / 1000 for d in decisions], [.5, .95, .99]).tolist(),
        'largest_image_hwm_kib': max(r['process_image_memory_kib']['VmHWM'] for r in rows),
        'mean_model_transitions': float(np.mean([r['planning_transitions'] for r in rows])),
        'maximum_decision_transitions': max(d['model_transitions'] for d in decisions),
        'mean_charged_ticks': float(np.mean([
            sum(d['computation_clock_ticks'] for d in r['decisions']) for r in rows])),
        'finite_bank_decisions': sum('finite_bank_gap' in d for d in decisions),
        'zero_bank_gap_decisions': sum(d.get('finite_bank_gap') == 0 for d in decisions),
        'mean_bank_gap_seconds': float(np.mean([d['finite_bank_gap'] for d in decisions
                                                if 'finite_bank_gap' in d]))
                                if any('finite_bank_gap' in d for d in decisions) else None}


def main():
    panels, paths, references = {}, [], {}
    for clock in ('logical', 'charged'):
        for budget in (64, 128, 512):
            stem = f'city_budget{budget}_{clock}' if budget != 512 else f'city_frontier_{clock}'
            source = read(stem)
            paths.append(f'{stem}_results.json')
            rows = source['rows']
            if budget == 512:
                for method in ('schedule_nominal', 'trained_nominal'):
                    references[f'{clock}/{method}'] = metrics([r for r in rows if r['method'] == method])
                cutoff = read(f'city_cutoff_{clock}')
                paths.append(f'city_cutoff_{clock}_results.json')
                rows = [r for r in rows if r['method'] != 'certified_rollout'] + cutoff['rows']
            for method in METHODS:
                group = [r for r in rows if r['method'] == method]
                if len(group) != 4 or any(r['transition_budget_per_decision'] != budget for r in group):
                    raise ValueError(f'Incomplete or mismatched city panel: {clock}/{budget}/{method}')
                values = metrics(group)
                if values['maximum_decision_transitions'] > budget:
                    raise ValueError('A policy exceeded the matched transition budget')
                panels[f'{clock}/{budget}/{method}'] = values
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=True)
    colors = ('#1767a6', '#c46d13', '#327844')
    labels = ('Causal rollout', 'UCT', 'Sparse certificate')
    for ax, clock in zip(axes, ('logical', 'charged')):
        for method, color, label in zip(METHODS, colors, labels):
            values = [panels[f'{clock}/{b}/{method}'] for b in (64, 128, 512)]
            x = [v['mean_end_to_end_seconds'] for v in values]
            y = [v['mean_loss_lower_seconds'] for v in values]
            upper = [v['mean_loss_upper_seconds'] for v in values]
            ax.errorbar(x, y, yerr=[np.zeros(len(y)), np.subtract(upper, y)],
                        fmt='o-', color=color, label=label, linewidth=1.4, markersize=4, capsize=3)
            for xi, yi, budget in zip(x, y, (64, 128, 512)):
                ax.annotate(str(budget), (xi, yi), xytext=(4, -13 if budget == 64 else 5),
                            textcoords='offset points', fontsize=7, color=color)
        ax.set_title('Logical event clock' if clock == 'logical' else 'Measured computation charged')
        ax.set_xlabel('Mean complete journey computation (s)')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Mean capped journey loss (s)')
    axes[1].legend(loc='best', fontsize=8, frameon=False)
    figure.suptitle('Four fixed city queries; labels are model-transition budgets', fontsize=11)
    figure.tight_layout()
    for extension in ('png', 'pdf'):
        figure.savefig(HERE / f'city_quality_cost.{extension}', dpi=180)
    result = {'sources': paths, 'scope': 'July26/27, two static OD queries each, July25-only training. '
              'Four date-query units, not independent simulated vehicles. The512 certificate uses the '
              'separately frozen witness-first cutoff run; baseline rows retain their original source protocol.',
              'timing': 'End-to-end includes graph/world and policy construction. Input reading and past-only '
              'fit are outside the passenger clock; decision processing is charged in the charged panel. '
              'Panels ran on the same32-logical-CPU host with up to2 workers per process; '
              'concurrent panels make these measured workload costs, not a portable hardware guarantee.',
              'panels': panels, 'nominal_references': references}
    (HERE / 'city_budget_summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
