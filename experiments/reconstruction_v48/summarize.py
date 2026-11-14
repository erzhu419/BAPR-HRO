"""All frozen states, saved certificates, actual stops and complete fees."""
import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.reconstruction_v34.discover import read
from experiments.reconstruction_v38.run import save
from experiments.reconstruction_v47 import audit as old_audit
from experiments.reconstruction_v48 import audit

HERE = Path(__file__).resolve().parent
SOLVERS = ('legacy47', 'global_a_full', 'global_both_full', 'global_both_stop')


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def complete_checkpoint_cost(bank, point):
    return {key: bank['prefix_cost'][key]+point['cost'][key] for key in ('process_cpu', 'wall')}


def summarize(lazy=False):
    contract = read(HERE/'contract.json')
    labels = list(contract['configurations'])
    solvers = SOLVERS
    if lazy:
        labels += read(HERE/'lazy_contract.json')['configurations']
        solvers += ('lazy_both_full', 'lazy_both_stop')
    records, flat, curve_rows = [], [], []
    for index in contract['state_indices']:
        banks = {label:read(HERE/'natural'/f'{index:03}_{label}.json.gz')
                 for label in labels}
        markov = banks['markov128']
        sample_validation = old_audit.frozen_samples(index, markov, banks['full128'], HERE.parent)
        old = read(HERE.parent/'reconstruction_v47/natural'/f'{index:03}_bulk_recycle.json.gz')
        kappa = read(HERE.parent/'reconstruction_v45/banks'/f'{index:03}_absorbing.json.gz')
        kappa_gap = kappa['paired']['full_bank_mean_gain_interval']
        old_audit.matching_values(banks['legacy47'], old)
        record = dict(index=index, root_index=markov['root_index'], v45_action=markov['selected_action'],
            sample_validation=sample_validation, population_kappa_gap=kappa_gap,
            samples={label:dict(action=banks[label]['selected_action'], paired=banks[label]['paired'],
                              cost=banks[label]['complete_cost']) for label in ('markov128', 'full128')},
            results={})
        for label in solvers:
            bank = banks[label]
            for field in ('clock', 'deadline', 'public_history', 'source', 'base_action'):
                if bank[field] != markov[field]:
                    raise ValueError(f'Frozen input changed: {index}/{label}/{field}')
            checked = audit.bounded_receipt(bank)
            old_audit.overlapping_legacy_bounds(bank, old)
            old_audit.exact_legacy_arms(bank, old)
            audit.selected_action(bank, markov['selected_action'])
            _, first = audit.first_certificate(bank)
            low, high = bank['difference_interval']
            row = dict(actions=bank['actions'], gap=[low, high], width=high-low,
                certified_action=bank['certified_action'], selected_action=bank['selected_action'],
                fallback_used=bank['fallback_used'], complete=bank['complete'],
                charged=bank['counters']['charged_compiles'],
                cost=bank['complete_cost'], counters=bank['counters'],
                allocation=bank['budget_allocation'], stop_reason=bank.get('stop_reason', 'original_V47_schedule'),
                first_certificate=None if first is None else dict(
                    action=first['certified_action'], gap=first['difference_interval'],
                    charged=first['counters']['charged_compiles'], phase=first['phase'],
                    cost=complete_checkpoint_cost(bank, first)),
                v45_regret_upper=old_audit.regret_upper(bank, markov['selected_action']),
                full_pi_minus_population_kappa_gap=[low-kappa_gap[1], high-kappa_gap[0]],
                audit=checked, frontier=bank.get('frontier_summary'),
                cost_breakdown=bank['cost'].get('components_cpu'))
            record['results'][label] = row
            flat.append(dict(index=index, root_index=bank['root_index'], label=label,
                certified_action=row['certified_action'], selected_action=row['selected_action'],
                fallback_used=row['fallback_used'], gap_lower=low, gap_upper=high, gap_width=high-low,
                complete=row['complete'], charged=row['charged'],
                alight_unresolved=bank['actions']['alight']['unresolved_probability'],
                continue_unresolved=bank['actions']['continue']['unresolved_probability'],
                alight_pending=bank['actions']['alight'].get('pending_dependencies'),
                continue_pending=bank['actions']['continue'].get('pending_dependencies'),
                v45_regret_upper=row['v45_regret_upper'],
                first_certificate_charged=None if first is None else row['first_certificate']['charged'],
                first_certificate_cpu=None if first is None else row['first_certificate']['cost']['process_cpu'],
                complete_cpu=row['cost']['process_cpu'], complete_wall=row['cost']['wall'],
                stop_reason=row['stop_reason']))
            for sequence, point in enumerate(bank['checkpoints']):
                timing = complete_checkpoint_cost(bank, point)
                curve_rows.append(dict(index=index, root_index=bank['root_index'], label=label,
                    checkpoint=sequence, phase=point['phase'], charged=point['counters']['charged_compiles'],
                    gap_lower=point['difference_interval'][0], gap_upper=point['difference_interval'][1],
                    gap_width=point['difference_interval'][1]-point['difference_interval'][0],
                    certified_action=point['certified_action'],
                    alight_unresolved=point['actions']['alight']['unresolved_probability'],
                    continue_unresolved=point['actions']['continue']['unresolved_probability'],
                    complete_cpu=timing['process_cpu'], complete_wall=timing['wall']))
        record['early_stop_prefix'] = audit.stopped_prefix(banks['global_both_full'], banks['global_both_stop'])
        if lazy:
            record['lazy_stop_prefix'] = audit.stopped_prefix(banks['lazy_both_full'], banks['lazy_both_stop'])
            for ending in ('full', 'stop'):
                eager, deferred = banks['global_both_'+ending], banks['lazy_both_'+ending]
                old_audit.matching_values(eager, deferred)
                if len(eager['checkpoints']) != len(deferred['checkpoints']):
                    raise ValueError('Lazy maintenance changed checkpoint sequence')
                for a, b in zip(eager['checkpoints'], deferred['checkpoints'], strict=True):
                    old_audit.matching_values(a, b)
                    if a['counters']['charged_compiles'] != b['counters']['charged_compiles']:
                        raise ValueError('Lazy maintenance changed paid checkpoint work')
                for key in ('charged_compiles', 'dag_nodes', 'dag_expansions'):
                    if eager['counters'][key] != deferred['counters'][key]:
                        raise ValueError(f'Lazy maintenance changed retained physical work: {key}')
            record['lazy_eager_scientific_and_work_equivalence'] = True
        records.append(record)
    aggregate = {}
    for label in solvers:
        rows = [r['results'][label] for r in records]
        aggregate[label] = dict(certified=sum(r['certified_action'] is not None for r in rows),
            complete=sum(r['complete'] for r in rows), mean_width=float(np.mean([r['width'] for r in rows])),
            mean_charged=float(np.mean([r['charged'] for r in rows])),
            mean_cpu=float(np.mean([r['cost']['process_cpu'] for r in rows])),
            mean_wall=float(np.mean([r['cost']['wall'] for r in rows])),
            changed_vs_v45=sum(r['results'][label]['selected_action'] != r['v45_action'] for r in records))
    for label in ('markov128', 'full128'):
        aggregate[label] = dict(mean_cpu=float(np.mean([r['samples'][label]['cost']['process_cpu'] for r in records])),
            mean_wall=float(np.mean([r['samples'][label]['cost']['wall'] for r in records])),
            scope='Fixed128-world estimate, not deterministic population certification')
    result = dict(stage='V48', states=len(records), calls=len(records)*len(labels),
        aggregate=aggregate, records=records, original_model_menu_and_information_preserved=True,
        supplied_prototype_reused=True, old_receipts_preserved=True, g2_passed=False,
        independent_method_contribution_established=False, new_online_replays=0, server_transfer_bytes=0,
        scope='Root-local complete-pi certificates under frozen P0 and unchanged W/pi. '
              'Ordinary root-influence scheduling is the strengthened baseline, not a new algorithmic principle. '
              'First saved certificate costs and actual stop settlement fees are distinct; floating point ranges '
              'may certify direction while unresolved dependencies remain. No journey-time gain inferred.')
    output = HERE if lazy else HERE/'eager_pass'
    output.mkdir(exist_ok=True)
    save(output/'summary.json', result)
    write_csv(output/'paired.csv', flat)
    write_csv(output/'checkpoints.csv', curve_rows)
    plot(records, output, lazy)
    print(aggregate, flush=True)
    return result


def plot(records, output, lazy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (gap, timing) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    configs = ('legacy47', 'lazy_both_full', 'lazy_both_stop') if lazy else (
        'legacy47', 'global_both_full', 'global_both_stop')
    names = ('V47 DFS', 'Both-arm priority / full budget', 'Both-arm priority / actual stop')
    colors = ('tab:gray', 'tab:orange', 'tab:blue')
    for j, (label, name, color) in enumerate(zip(configs, names, colors, strict=True)):
        for i, record in enumerate(records):
            row = record['results'][label]
            y = i+(j-1)*.18
            gap.plot(row['gap'], [y, y], color=color, linewidth=1.5, marker='o', markersize=3,
                     label=name if i == 0 else None)
            timing.scatter(row['cost']['process_cpu'], y, edgecolor=color,
                           facecolor=color if row['certified_action'] else 'none', s=30)
    gap.axvline(0, color='black', linestyle='--', linewidth=.8)
    gap.set(yticks=range(len(records)), yticklabels=[r['root_index'] for r in records],
            ylabel='Frozen root', xlabel='ALIGHT minus CONTINUE interval (s)',
            title='Full-pi population bounds')
    timing.set(xlabel='Complete query CPU (s)', title='Filled markers: certified action')
    for axis in (gap, timing):
        axis.grid(alpha=.2)
    handles, labels = gap.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(output/'quality_time.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lazy', action='store_true')
    args = parser.parse_args()
    summarize(lazy=args.lazy)
