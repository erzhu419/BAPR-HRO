"""Validate and compare all seven roots without treating samples as truth."""
import csv
from pathlib import Path

import numpy as np

from experiments.reconstruction_v34.discover import read
from experiments.reconstruction_v38.run import save
from experiments.reconstruction_v47 import audit as old
from experiments.reconstruction_v48 import audit
from experiments.reconstruction_v48.summarize import complete_checkpoint_cost

HERE = Path(__file__).resolve().parent
SOLVERS = ('v48_full', 'v48_stop', 'shared_full', 'shared_stop', 'ordinary_full', 'ordinary_stop')


def equivalent_work(left, right):
    """The bulk optimization alone must preserve the measured scientific work."""
    old.matching_values(left, right)
    if len(left['checkpoints']) != len(right['checkpoints']):
        raise ValueError('Bulk maintenance changed checkpoint cadence')
    maximum = 0.
    for a, b in zip(left['checkpoints'], right['checkpoints'], strict=True):
        old.matching_values(a, b)
        for key in ('charged_compiles', 'dag_expansions', 'dag_nodes'):
            if a['counters'][key] != b['counters'][key]:
                raise ValueError(f'Bulk maintenance changed work: {key}')
        for arm in old.ARMS:
            if a['actions'][arm]['pending_dependencies'] != b['actions'][arm]['pending_dependencies']:
                raise ValueError('Bulk maintenance lost unfinished dependencies')
            for key in ('lower', 'upper', 'unresolved_probability'):
                maximum = max(maximum, abs(a['actions'][arm][key]-b['actions'][arm][key]))
    return dict(checkpoints=len(left['checkpoints']), max_value_difference=maximum,
                same_charged_work_and_dag_counts=True)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    contract = read(HERE/'contract.json')
    records, flat, curves = [], [], []
    for index in contract['state_indices']:
        banks = {label:read(HERE/'natural'/f'{index:03}_{label}.json.gz')
                 for label in contract['configurations']}
        markov = banks['markov128']
        record = dict(index=index, root_index=markov['root_index'],
            v45_action=markov['selected_action'],
            frozen_samples=old.frozen_samples(index, markov, banks['full128'], HERE.parent),
            samples={label:dict(action=banks[label]['selected_action'], paired=banks[label]['paired'],
                cost=banks[label]['complete_cost']) for label in ('markov128', 'full128')}, results={})
        for label in SOLVERS:
            bank = banks[label]
            for key in ('clock', 'deadline', 'public_history', 'source', 'base_action'):
                if bank[key] != markov[key]:
                    raise ValueError(f'Frozen input changed: {index}/{label}/{key}')
            checked = audit.bounded_receipt(bank, max_compiles=contract['limits']['charged_compiles'])
            old.overlapping_legacy_bounds(bank, banks['v48_full'])
            audit.selected_action(bank, markov['selected_action'])
            _, first = audit.first_certificate(bank)
            low, high = bank['difference_interval']
            result = dict(actions=bank['actions'], gap=[low, high], width=high-low,
                certified_action=bank['certified_action'], selected_action=bank['selected_action'],
                fallback_used=bank['fallback_used'], complete=bank['complete'],
                charged=bank['counters']['charged_compiles'], cost=bank['complete_cost'],
                counters=bank['counters'], allocation=bank['budget_allocation'],
                stop_reason=bank['stop_reason'], audit=checked,
                first_certificate=None if first is None else dict(action=first['certified_action'],
                    charged=first['counters']['charged_compiles'], gap=first['difference_interval'],
                    cost=complete_checkpoint_cost(bank, first)),
                v45_regret_upper=old.regret_upper(bank, markov['selected_action']),
                cost_breakdown=bank['cost'].get('components_cpu'), frontier=bank['frontier_summary'])
            if 'relevance_profiles' in bank:
                result['relevance_profiles'] = bank['relevance_profiles']
            record['results'][label] = result
            flat.append(dict(index=index, root_index=bank['root_index'], label=label,
                certified_action=result['certified_action'], gap_lower=low, gap_upper=high,
                gap_width=high-low, charged=result['charged'],
                complete_cpu=result['cost']['process_cpu'], complete_wall=result['cost']['wall'],
                first_certificate_charged=None if first is None else first['counters']['charged_compiles'],
                selected_action=result['selected_action'], fallback_used=result['fallback_used'],
                complete=result['complete'], alight_pending=bank['actions']['alight']['pending_dependencies'],
                continue_pending=bank['actions']['continue']['pending_dependencies'],
                stop_reason=result['stop_reason']))
            for sequence, point in enumerate(bank['checkpoints']):
                timing = complete_checkpoint_cost(bank, point)
                curves.append(dict(index=index, root_index=bank['root_index'], label=label,
                    checkpoint=sequence, phase=point['phase'], charged=point['counters']['charged_compiles'],
                    gap_lower=point['difference_interval'][0], gap_upper=point['difference_interval'][1],
                    gap_width=point['difference_interval'][1]-point['difference_interval'][0],
                    certified_action=point['certified_action'],
                    complete_cpu=timing['process_cpu'], complete_wall=timing['wall']))
        record['stopped_prefixes'] = {name:audit.stopped_prefix(banks[name+'_full'], banks[name+'_stop'])
                                     for name in ('v48', 'shared', 'ordinary')}
        record['bulk_equivalence'] = {ending:equivalent_work(banks['v48_'+ending], banks['shared_'+ending])
                                      for ending in ('full', 'stop')}
        for ending in ('full', 'stop'):
            archived = read(HERE.parent/'reconstruction_v48/natural'/f'{index:03}_lazy_both_{ending}.json.gz')
            equivalent_work(archived, banks['v48_'+ending])
        # Dynamic ordinary projection may merge more states, changing finite-budget
        # work. Only overlapping valid bounds and identical stopping RULES are required.
        record['ordinary_comparison'] = {
            ending:dict(cpu_ratio=banks['ordinary_'+ending]['complete_cost']['process_cpu']/
                                  banks['shared_'+ending]['complete_cost']['process_cpu'],
                charged_difference=banks['ordinary_'+ending]['counters']['charged_compiles']-
                                   banks['shared_'+ending]['counters']['charged_compiles'],
                dag_nodes_difference=banks['ordinary_'+ending]['counters']['dag_nodes']-
                                     banks['shared_'+ending]['counters']['dag_nodes'],
                same_action=banks['ordinary_'+ending]['certified_action']==banks['shared_'+ending]['certified_action'])
            for ending in ('full', 'stop')}
        records.append(record)
    aggregate = {}
    for label in SOLVERS:
        rows = [r['results'][label] for r in records]
        aggregate[label] = dict(certified=sum(r['certified_action'] is not None for r in rows),
            exact_complete=sum(r['complete'] for r in rows),
            mean_cpu=float(np.mean([r['cost']['process_cpu'] for r in rows])),
            mean_wall=float(np.mean([r['cost']['wall'] for r in rows])),
            mean_charged=float(np.mean([r['charged'] for r in rows])),
            mean_width=float(np.mean([r['width'] for r in rows])),
            changed_vs_v45=sum(r['results'][label]['selected_action'] != r['v45_action'] for r in records),
            cpu_limited=sum(r['stop_reason']=='cpu_limit' for r in rows))
        aggregate[label]['mean_bulk_refresh_cpu'] = float(np.mean([
            r['counters']['priority_block_refresh_cpu'] for r in rows]))
        aggregate[label]['mean_bulk_reducer_calls'] = float(np.mean([
            r['counters']['bulk_interval_reductions'] for r in rows]))
        if label.startswith(('shared_', 'ordinary_')):
            actual = sum(r['counters']['bulk_recomputed_rows'] for r in rows)
            equivalent = sum(r['counters']['bulk_full_rows_equivalent'] for r in rows)
            aggregate[label]['bulk_row_work'] = dict(actual=actual,
                same_v49_call_sequence_full_reduction=equivalent,
                reduction_fraction=1-actual/equivalent if equivalent else 0.,
                scope='Same V49 reducer-call sequence, not a measured V48 row count; persistent memo also avoids calls')
    for label in ('markov128', 'full128'):
        aggregate[label] = dict(mean_cpu=float(np.mean([r['samples'][label]['cost']['process_cpu'] for r in records])),
            mean_wall=float(np.mean([r['samples'][label]['cost']['wall'] for r in records])),
            scope='Original128-world estimate, not a population action certificate')
    result = dict(stage='V49', states=len(records), natural_calls=len(records)*len(contract['configurations']),
        aggregate=aggregate, records=records, original_model_menu_information_preserved=True,
        original_per_query_limits_preserved=True, g2_passed=False,
        independent_method_contribution_established=False, new_online_replays=0, server_transfer_bytes=0)
    save(HERE/'summary.json', result)
    write_csv(HERE/'paired.csv', flat)
    write_csv(HERE/'checkpoints.csv', curves)
    plot(records, curves)
    print(aggregate, flush=True)
    return result


def plot(records, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    names = {'v48':'V48 full reductions', 'shared':'Incremental / shared interface',
             'ordinary':'Incremental / ordinary relevance'}
    colors = {'v48':'tab:gray', 'shared':'tab:blue', 'ordinary':'tab:orange'}
    for j, name in enumerate(names):
        color = colors[name]
        for i, record in enumerate(records):
            stop = record['results'][name+'_stop']
            y = i+(j-1)*.18
            axes[0, 0].plot(stop['gap'], [y, y], marker='o', markersize=3, color=color,
                label=names[name] if i == 0 else None)
            axes[0, 1].scatter(stop['cost']['process_cpu'], y, edgecolor=color,
                facecolor=color if stop['certified_action'] else 'none', s=30)
        for axis, xkey in ((axes[1, 0], 'charged'), (axes[1, 1], 'complete_cpu')):
            for i, record in enumerate(records):
                points = [p for p in curves if p['index']==record['index'] and p['label']==name+'_full']
                axis.plot([p[xkey] for p in points], [max(p['gap_width'], 1e-8) for p in points],
                    color=color, alpha=.55, linewidth=.8)
            axis.set_yscale('log')
    axes[0, 0].axvline(0, color='black', ls='--', lw=.8)
    for axis in axes[0]:
        axis.set_yticks(range(len(records)), [r['root_index'] for r in records])
        axis.set_ylabel('Frozen root')
    axes[0, 0].set(title='Actual-stop action gap', xlabel='ALIGHT minus CONTINUE (s)')
    axes[0, 1].set(title='Actual complete query; filled = certified', xlabel='CPU (s)')
    axes[1, 0].set(title='All seven full-budget curves', xlabel='Charged entries', ylabel='Gap width (s; display floor 1e-8)')
    axes[1, 1].set(title='Same curves with complete checkpoint costs', xlabel='CPU (s)', ylabel='Gap width (s; display floor 1e-8)')
    for axis in axes.flat:
        axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(HERE/'quality_time.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    summarize()
