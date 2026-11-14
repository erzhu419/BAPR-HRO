"""All seven paired quality/cost results; no sampled means used as truth."""
import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.reconstruction_v34.discover import read
from experiments.reconstruction_v38.run import save
from experiments.reconstruction_v47 import audit

HERE = Path(__file__).resolve().parent
DETERMINISTIC = ('legacy46', 'incremental_split', 'incremental_recycle')


def csv_write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def costs(rows):
    return {kind: dict(mean=float(np.mean([r[kind] for r in rows])),
                      total=float(sum(r[kind] for r in rows)))
            for kind in ('process_cpu', 'wall')}


def summarize(hybrid=False, bulk=False):
    contract = read(HERE/'contract.json')
    labels = list(contract['configurations'])
    deterministic = DETERMINISTIC
    if hybrid or bulk:
        labels += read(HERE/'hybrid_amendment.json')['configurations']
        deterministic += ('hybrid_split', 'hybrid_recycle')
    if bulk:
        labels += read(HERE/'bulk_amendment.json')['configurations']
        deterministic += ('bulk_split', 'bulk_recycle')
    records, flat, checkpoints = [], [], []
    for index in contract['state_indices']:
        banks = {label: read(HERE/'natural'/f'{index:03}_{label}.json.gz')
                 for label in labels}
        legacy = read(HERE.parent/'reconstruction_v46/natural'/f'{index:03}_full_policy_oracle.json.gz')
        frozen = audit.frozen_samples(index, banks['markov128'], banks['full128'], HERE.parent)
        audit.matching_values(banks['legacy46'], legacy)
        reference = banks['markov128']
        record = dict(index=index, root_index=reference['root_index'],
            frozen_sample_validation=frozen,
            v45_action=reference['selected_action'], full128_action=banks['full128']['selected_action'],
            base_action=reference['base_action'],
            sample_costs={label:banks[label]['complete_cost'] for label in ('markov128', 'full128')},
            results={})
        for label in deterministic:
            bank = banks[label]
            for key in ('clock', 'deadline', 'public_history', 'source', 'base_action'):
                if bank[key] != reference[key]:
                    raise ValueError(f'Frozen input changed: {index}/{label}/{key}')
            audit.values(bank, clock=bank['clock'], deadline=bank['deadline'])
            audit.overlapping_legacy_bounds(bank, legacy)
            exact_checked = audit.exact_legacy_arms(bank, legacy)
            low, high = bank['difference_interval']
            selected = bank['certified_action'] or reference['selected_action']
            result = dict(actions=bank['actions'], difference_interval=[low, high],
                gap_width=high-low, certified_action=bank['certified_action'],
                complete=bank['complete'], original_v45_action=reference['selected_action'],
                certification_or_v45_fallback=selected,
                v45_regret_upper=audit.regret_upper(bank, reference['selected_action']),
                selected_regret_upper=audit.regret_upper(bank, selected),
                old_exact_arms_checked=exact_checked, counters=bank['counters'],
                cost=bank['complete_cost'], budget_allocation=bank['budget_allocation'])
            if label.startswith(('incremental_', 'hybrid_', 'bulk_')):
                result['curve_audit'] = audit.curve(bank['checkpoints'],
                    clock=bank['clock'], deadline=bank['deadline'])
                audit.matching_values(bank['checkpoints'][-1], bank)
                for sequence, point in enumerate(bank['checkpoints']):
                    a, c = (point['actions'][arm] for arm in audit.ARMS)
                    checkpoints.append(dict(index=index, root_index=bank['root_index'], label=label,
                        checkpoint=sequence, charged_compiles=point['counters']['charged_compiles'],
                        gap_lower=point['difference_interval'][0], gap_upper=point['difference_interval'][1],
                        gap_width=point['difference_interval'][1]-point['difference_interval'][0],
                        alight_unresolved=a['unresolved_probability'],
                        continue_unresolved=c['unresolved_probability'],
                        certified_action=point['certified_action'],
                        v45_regret_upper=audit.regret_upper(point, reference['selected_action']),
                        complete_cpu=bank['prefix_cost']['process_cpu']+point['cost']['process_cpu'],
                        complete_wall=bank['prefix_cost']['wall']+point['cost']['wall']))
            record['results'][label] = result
            flat.append(dict(index=index, root_index=bank['root_index'], label=label,
                gap_lower=low, gap_upper=high, gap_width=high-low,
                certified_action=bank['certified_action'], complete=bank['complete'],
                v45_action=reference['selected_action'], selected_action=selected,
                v45_regret_upper=result['v45_regret_upper'],
                alight_unresolved=bank['actions']['alight']['unresolved_probability'],
                continue_unresolved=bank['actions']['continue']['unresolved_probability'],
                charged_compiles=bank['counters']['charged_compiles'],
                complete_cpu=bank['complete_cost']['process_cpu'],
                complete_wall=bank['complete_cost']['wall']))
        record['allocation_effect'] = {}
        kernels = ('incremental', 'hybrid', 'bulk') if bulk else ('incremental', 'hybrid') if hybrid else ('incremental',)
        for kernel in kernels:
            fixed, pooled = (banks[kernel+suffix] for suffix in ('_split', '_recycle'))
            common = next(p for p in pooled['checkpoints'] if p['phase'] == 'after_continue')
            if common['counters']['charged_compiles'] != fixed['counters']['charged_compiles']:
                raise ValueError('Shared split/recycle prefix used different quotas')
            audit.matching_values(fixed, common)
            for arm in audit.ARMS:
                a, b = fixed['actions'][arm], pooled['actions'][arm]
                if b['lower'] < a['lower']-audit.TOL or b['upper'] > a['upper']+audit.TOL:
                    raise ValueError('Quota recycling widened the split endpoint')
            a, b = (record['results'][kernel+suffix] for suffix in ('_split', '_recycle'))
            record['allocation_effect'][kernel] = dict(shared_prefix_equal=True,
                recycled_compiles=pooled['counters']['charged_compiles']-fixed['counters']['charged_compiles'],
                gap_width_reduction=a['gap_width']-b['gap_width'],
                v45_regret_bound_reduction=a['v45_regret_upper']-b['v45_regret_upper'],
                complete_cpu_change=b['cost']['process_cpu']-a['cost']['process_cpu'])
        records.append(record)
    aggregate = {}
    for label in deterministic:
        rows = [r['results'][label] for r in records]
        aggregate[label] = dict(certifications=sum(r['certified_action'] is not None for r in rows),
            exact_states=sum(r['complete'] for r in rows),
            mean_gap_width=float(np.mean([r['gap_width'] for r in rows])),
            mean_v45_regret_upper=float(np.mean([r['v45_regret_upper'] for r in rows])),
            mean_charged_compiles=float(np.mean([r['counters']['charged_compiles'] for r in rows])),
            cost=costs([r['cost'] for r in rows]))
    for label in ('markov128', 'full128'):
        aggregate[label] = dict(cost=costs([r['sample_costs'][label] for r in records]),
            scope='Unchanged128-world point estimate; not deterministic full-pi certification')
    result = dict(stage='V47', states=len(records), fresh_calls=len(records)*len(labels),
        aggregate=aggregate, records=records,
        certifications_are_root_local=True, independent_method_contribution_established=False,
        g2_passed=False, online_controller_launched=False, server_transfer_bytes=0,
        scope='Same-kernel2000-entry split/recycle comparison; legacy46 has different primitive accounting. '
              'Every complete query cost is measured. Bounds concern fixed P0 complete pi; sample estimates '
              'and root-local certificates do not establish whole-journey quality or independent novelty.')
    output = HERE if bulk else HERE/'terminal_pass' if hybrid else HERE/'first_pass'
    output.mkdir(exist_ok=True)
    save(output/'summary.json', result)
    csv_write(output/'paired.csv', flat)
    csv_write(output/'checkpoints.csv', checkpoints)
    plot(records, output, 'bulk' if bulk else 'hybrid' if hybrid else 'incremental')
    print({label: {k:v for k,v in value.items() if k != 'cost'} for label,value in aggregate.items()}, flush=True)
    return result


def plot(records, output, kernel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (axis, quality) = plt.subplots(1, 2, figsize=(11, 5))
    colors = ('tab:gray', 'tab:orange', 'tab:blue')
    names = ('V46 original', 'Retained bulk DP / split', 'Retained bulk DP / recycle') if kernel != 'incremental' else (
        'V46 original', 'Retained event DP / split', 'Retained event DP / recycle')
    shown = ('legacy46', kernel+'_split', kernel+'_recycle')
    for j, (label, color, name) in enumerate(zip(shown, colors, names, strict=True)):
        for i, record in enumerate(records):
            row = record['results'][label]
            y = i+(j-1)*.18
            axis.plot(row['difference_interval'], [y, y], color=color, marker='o',
                      markersize=3, linewidth=1.6, label=name if i == 0 else None)
            quality.scatter(row['cost']['process_cpu'], row['gap_width'], color=color,
                            label=name if i == 0 else None, s=24)
            quality.annotate(str(record['root_index']),
                (row['cost']['process_cpu'], row['gap_width']), fontsize=7,
                xytext=((3, 5), (-2, -12), (4, 5))[j],
                textcoords='offset points')
    axis.axvline(0, color='black', linestyle='--', linewidth=.8)
    axis.set(yticks=range(len(records)), yticklabels=[r['root_index'] for r in records],
             ylabel='Frozen root', xlabel='ALIGHT minus CONTINUE bound (s)',
             title='All seven population gap intervals')
    quality.set(xlabel='Complete query CPU (s)', ylabel='Population gap interval width (s)',
                title='Same guarantee target; lower and left is better')
    for ax in (axis, quality):
        ax.grid(alpha=.2)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(output/'quality_time.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('--hybrid', action='store_true')
    stages.add_argument('--bulk', action='store_true')
    args = parser.parse_args()
    summarize(hybrid=args.hybrid, bulk=args.bulk)
