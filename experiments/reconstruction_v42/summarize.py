"""Complete-cohort V42 quality, coalescence and cost comparison."""
import csv
from numbers import Real
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.reconstruction_v34.summarize import numeric, read, write
from experiments.reconstruction_v41.summarize import (
    OLD_CASES, compact_root, control_summary, grouped_comparison, meeting_science_equal, pair,
)

HERE = Path(__file__).resolve().parent
V41 = HERE.parent/'reconstruction_v41'
FAMILIES = ('physical', 'shallow')
MODES = ('full', 'entry', 'event_full', 'event_live')
LABELS = ('shallow_cheap', 'shallow_full128', 'shallow_event_full128', 'shallow_event_live128')


def scientific_bank(row):
    return {**{key:row[key] for key in ('seed','count','generated_count','public_history','future_messages','paired')},
        'rows':{arm:[{key:r[key] for key in ('loss','status','trace','decisions','world_index')}
                     for r in records] for arm,records in row['rows'].items()}}


def require_same_bank(subject, baseline, label):
    if scientific_bank(subject) != scientific_bank(baseline):
        raise ValueError(f'Conditional science changed: {label}')
    if any(subject[key] != baseline[key] for key in ('base_action','selected_action')):
        raise ValueError(f'Conditional action changed: {label}')


def online_science_equal(subject, baseline):
    actions=lambda row:[(g['base_action'],g['selected_action']) for g in row['gates']]
    return meeting_science_equal(subject,baseline) and actions(subject)==actions(baseline)


def total_stats(rows):
    """Add recorded counters/CPU once; do not mix them into total runtime."""
    keys = set().union(*(row.keys() for row in rows))
    return {key:sum(row.get(key,0) for row in rows) for key in sorted(keys)
            if all(isinstance(row.get(key,0), Real) for row in rows)}


def paired_cost(subject, baseline):
    return dict(denominator=len(subject),
        process_cpu_difference=numeric([a['process_cpu']-b['process_cpu'] for a,b in zip(subject,baseline,strict=True)]),
        wall_difference=numeric([a['wall']-b['wall'] for a,b in zip(subject,baseline,strict=True)]),
        aggregate_cpu_ratio=sum(a['process_cpu'] for a in subject)/sum(b['process_cpu'] for b in baseline),
        faster_states=sum(a['process_cpu']<b['process_cpu'] for a,b in zip(subject,baseline,strict=True)))


def reference_summary():
    records, arms = [], {f'{family}_{mode}':[] for family in FAMILIES for mode in MODES}
    for index in range(148):
        row = dict(index=index, families={})
        for family in FAMILIES:
            old = read(V41/'banks'/f'{index:03}_{family}.json.gz')
            banks = {mode:read(HERE/'banks'/f'{index:03}_{family}_{mode}.json.gz') for mode in MODES}
            row.update(root_index=old['root_index'], day=old['state']['day'],
                       clock=old['state']['clock'],cid=old['state']['cid'])
            compact = {}
            for mode, bank in banks.items():
                if (bank['index'] != index or bank['root_index'] != old['root_index']
                    or bank['count'] != 128 or bank['generated_count'] != 128 or bank['state'] != old['state']):
                    raise ValueError(f'Frozen state changed: {index}/{family}/{mode}')
                if bank['v41_science_exact'] is not True:
                    raise ValueError(f'V41 check failed: {index}/{family}/{mode}')
                require_same_bank(bank,old,f'{index}/{family}/{mode}/V41')
                require_same_bank(bank,banks['full'],f'{index}/{family}/{mode}/full')
                item = dict(base_action=bank['base_action'],selected_action=bank['selected_action'],
                    paired=bank['paired'],complete_cost=bank['complete_cost'],
                    evaluation_cost={k:bank['cost'][k] for k in ('total_process_cpu','total_wall')},
                    meeting_stats=bank['meeting_stats'],decision_stats=bank['cost']['decision_stats'],
                    planner_stats=bank['cost']['planner_stats'])
                compact[mode] = item
                arms[f'{family}_{mode}'].append(item)
            row['families'][family] = compact
        records.append(row)
    configurations = {label:dict(states=len(rows),
        complete_cpu=numeric([r['complete_cost']['process_cpu'] for r in rows]),
        complete_wall=numeric([r['complete_cost']['wall'] for r in rows]),
        evaluation_cpu=numeric([r['evaluation_cost']['total_process_cpu'] for r in rows]),
        meeting_stats=total_stats([r['meeting_stats'] for r in rows]),
        decision_stats=total_stats([r['decision_stats'] for r in rows]),
        planner_stats=total_stats([r['planner_stats'] for r in rows]),
        unresolved_pairs=sum(r['paired']['unresolved_pairs'] for r in rows)) for label,rows in arms.items()}
    comparisons = {}
    for family in FAMILIES:
        base = [r['complete_cost'] for r in arms[f'{family}_full']]
        for mode in MODES[1:]:
            label=f'{family}_{mode}'
            comparisons[label+'__vs__'+family+'_full']=paired_cost([r['complete_cost'] for r in arms[label]],base)
            comparisons[label+'__vs__'+family+'_full']['unique_score_counts_equal_states']={
                kind:sum(subject['decision_stats'][kind+'_unique']==baseline['decision_stats'][kind+'_unique']
                    for subject,baseline in zip(arms[label],arms[f'{family}_full'],strict=True))
                for kind in ('onboard','platform')}
    return dict(states=148,banks=148*len(arms),worlds_per_bank=128,all_v41_science_exact=True,
        all_variants_science_exact=True,configurations=configurations,comparisons=comparisons,records=records)


def online_summary():
    compact={label:[] for label in LABELS}
    compared={(label,'shallow_cheap') for label in LABELS}
    compared.update((label,'shallow_full128') for label in LABELS[2:])
    pairs={a+'__vs__'+b:[] for a,b in sorted(compared)}
    cases=[]
    for index in range(72):
        roots={label:read(HERE/'roots'/f'{index:03}_{label}.json.gz') for label in LABELS}
        for label,root in roots.items():
            if root['root_index'] != index:
                raise ValueError(f'Wrong root: {index}/{label}')
            source='shallow_cheap' if label=='shallow_cheap' else 'shallow_rollout128'
            old=read(V41/'roots'/f'{index:03}_{source}.json.gz')
            if root['root_id'] != old['root_id'] or root['loss_cap'] != old['loss_cap']:
                raise ValueError(f'Frozen root identity changed: {index}/{label}')
            if not online_science_equal(root,old):
                raise ValueError(f'Online V41 science changed: {index}/{label}')
            if label not in ('shallow_cheap','shallow_full128') and not online_science_equal(root,roots['shallow_full128']):
                raise ValueError(f'Online full science changed: {index}/{label}')
            item=compact_root(root,f'roots/{index:03}_{label}.json.gz')
            if not item['cost_partition_exact']:
                raise ValueError(f'Whole-journey cost partition failed: {index}/{label}')
            item['meeting_stats']=total_stats([gate['meeting_stats'] for gate in root['gates']])
            item['gate_decision_stats']=total_stats([gate['cost']['decision_stats'] for gate in root['gates']])
            item['gate_planner_stats']=total_stats([gate['cost']['planner_stats'] for gate in root['gates']])
            compact[label].append(item)
        for subject,baseline in sorted(compared):
            pairs[subject+'__vs__'+baseline].append(pair(roots[subject],roots[baseline]))
        first=roots[LABELS[0]]
        cases.append(dict(root_index=index,root_id=first['root_id'],day=first['day'],origin=first['origin'],
            goal=first['goal'],losses={label:root['outcome']['loss'] for label,root in roots.items()},
            statuses={label:root['outcome']['status'] for label,root in roots.items()},
            process_cpu={label:root['cost']['total_process_cpu'] for label,root in roots.items()}))
    controls={}
    for label,rows in compact.items():
        # The shared V41 helper aggregates a fixed set of counters. All V42
        # statistics are additionally retained below without double billing.
        stats=[row['meeting_stats'] for row in rows]
        for row in rows:
            row['meeting_stats']={key:row['meeting_stats'].get(key,0)
                for key in ('checks','hits','avoided_decisions','check_cpu')}
        controls[label]=control_summary(rows)
        controls[label]['meeting_stats']=total_stats(stats)
        controls[label]['gate_decision_stats']=total_stats([row['gate_decision_stats'] for row in rows])
        controls[label]['gate_planner_stats']=total_stats([row['gate_planner_stats'] for row in rows])
        for row,details in zip(rows,stats,strict=True):
            row['meeting_stats']=details
    unique_checks={label:{kind:sum(subject['gate_decision_stats'].get(kind+'_unique',0)==baseline['gate_decision_stats'].get(kind+'_unique',0)
        for subject,baseline in zip(compact[label],compact['shallow_full128'],strict=True))
        for kind in ('onboard','platform')} for label in LABELS[2:]}
    return dict(controls=controls,comparisons={label:grouped_comparison(rows) for label,rows in pairs.items()},
        root_cases=cases,original_six_cases=[r for r in cases if r['root_index'] in OLD_CASES],
        root_receipts=compact,unique_score_counts_equal_roots=unique_checks,
        all_v41_science_exact=True,all_variants_science_exact=True)


def outputs(result):
    fields=('configuration','mean_loss_lower','mean_loss_upper','mean_journey_cpu','mean_journey_wall','wins','harms','ties',
            'mean_gain_vs_shallow_cheap','meeting_checks','meeting_hits','avoided_decisions')
    with (HERE/'controls.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for label,control in result['controls'].items():
            comparison=result['comparisons'][label+'__vs__shallow_cheap']
            writer.writerow(dict(configuration=label,
                mean_loss_lower=control['quality']['full_cohort_mean_loss_interval'][0],
                mean_loss_upper=control['quality']['full_cohort_mean_loss_interval'][1],
                mean_journey_cpu=control['process_cpu']['mean'],mean_journey_wall=control['wall']['mean'],
                wins=comparison['wins'],harms=comparison['harms'],ties=comparison['ties'],
                mean_gain_vs_shallow_cheap=comparison['full_cohort_mean_gain'],
                meeting_checks=control['meeting_stats'].get('checks',0),meeting_hits=control['meeting_stats'].get('hits',0),
                avoided_decisions=control['meeting_stats'].get('avoided_decisions',0)))
    fig,axis=plt.subplots(figsize=(8,5))
    for label,control in result['controls'].items():
        low,high=control['quality']['full_cohort_mean_loss_interval']
        axis.plot([control['process_cpu']['mean']]*2,[low,high],marker='o',label=label)
    axis.set(xscale='log',xlabel='Mean whole-journey process CPU (s)',
             ylabel='Mean truncated travel loss over all 72 roots (s)')
    axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/'quality_cpu.png',dpi=180);plt.close(fig)


def main():
    required=[HERE/'protocol.json']
    required += [HERE/'banks'/f'{index:03}_{family}_{mode}.json.gz'
                 for index in range(148) for family in FAMILIES for mode in MODES]
    required += [HERE/'roots'/f'{index:03}_{label}.json.gz' for index in range(72) for label in LABELS]
    absent=[str(p.relative_to(HERE)) for p in required if not p.exists()]
    if absent:
        raise SystemExit(f'Awaiting {len(absent)} declared receipts; first missing: {absent[:5]}')
    reference=reference_summary()
    online=online_summary()
    result=dict(stage='V42',protocol='protocol.json',development_only=True,
        cohort=dict(states=148,conditional_banks=1184,physical_roots=72,fixed_controls=4,physical_replays=288),
        reference=reference,**online,
        limitations='Frozen V41 model, menus and information. Same-bank parity is correctness, not novelty. '
                    'CPU phases are subsets of whole-journey CPU. All six dates remain development data.')
    write(HERE/'summary.json',result)
    outputs(result)
    print('V42 complete: 1184 banks, 288 replays; all conditional and online science exactly match V41.')


if __name__ == '__main__':
    main()
