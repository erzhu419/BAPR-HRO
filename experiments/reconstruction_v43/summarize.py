"""V43 all-case bridge diagnostics and full-cost strong-baseline comparisons."""
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reconstruction_v34.summarize import numeric, read, write
from experiments.reconstruction_v41.summarize import (
    OLD_CASES, compact_root, control_summary, grouped_comparison, meeting_science_equal, pair,
)

HERE=Path(__file__).resolve().parent
V41=HERE.parent/'reconstruction_v41'
LABELS=('rolling_fixed','one_step_rollout','optimal','full128')
LEVELS=('cheap','root','updated','working','full')


def point(value):
    return dict(lower=float(value),upper=float(value),mean=float(value),unresolved_probability=0.)


def difference(left,right):
    return dict(lower=left['lower']-right['upper'],upper=left['upper']-right['lower'],
        mean=None if left['mean'] is None or right['mean'] is None else left['mean']-right['mean'],
        unresolved_probability_bound=min(1.,left.get('unresolved_probability',left.get('unresolved_probability_bound',0.))+
            right.get('unresolved_probability',right.get('unresolved_probability_bound',0.))))


def distribution_summary(values):
    count=len(values)
    means=[v['mean'] for v in values]
    all_resolved=all(v is not None for v in means)
    low=sum(v['lower'] for v in values)/count
    high=sum(v['upper'] for v in values)/count
    mean=float(np.mean(means)) if all_resolved else None
    se=float(np.std(means,ddof=1)/np.sqrt(count)) if all_resolved and count>1 else None
    return dict(worlds=count,resolved_worlds=sum(v is not None for v in means),
        mean=mean,mean_interval=[low,high],paired_se=se,
        ci95=None if se is None else [mean-1.96*se,mean+1.96*se],
        mean_unresolved_probability_bound=sum(v.get('unresolved_probability',v.get('unresolved_probability_bound',0.)) for v in values)/count)


def error_aggregate(errors):
    """Average absolute-error intervals over every state, including unresolved."""
    bounds=[e['mean_interval'] for e in errors]
    absolute=[(0. if low<=0<=high else min(abs(low),abs(high)),max(abs(low),abs(high))) for low,high in bounds]
    return dict(denominator=len(errors),fully_resolved_states=sum(e['mean'] is not None for e in errors),
        mean_signed_error_interval=[sum(v[j] for v in bounds)/len(bounds) for j in (0,1)],
        mean_absolute_error_interval=[sum(v[j] for v in absolute)/len(absolute) for j in (0,1)],
        resolved_only_signed=numeric([e['mean'] for e in errors if e['mean'] is not None]))


def rank(value,base):
    low,high=value['mean_interval']
    if high<0:return 'alight'
    if low>0:return 'continue'
    if low==high==0:return base
    return 'undetermined'


def prefix_matches(prefix,original):
    trace=prefix['trace']
    if any(row['kind']=='platform' for row in trace):
        return False
    if trace!=original['trace'][:len(trace)]:
        return False
    if prefix['kind']=='terminal':
        return trace==original['trace'] and prefix['status']==original['status']
    # Platform prefix ends before its first local opportunity decision. A
    # no-offer terminal continuation legitimately has no platform trace row.
    remaining=original['trace'][len(trace):]
    return not remaining or (remaining[0]['kind']=='platform' and remaining[0]['stop']==prefix['stop']
                             and remaining[0]['clock']>=prefix.get('entry',float('-inf')))


def serialized_structure(node):
    if node['kind']=='ride':
        exits={row['cid']:row['next'] for row in node['exits']}
        return ['ride',node['first_cid'],node['path'],[[cid,None if exits[cid] is None else serialized_structure(exits[cid])]
            for cid in node['path'] if cid in exits]]
    return ['platform',node['stop'],[[row['cid'],serialized_structure(row['next'])]
                                   for row in sorted(node['rides'],key=lambda row:row['cid'])]]


def validate_reference_prefix(row,original,state,arm):
    prefix=row['prefix']
    if not prefix_matches(prefix,original):
        return False
    if prefix['decisions'].get('platform',0)!=0 or prefix['decisions'].get('onboard',0)!=len(prefix['trace']):
        return False
    if prefix['kind']=='terminal':
        return all(row[level][field]==original['loss'] for level in ('root','updated','working')
                   for field in ('lower','upper','mean'))
    if arm=='alight':
        if prefix['trace']:
            return False
        cid,entry=state['cid'],state['clock']
    else:
        if not prefix['trace'] or prefix['trace'][-1]['choice']!='alight':
            return False
        last=prefix['trace'][-1]
        cid,entry=last['cid'],last['clock']
    platform=next(item['next'] for item in state['node']['exits'] if item['cid']==cid)
    return (prefix['entry']==entry and platform is not None and prefix['stop']==platform['stop']
            and prefix['node_structure']==serialized_structure(platform))


def reference_summary():
    records=[]
    for index in range(148):
        ref=read(HERE/'references'/f'{index:03}.json.gz')
        full=read(V41/'banks'/f'{index:03}_shallow.json.gz')
        state=full['state']
        if (ref['index']!=index or ref['root_index']!=full['root_index'] or ref['seed']!=full['seed']
            or ref['count']!=128 or ref['generated_count']!=128 or ref['future_messages']!=0
            or ref['public_history']!=full['public_history'] or ref['clock']!=state['clock'] or ref['deadline']!=state['deadline']):
            raise ValueError(f'Frozen reference identity changed: {index}')
        cheap=full['base_receipt']
        rolling=ref['stopping_scores']['rolling_fixed']
        if rolling['action']!=full['base_action'] or any(abs(rolling['receipt'][key]-cheap[key])>1e-8 for key in ('alight','continuation')):
            raise ValueError(f'Fixed pi scoring changed: {index}')
        values={level:{arm:[] for arm in ('alight','continue')} for level in LEVELS}
        prefix_counts={arm:dict(platform=0,terminal=0,decision_count=0) for arm in ('alight','continue')}
        prefix_rows={arm:[] for arm in ('alight','continue')}
        for arm in ('alight','continue'):
            if len(ref['rows'][arm])!=128:
                raise ValueError(f'Missing paired prefix worlds: {index}/{arm}')
            for world,(row,original) in enumerate(zip(ref['rows'][arm],full['rows'][arm],strict=True)):
                if row['world_index']!=world or original['world_index']!=world or not validate_reference_prefix(row,original,state,arm):
                    raise ValueError(f'Fixed pi prefix changed: {index}/{arm}/{world}')
                prefix=row['prefix']
                prefix_counts[arm][prefix['kind']]+=1
                prefix_counts[arm]['decision_count']+=sum(prefix['decisions'].values())
                prefix_rows[arm].append(dict(world_index=world,kind=prefix['kind'],entry=prefix.get('entry'),
                    stop=prefix.get('stop'),status=prefix.get('status')))
                values['cheap'][arm].append(point(cheap['alight' if arm=='alight' else 'continuation']-state['clock']))
                values['full'][arm].append(point(original['loss']))
                for level in ('root','updated','working'):
                    values[level][arm].append(row[level])
        gaps={level:[difference(a,c) for a,c in zip(arms['alight'],arms['continue'],strict=True)] for level,arms in values.items()}
        summaries={level:dict(actions={arm:distribution_summary(rows) for arm,rows in arms.items()},
            gap=distribution_summary(gaps[level])) for level,arms in values.items()}
        errors={level:{arm:distribution_summary([difference(a,b) for a,b in zip(values[level][arm],values['full'][arm],strict=True)])
            for arm in ('alight','continue')} for level in LEVELS[:-1]}
        for level in LEVELS[:-1]:
            errors[level]['gap']=distribution_summary([difference(a,b) for a,b in zip(gaps[level],gaps['full'],strict=True)])
        bridges={}
        world_bridges={}
        for left,right in (('root','cheap'),('updated','root'),('full','updated'),('working','updated')):
            name=left+'_minus_'+right
            world_bridges[name]={arm:[difference(a,b) for a,b in zip(values[left][arm],values[right][arm],strict=True)] for arm in ('alight','continue')}
            world_bridges[name]['gap']=[difference(a,b) for a,b in zip(gaps[left],gaps[right],strict=True)]
            bridges[name]={arm:distribution_summary(rows) for arm,rows in world_bridges[name].items()}
        world_bridges={name:{arm:{field:[value[field] for value in rows]
            for field in ('lower','upper','mean')} for arm,rows in arms.items()} for name,arms in world_bridges.items()}
        ranks={level:rank(summaries[level]['gap'],full['base_action']) for level in LEVELS}
        scores={}
        for method,score in ref['stopping_scores'].items():
            a,c=(score['receipt'][name]-state['clock'] for name in ('alight','continuation'))
            scores[method]=dict(action=score['action'],alight=a,continuation=c,gap=a-c,
                gap_error=a-c-summaries['full']['gap']['mean'],cost={name:score[name] for name in ('process_cpu','wall')})
        records.append(dict(index=index,root_index=full['root_index'],source=full['source'],day=state['day'],cid=state['cid'],
            clock=state['clock'],levels=summaries,errors=errors,bridges=bridges,ranks=ranks,
            world_bridges=world_bridges,
            stopping_scores=scores,prefix_counts=prefix_counts,prefixes=prefix_rows,
            root_independence=ref['root_independence'],cache_stats=ref['cache_stats'],root_law_stats=ref['root_law_stats'],
            cost=ref['cost'],complete_cost=ref['complete_cost'],all_prefixes_match=True))
    aggregates={level:{arm:error_aggregate([row['errors'][level][arm] for row in records])
        for arm in ('alight','continue','gap')} for level in LEVELS[:-1]}
    sign_changes={level:dict(denominator=148,undetermined=sum(row['ranks'][level]=='undetermined' for row in records),
        differs_full=sum(row['ranks'][level] not in ('undetermined',row['ranks']['full']) for row in records),
        changed_from_cheap=sum(row['ranks'][level] not in ('undetermined',row['ranks']['cheap']) for row in records),
        resolves_cheap_disagreement=sum(row['ranks']['cheap']!=row['ranks']['full'] and row['ranks'][level]==row['ranks']['full'] for row in records),
        introduces_new_disagreement=sum(row['ranks']['cheap']==row['ranks']['full'] and row['ranks'][level] not in ('undetermined',row['ranks']['full']) for row in records),
        rows=[row['index'] for row in records if row['ranks'][level] not in ('undetermined',row['ranks']['full'])]) for level in LEVELS[:-1]}
    bridge_summary={name:{arm:error_aggregate([row['bridges'][name][arm] for row in records])
        for arm in ('alight','continue','gap')} for name in records[0]['bridges']}
    attribution=read(HERE/'existing_attribution.json')
    changed_sources={row['source'] for row in attribution['changed_online_gates']}
    stopping_summary={method:dict(action_changes_from_rolling=sum(row['stopping_scores'][method]['action']!=row['stopping_scores']['rolling_fixed']['action'] for row in records),
        action_disagreements_with_fixed_pi_mc=sum(row['stopping_scores'][method]['action']!=row['ranks']['full'] for row in records),
        gap_absolute_error=numeric([abs(row['stopping_scores'][method]['gap_error']) for row in records]))
        for method in LABELS[:-1]}
    return dict(states=148,worlds_per_state=128,all_fixed_pi_prefixes_match=True,
        world_bridge_array_order='Each per-world bridge array is in original world_index order 0..127.',
        error_summary=aggregates,sign_changes=sign_changes,bridge_summary=bridge_summary,
        stopping_score_summary=stopping_summary,
        original_eleven_changed_gate_states=[row['index'] for row in records if row['source'] in changed_sources],
        first_platform_counts={arm:{key:sum(row['prefix_counts'][arm][key] for row in records) for key in ('platform','terminal','decision_count')}
            for arm in ('alight','continue')},
        complete_cpu=numeric([row['complete_cost']['process_cpu'] for row in records]),records=records)


def online_summary():
    compact={label:[] for label in LABELS}
    compared={(label,'rolling_fixed') for label in LABELS}
    compared.update((label,'full128') for label in LABELS[:-1])
    pairs={a+'__vs__'+b:[] for a,b in sorted(compared)}
    cases=[]
    for index in range(72):
        roots={label:read(HERE/'roots'/f'{index:03}_{label}.json.gz') for label in LABELS}
        for label,root in roots.items():
            if root['root_index']!=index or root['root_id']!=roots['rolling_fixed']['root_id']:
                raise ValueError(f'Frozen online identity changed: {index}/{label}')
            if label in ('rolling_fixed','full128'):
                old_label='shallow_cheap' if label=='rolling_fixed' else 'shallow_rollout128'
                old=read(V41/'roots'/f'{index:03}_{old_label}.json.gz')
                if not meeting_science_equal(root,old):
                    raise ValueError(f'V41 comparator science changed: {index}/{label}')
            item=compact_root(root,f'roots/{index:03}_{label}.json.gz')
            if not item['cost_partition_exact']:
                raise ValueError(f'Incomplete CPU accounting: {index}/{label}')
            compact[label].append(item)
        for subject,baseline in sorted(compared):
            pairs[subject+'__vs__'+baseline].append(pair(roots[subject],roots[baseline]))
        first=roots['rolling_fixed']
        cases.append(dict(root_index=index,root_id=first['root_id'],day=first['day'],origin=first['origin'],goal=first['goal'],
            losses={label:r['outcome']['loss'] for label,r in roots.items()},
            statuses={label:r['outcome']['status'] for label,r in roots.items()},
            process_cpu={label:r['cost']['total_process_cpu'] for label,r in roots.items()},
            wall={label:r['cost']['total_wall'] for label,r in roots.items()}))
    attribution=read(HERE/'existing_attribution.json')
    zero=set(attribution['changed_but_zero_outcome_roots'])
    oldnegative={row['root_index'] for row in attribution['old_negative_roots']}
    return dict(controls={label:control_summary(rows) for label,rows in compact.items()},
        comparisons={label:grouped_comparison(rows) for label,rows in pairs.items()},
        root_cases=cases,root_receipts=compact,original_six_cases=[row for row in cases if row['root_index'] in OLD_CASES],
        original_four_gate_change_zero_outcome_cases=[row for row in cases if row['root_index'] in zero],
        historical_negative_cases=[dict(current=row,historical=next(old for old in attribution['historical_roots'] if old['root_index']==row['root_index']))
            for row in cases if row['root_index'] in oldnegative],all_v41_comparators_exact=True)


def outputs(result):
    fields=('configuration','mean_loss_lower','mean_loss_upper','mean_journey_cpu','mean_journey_wall','wins','harms','ties','mean_gain_vs_rolling')
    with (HERE/'controls.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for label,control in result['controls'].items():
            comp=result['comparisons'][label+'__vs__rolling_fixed']
            writer.writerow(dict(configuration=label,mean_loss_lower=control['quality']['full_cohort_mean_loss_interval'][0],
                mean_loss_upper=control['quality']['full_cohort_mean_loss_interval'][1],mean_journey_cpu=control['process_cpu']['mean'],
                mean_journey_wall=control['wall']['mean'],wins=comp['wins'],harms=comp['harms'],ties=comp['ties'],
                mean_gain_vs_rolling=comp['full_cohort_mean_gain']))
    fields=['index','root_index','clock']
    for level in LEVELS:
        fields.extend([f'{level}_{name}' for name in ('alight','continue','gap','gap_lower','gap_upper','gap_se','unresolved_bound','rank')])
    with (HERE/'reference.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in result['reference']['records']:
            item={key:row[key] for key in ('index','root_index','clock')}
            for level in LEVELS:
                value=row['levels'][level]
                item.update({f'{level}_alight':value['actions']['alight']['mean'],
                    f'{level}_continue':value['actions']['continue']['mean'],f'{level}_gap':value['gap']['mean'],
                    f'{level}_gap_lower':value['gap']['mean_interval'][0],f'{level}_gap_upper':value['gap']['mean_interval'][1],
                    f'{level}_gap_se':value['gap']['paired_se'],f'{level}_unresolved_bound':value['gap']['mean_unresolved_probability_bound'],
                    f'{level}_rank':row['ranks'][level]})
            writer.writerow(item)
    fig,axis=plt.subplots(figsize=(8,5))
    for label,control in result['controls'].items():
        low,high=control['quality']['full_cohort_mean_loss_interval']
        axis.plot([control['process_cpu']['mean']]*2,[low,high],marker='o',label=label)
    axis.set(xscale='log',xlabel='Mean whole-journey process CPU (s)',ylabel='Mean truncated loss over all 72 roots (s)')
    axis.grid(alpha=.2);axis.legend(fontsize=8);fig.tight_layout();fig.savefig(HERE/'quality_time.png',dpi=180);plt.close(fig)


def main():
    required=[HERE/'protocol.json',HERE/'existing_attribution.json']
    required += [HERE/'references'/f'{i:03}.json.gz' for i in range(148)]
    required += [HERE/'roots'/f'{i:03}_{label}.json.gz' for i in range(72) for label in LABELS]
    absent=[str(path.relative_to(HERE)) for path in required if not path.exists()]
    if absent:
        raise SystemExit(f'Awaiting {len(absent)} required V43 receipts; first missing: {absent[:5]}')
    result=dict(stage='V43',development_only=True,cohort=dict(states=148,paired_worlds_per_state=128,roots=72,controls=4,physical_replays=288),
        reference=reference_summary(),**online_summary(),
        limits='Fixed-pi bridge targets differ from full pi and telescope algebraically; not causal effects. '
            'All residual unresolved mass retained in bounds. Ordinary paired MC intervals are not certificates. '
            'Existing optimal stopping control is optimal only for supplied frozen marks; six days remain development data.')
    bridge_path='paired_bridges.json.gz'
    bridge_records=[dict(index=row['index'],root_index=row['root_index'],world_bridges=row.pop('world_bridges'))
                    for row in result['reference']['records']]
    with gzip.open(HERE/bridge_path,'wt',encoding='utf-8') as handle:
        json.dump(dict(world_index_order='0..127',states=bridge_records),handle,allow_nan=False)
    result['reference']['paired_bridge_records']=bridge_path
    write(HERE/'summary.json',result);outputs(result)
    print('V43 complete: 148 paired prefix references, 288 physical controls; old V41 comparators preserved.')


if __name__=='__main__':
    main()
