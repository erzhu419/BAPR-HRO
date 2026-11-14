"""All-case V44 single-level matched control and full-cost paired summaries."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reconstruction_v34.summarize import numeric, read, write
from experiments.reconstruction_v40.summarize import cohort_quality
from experiments.reconstruction_v41.summarize import OLD_CASES, grouped_comparison, meeting_science_equal, pair
from experiments.reconstruction_v43.summarize import difference, distribution_summary, point

HERE=Path(__file__).resolve().parent
V43=HERE.parent/'reconstruction_v43'
LABELS=('rolling_fixed','one_step_rollout','optimal','full128','matched128')
BUDGETS=(.05,.1,.25,.5,1.)


def quantiles(values):
    result=numeric(values)
    result.update(p50=float(np.percentile(values,50)) if values else None,
                  p95=float(np.percentile(values,95)) if values else None)
    return result


def local_value_equal(fresh,saved):
    for key in ('lower','upper','mean','unresolved_probability'):
        a,b=fresh[key],saved[key]
        if (a is None)!=(b is None) or a is not None and not np.isclose(a,b,atol=1e-8,rtol=0):
            return False
    return True


def decision_budget(rows,measure,budget):
    samples=[decision[measure] for row in rows for decision in row['decision_costs']]
    return dict(decisions=len(samples),decisions_within=sum(value<=budget for value in samples),
        decision_fraction_within=sum(value<=budget for value in samples)/len(samples),
        roots=len(rows),all_decisions_within_roots=sum(all(d[measure]<=budget for d in row['decision_costs']) for row in rows))


def joint_budgets(subject,baseline):
    result={}
    for measure in ('process_cpu','wall'):
        result[measure]={}
        for budget in BUDGETS:
            states=[(all(d[measure]<=budget for d in a['decision_costs']),
                     all(d[measure]<=budget for d in b['decision_costs'])) for a,b in zip(subject,baseline,strict=True)]
            result[measure][str(budget)]=dict(roots=len(states),both_within=sum(a and b for a,b in states),
                subject_only=sum(a and not b for a,b in states),baseline_only=sum(b and not a for a,b in states),
                neither_within=sum(not a and not b for a,b in states))
    return result


def validate_partition(row):
    cost=row['cost']
    cpu=cost['decision_cpu_sum']+cost['final_execution_cpu']-cost['total_process_cpu']
    wall=sum(d['wall'] for d in row['decision_costs'])+cost['final_execution_wall']-cost['total_wall']
    if abs(cpu)>1e-9 or abs(wall)>1e-8:
        raise ValueError(f'Incomplete whole-journey cost partition: {row["root_index"]}/{row["label"]}')
    if row['saved_answers_reused']!=0:
        raise ValueError('Online controller consumed saved offline answers')
    return dict(cpu_residual=cpu,wall_residual=wall,exact=True)


def candidate_values(gate):
    return {arm:[row['value'] for row in gate['rows'][arm]] for arm in ('alight','continue')}


def compact_gate(gate,root_index,gate_index,matched):
    if matched:
        values=candidate_values(gate)
    else:
        values={arm:[point(row['loss']) for row in gate['rows'][arm]] for arm in ('alight','continue')}
    gaps=[difference(a,c) for a,c in zip(values['alight'],values['continue'],strict=True)]
    return dict(root_index=root_index,gate_index=gate_index,cid=gate['cid'],clock=gate['clock'],
        source=gate['source'],seed=gate['seed'],count=gate['count'],generated_count=gate['generated_count'],
        base_action=gate['base_action'],selected_action=gate['selected_action'],
        selected_changed=gate['base_action']!=gate['selected_action'],
        actions={arm:distribution_summary(rows) for arm,rows in values.items()},gap=distribution_summary(gaps),
        cost=gate['cost'],cache_stats=gate.get('cache_stats'),root_independence=gate.get('root_independence'))


def common_gate_comparison(subject,baseline,root_index):
    """Match legal public histories, never match unrelated new-state seeds."""
    matches=[]
    used=set()
    unmatched=[]
    for index,gate in enumerate(subject['gates']):
        candidates=[j for j,other in enumerate(baseline['gates']) if j not in used
            and gate['cid']==other['cid'] and gate['clock']==other['clock'] and gate['public_history']==other['public_history']]
        if not candidates:
            unmatched.append(index)
            continue
        j=candidates[0];used.add(j);other=baseline['gates'][j]
        if (gate['seed']!=other['seed'] or gate['source']!=other['source'] or gate['count']!=other['count']
            or gate['generated_count']!=other['generated_count']):
            raise ValueError(f'Common public state lost paired sample identity: {root_index}/{index}/{j}')
        own=candidate_values(gate)
        full={arm:[point(row['loss']) for row in other['rows'][arm]] for arm in ('alight','continue')}
        if any(row['world_index']!=k for arm in ('alight','continue') for k,row in enumerate(gate['rows'][arm])):
            raise ValueError('Matched candidate changed world ordering')
        own_gap=[difference(a,c) for a,c in zip(own['alight'],own['continue'],strict=True)]
        full_gap=[difference(a,c) for a,c in zip(full['alight'],full['continue'],strict=True)]
        matches.append(dict(root_index=root_index,matched_gate_index=index,full_gate_index=j,source=gate['source'],
            cid=gate['cid'],clock=gate['clock'],seed=gate['seed'],
            same_base_action=gate['base_action']==other['base_action'],
            same_selected_action=gate['selected_action']==other['selected_action'],
            matched_selected=gate['selected_action'],full_selected=other['selected_action'],
            matched_gap=distribution_summary(own_gap),full_gap=distribution_summary(full_gap),
            matched_minus_full_gap=distribution_summary([difference(a,b) for a,b in zip(own_gap,full_gap,strict=True)])))
    return dict(root_index=root_index,matched_gate_count=len(subject['gates']),full_gate_count=len(baseline['gates']),
        common=matches,unmatched_candidate_gate_indices=unmatched,
        unmatched_full_gate_indices=[j for j in range(len(baseline['gates'])) if j not in used])


def reference_summary():
    records=[]
    for index in range(148):
        row=read(HERE/'references'/f'{index:03}.json.gz')
        old=read(V43/'references'/f'{index:03}.json.gz')
        fields=('seed','count','generated_count','future_messages','public_history','clock','deadline')
        if row['index']!=index or row['root_index']!=old['root_index'] or any(row[key]!=old[key] for key in fields):
            raise ValueError(f'Frozen matched reference identity changed: {index}')
        for arm in ('alight','continue'):
            if len(row['rows'][arm])!=128:
                raise ValueError(f'Missing frozen worlds: {index}/{arm}')
            for world,(a,b) in enumerate(zip(row['rows'][arm],old['rows'][arm],strict=True)):
                if a['world_index']!=world or b['world_index']!=world or a['prefix']!=b['prefix'] or not local_value_equal(a['value'],b['updated']):
                    raise ValueError(f'Single-level evaluator changed V43 updated target: {index}/{arm}/{world}')
        values=candidate_values(row)
        records.append(dict(index=index,root_index=row['root_index'],source=row['source'],seed=row['seed'],
            actions={arm:distribution_summary(values[arm]) for arm in values},
            gap=distribution_summary([difference(a,b) for a,b in zip(values['alight'],values['continue'],strict=True)]),
            cost=row['cost'],complete_cost=row['complete_cost'],cache_stats=row['cache_stats'],root_independence=row['root_independence']))
    return dict(states=148,paired_worlds_per_state=128,all_prefix_and_value_match=True,value_absolute_tolerance=1e-8,
        reference_scope='Same single-level updated local target as V43, not exact full-policy value.',
        complete_cpu=numeric([r['complete_cost']['process_cpu'] for r in records]),
        complete_wall=numeric([r['complete_cost']['wall'] for r in records]),records=records)


def control_summary(rows):
    gate_rows=[gate for row in rows for gate in row['gate_records']]
    result=dict(quality=cohort_quality(rows),process_cpu=quantiles([r['cost']['total_process_cpu'] for r in rows]),
        wall=quantiles([r['cost']['total_wall'] for r in rows]),
        decision_cpu=quantiles([d['process_cpu'] for r in rows for d in r['decision_costs']]),
        decision_wall=quantiles([d['wall'] for r in rows for d in r['decision_costs']]),
        cold_cpu_subset=numeric([r['cost']['cold']['process_cpu'] for r in rows]),
        cold_wall_subset=numeric([r['cost']['cold']['wall'] for r in rows]),
        gate_calls=len(gate_rows),evaluated_pairs=sum(g['count'] for g in gate_rows),
        generated_pairs=sum(g['generated_count'] for g in gate_rows),gate_changes=sum(g['selected_changed'] for g in gate_rows),
        gate_new_states=sum(g['source'] is None for g in gate_rows),
        gates_with_unresolved_values=sum(g['gap']['mean'] is None for g in gate_rows),
        gate_cpu_subset=sum(g['cost']['total_process_cpu'] for g in gate_rows),
        gate_wall_subset=sum(g['cost']['total_wall'] for g in gate_rows),
        all_cost_partitions_exact=all(r['cost_partition']['exact'] for r in rows),
        saved_answers_reused=sum(r['saved_answers_reused'] for r in rows),
        budgets={measure:{str(b):decision_budget(rows,measure,b) for b in BUDGETS} for measure in ('process_cpu','wall')})
    return result


def net_compute(subject,baseline):
    resolved=all(a['outcome']['loss'] is not None and b['outcome']['loss'] is not None for a,b in zip(subject,baseline,strict=True))
    all_no_worse=resolved and all(a['outcome']['loss']<=b['outcome']['loss'] for a,b in zip(subject,baseline,strict=True))
    counts={measure:dict(subject_total=sum(row['cost'][field] for row in subject),
        baseline_total=sum(row['cost'][field] for row in baseline),
        slower_roots=sum(a['cost'][field]>b['cost'][field] for a,b in zip(subject,baseline,strict=True)))
        for measure,field in (('cpu','total_process_cpu'),('wall','total_wall'))}
    cheaper=all(row['subject_total']<row['baseline_total'] for row in counts.values())
    return dict(roots=len(subject),all_outcomes_resolved=resolved,all_root_losses_no_worse=all_no_worse,
        lower_aggregate_cpu_and_wall=cheaper,passed=all_no_worse and cheaper,costs=counts,
        scope='Empirical engineering condition: all 72 losses no worse and both aggregate CPU/wall lower; not G2 novelty.')


def online_summary():
    compact={label:[] for label in LABELS}
    compared={(label,'rolling_fixed') for label in LABELS}
    compared.update((label,'full128') for label in LABELS if label!='full128')
    compared.update(('matched128',label) for label in ('one_step_rollout','optimal'))
    pairs={a+'__vs__'+b:[] for a,b in sorted(compared)}
    cases,gate_matches=[],[]
    for index in range(72):
        roots={label:read(HERE/'roots'/f'{index:03}_{label}.json.gz') for label in LABELS}
        for label,row in roots.items():
            if row['root_index']!=index or row['root_id']!=roots['rolling_fixed']['root_id']:
                raise ValueError(f'Wrong frozen root: {index}/{label}')
            if label!='matched128':
                old=read(V43/'roots'/f'{index:03}_{label}.json.gz')
                if not meeting_science_equal(row,old):
                    raise ValueError(f'Existing comparator science changed: {index}/{label}')
            partition=validate_partition(row)
            compact[label].append(dict(root_index=index,root_id=row['root_id'],day=row['day'],origin=row['origin'],goal=row['goal'],
                loss_cap=row['loss_cap'],outcome={k:row['outcome'][k] for k in ('loss','status')},cost=row['cost'],
                decision_costs=row['decision_costs'],cost_partition=partition,saved_answers_reused=row['saved_answers_reused'],
                source=f'roots/{index:03}_{label}.json.gz',
                gate_records=[compact_gate(gate,index,i,label=='matched128') for i,gate in enumerate(row['gates'])]))
        for subject,baseline in sorted(compared):
            pairs[subject+'__vs__'+baseline].append(pair(roots[subject],roots[baseline]))
        gate_matches.append(common_gate_comparison(roots['matched128'],roots['full128'],index))
        first=roots['rolling_fixed']
        cases.append(dict(root_index=index,root_id=first['root_id'],day=first['day'],origin=first['origin'],goal=first['goal'],
            losses={label:r['outcome']['loss'] for label,r in roots.items()},statuses={label:r['outcome']['status'] for label,r in roots.items()},
            process_cpu={label:r['cost']['total_process_cpu'] for label,r in roots.items()},wall={label:r['cost']['total_wall'] for label,r in roots.items()}))
    history=read(V43/'existing_attribution.json')
    sets=dict(original_six_cases=set(OLD_CASES),original_seven_nonzero_cases=set(history['nonzero_outcome_roots']),
        original_four_changed_ties=set(history['changed_but_zero_outcome_roots']),
        historical_negative_cases={r['root_index'] for r in history['old_negative_roots']})
    matches=[match for row in gate_matches for match in row['common']]
    return dict(controls={label:control_summary(rows) for label,rows in compact.items()},
        comparisons={label:grouped_comparison(rows) for label,rows in pairs.items()},
        joint_budget_matched_vs_full=joint_budgets(compact['matched128'],compact['full128']),
        net_compute_matched_vs_full=net_compute(compact['matched128'],compact['full128']),
        root_cases=cases,root_receipts=compact,**{name:[r for r in cases if r['root_index'] in selected] for name,selected in sets.items()},
        common_gate_comparison=dict(roots=72,common_gates=len(matches),same_base_actions=sum(r['same_base_action'] for r in matches),
            same_selected_actions=sum(r['same_selected_action'] for r in matches),
            unmatched_candidate_gates=sum(len(r['unmatched_candidate_gate_indices']) for r in gate_matches),
            unmatched_full_gates=sum(len(r['unmatched_full_gate_indices']) for r in gate_matches),records=gate_matches),
        all_existing_comparators_exact=True)


def attempt_disclosure():
    from experiments.reconstruction_v44.matched import ACTION_TIE_SECONDS, choose
    attempts={}
    for name,folder in (('strict_sign',HERE/'strict_sign_attempt'),('corrected',HERE)):
        records=[]
        overrides=0
        for index in range(72):
            root=read(folder/'roots'/f'{index:03}_matched128.json.gz')
            for gate_index,gate in enumerate(root['gates']):
                low,high=gate['paired']['full_bank_mean_gain_interval']
                if (low!=0 or high!=0) and max(abs(low),abs(high))<=ACTION_TIE_SECONDS:
                    changed=gate['selected_action']!=gate['base_action']
                    overrides+=changed
                    records.append(dict(root_index=index,gate_index=gate_index,cid=gate['cid'],clock=gate['clock'],
                        gap_interval=[low,high],base_action=gate['base_action'],selected_action=gate['selected_action'],
                        recorded_override=changed))
        attempts[name]=dict(physical_controls=360,matched_roots=72,
            nonzero_subprecision_intervals=len(records),recorded_overrides_from_subprecision=overrides,records=records)
    mismatches=[]
    for index in range(148):
        row=read(HERE/'references'/f'{index:03}.json.gz')
        old=read(HERE.parent/'reconstruction_v41/banks'/f'{index:03}_shallow.json.gz')
        action=choose(row,old['base_action'])
        if action!=old['selected_action']:
            mismatches.append(dict(index=index,selected=action,full_selected=old['selected_action']))
    return dict(amendment=read(HERE/'roundoff_amendment.json'),preserved_attempt='strict_sign_attempt/',
        physical_controls_executed=720,final_comparison_controls=360,unchanged_reference_states=148,
        action_tie_seconds=ACTION_TIE_SECONDS,attempts=attempts,
        corrected_saved_reference_rank_check=dict(states=148,mismatches=mismatches,
            scope='Read-only application of corrected choose to unchanged saved values; no new worlds.'),
        scope='Every final quality/cost metric uses the complete corrected batch; original batch retained, not timing-selected.')


def outputs(result):
    fields=('configuration','mean_loss_lower','mean_loss_upper','mean_journey_cpu','mean_journey_wall',
        'decision_cpu_p50','decision_cpu_p95','decision_wall_p50','decision_wall_p95','wins','harms','ties','mean_gain_vs_full')
    with (HERE/'controls.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for label,control in result['controls'].items():
            comp=result['comparisons'].get(label+'__vs__full128')
            writer.writerow(dict(configuration=label,mean_loss_lower=control['quality']['full_cohort_mean_loss_interval'][0],
                mean_loss_upper=control['quality']['full_cohort_mean_loss_interval'][1],mean_journey_cpu=control['process_cpu']['mean'],
                mean_journey_wall=control['wall']['mean'],decision_cpu_p50=control['decision_cpu']['p50'],
                decision_cpu_p95=control['decision_cpu']['p95'],decision_wall_p50=control['decision_wall']['p50'],
                decision_wall_p95=control['decision_wall']['p95'],wins=comp['wins'] if comp else 0,
                harms=comp['harms'] if comp else 0,ties=comp['ties'] if comp else 72,
                mean_gain_vs_full=comp['full_cohort_mean_gain'] if comp else 0.))
    fields=['root_index','day','origin','goal']
    fields += [label+'_'+measure for label in LABELS for measure in ('loss','cpu','wall')]
    fields += ['matched_gain_vs_full','matched_gain_vs_rolling','matched_cpu_minus_full','matched_wall_minus_full']
    with (HERE/'roots.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in result['root_cases']:
            item={key:row[key] for key in ('root_index','day','origin','goal')}
            for label in LABELS:
                item.update({label+'_loss':row['losses'][label],label+'_cpu':row['process_cpu'][label],label+'_wall':row['wall'][label]})
            item.update(matched_gain_vs_full=None if row['losses']['matched128'] is None or row['losses']['full128'] is None else row['losses']['full128']-row['losses']['matched128'],
                matched_gain_vs_rolling=None if row['losses']['matched128'] is None or row['losses']['rolling_fixed'] is None else row['losses']['rolling_fixed']-row['losses']['matched128'],
                matched_cpu_minus_full=row['process_cpu']['matched128']-row['process_cpu']['full128'],
                matched_wall_minus_full=row['wall']['matched128']-row['wall']['full128'])
            writer.writerow(item)
    with (HERE/'budgets.csv').open('w',newline='') as handle:
        fields=('configuration','measure','budget','decisions','decisions_within','decision_fraction_within','roots','all_decisions_within_roots')
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for label,control in result['controls'].items():
            for measure,values in control['budgets'].items():
                for budget,row in values.items():writer.writerow(dict(configuration=label,measure=measure,budget=budget,**row))
    fig,axis=plt.subplots(figsize=(8,5))
    for label,control in result['controls'].items():
        low,high=control['quality']['full_cohort_mean_loss_interval']
        axis.plot([control['process_cpu']['mean']]*2,[low,high],marker='o',label=label)
    axis.set(xscale='log',xlabel='Mean whole-journey process CPU (s)',ylabel='Mean truncated loss over all 72 roots (s)')
    axis.grid(alpha=.2);axis.legend(fontsize=8);fig.tight_layout();fig.savefig(HERE/'quality_time.png',dpi=180);plt.close(fig)


def main():
    required=[HERE/'protocol.json']
    required += [HERE/'references'/f'{index:03}.json.gz' for index in range(148)]
    required += [HERE/'roots'/f'{index:03}_{label}.json.gz' for index in range(72) for label in LABELS]
    absent=[str(path.relative_to(HERE)) for path in required if not path.exists()]
    if absent:raise SystemExit(f'Awaiting {len(absent)} required V44 receipts; first missing: {absent[:5]}')
    result=dict(stage='V44',development_only=True,cohort=dict(states=148,roots=72,controls=5,physical_replays=360),
        reference=reference_summary(),**online_summary(),attempt_disclosure=attempt_disclosure(),
        limits='Single-level matched local target differs from full pi. Candidate/full differences on new states are retained. '
            'Budget tables are empirical process-CPU/wall thresholds, not physical action deadlines. '
            'Cold and gate costs are subsets of total journey cost; no saved offline answers. All six dates remain development data.')
    write(HERE/'summary.json',result);outputs(result)
    print('V44 complete: 148 matched-reference checks and 360 online controls; all cases retained.')


if __name__=='__main__':
    main()
