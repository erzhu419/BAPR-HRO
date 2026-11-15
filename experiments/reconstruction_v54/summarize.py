"""V54 complete-panel comparison with both frozen V53 references."""
from collections import Counter
from copy import deepcopy
from math import isfinite
from pathlib import Path

from experiments.reconstruction_v34.discover import read
from experiments.reconstruction_v38.run import save
from experiments.reconstruction_v53.summarize import (
    TOL, GAPS, aggregate, check_working_root_action, compatible, gap,
    joint_attribution, mean, observed_action_gaps, valid_certificate, valid_value,
    width, write_csv,
)

HERE = Path(__file__).resolve().parent


def paired_cell(query, root, budget, mode, baseline, old, new):
    """Keep interval and CPU regressions; do not pool prior query work for free."""
    return dict(query_index=query,root_index=root,budget=budget,mode=mode,baseline=baseline,
        baseline_width=width(old['value']),v54_width=width(new['value']),
        width_reduction=width(old['value'])-width(new['value']),
        baseline_cpu=old['cost']['process_cpu'],v54_cpu=new['cost']['process_cpu'],
        cpu_change=new['cost']['process_cpu']-old['cost']['process_cpu'],
        baseline_complete=old['value']['complete'],v54_complete=new['value']['complete'],
        baseline_certified=old['certified_action'],v54_certified=new['certified_action'])


def paired_aggregate(rows):
    return dict(cells=len(rows),narrower=sum(r['width_reduction']>TOL for r in rows),
        wider=sum(r['width_reduction']<-TOL for r in rows),
        tied=sum(abs(r['width_reduction'])<=TOL for r in rows),
        mean_width_reduction=mean(rows,lambda r:r['width_reduction']),
        mean_cpu_change=mean(rows,lambda r:r['cpu_change']),
        baseline_certificates=sum(r['baseline_certified'] is not None for r in rows),
        v54_certificates=sum(r['v54_certified'] is not None for r in rows),
        gained_certificate_queries=[r['query_index'] for r in rows
            if r['baseline_certified'] is None and r['v54_certified'] is not None],
        lost_certificate_queries=[r['query_index'] for r in rows
            if r['baseline_certified'] is not None and r['v54_certified'] is None])


def gap_aggregate(rows):
    return dict(states=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
        mean_lower=mean(rows,lambda r:r['lower']),mean_upper=mean(rows,lambda r:r['upper']),
        positive_queries=[r['query_index'] for r in rows if r['lower']>TOL],
        negative_queries=[r['query_index'] for r in rows if r['upper']<-TOL],
        zero_queries=[r['query_index'] for r in rows if r['status'].startswith('certified_zero')])


def require_panel(panel, protocol, previous):
    if (len(panel['rows'])!=88 or {r['query_index'] for r in panel['rows']}!=set(range(88))
            or {r['state']['root_index'] for r in panel['rows']}!=set(range(72))):
        raise ValueError('The frozen 88-state/all72-root panel changed')
    if protocol['modes']!=previous['modes'] or protocol['budgets']!=previous['budgets']:
        raise ValueError('V53 modes or original budgets changed')


def normalized(bank, source):
    result={k:bank[k] for k in ('value','actions','selected_action','cheap_action',
        'certified_action','counters','reasons','root_bounds','prefix_cost','limits','screening_workers')}
    result.update(cost=bank['complete_cost'],source=source,root_refinement=bank['root_refinement'])
    return result


def refinement_row(bank):
    """Account for continued work inside one original query, never as free retries."""
    info=bank['root_refinement'];passes=info['passes']
    prior_states=0;prior_cpu=0.
    for work in passes:
        if (work['states_start']<prior_states or work['states_end']<work['states_start']
                or work['states_end']>bank['budget']['states']):
            raise ValueError('Refinement reset or exceeded the cumulative state budget')
        if work['cpu_start']<prior_cpu-1e-8 or work['cpu_end']<work['cpu_start']-1e-8:
            raise ValueError('Refinement CPU clock reset')
        before,after=work['before'],work['after']
        if before is not None and (after[0]<before[0]-1e-7 or after[1]>before[1]+1e-7):
            raise ValueError('Continued same-query root action interval widened')
        if work['states_end']>work['state_limit']:
            raise ValueError('Root work exceeded its cumulative state allocation')
        prior_states,prior_cpu=work['states_end'],work['cpu_end']
    if prior_states>bank['counters']['states'] or prior_cpu>bank['complete_cost']['process_cpu']+1e-8:
        raise ValueError('Refinement work omitted from complete query accounting')
    if info['final_cutoff']>info['initial_cutoff']:
        raise ValueError('Refinement raised its final screening cutoff')
    first=info['first_pass_value']
    clock=bank['clock']
    if (bank['value']['lower']<first[0]-clock-1e-7 or
            bank['value']['upper']>first[1]-clock+1e-7):
        raise ValueError('Continued same-query policy interval widened')
    return dict(query_index=bank['query_index'],root_index=bank['root_index'],
        budget=bank['budget']['name'],mode=bank['mode'],passes=len(passes),
        reflow_calls=info['reflow_calls'],threshold_refinements=info['threshold_refinements'],
        initial_cutoff=info['initial_cutoff'],final_cutoff=info['final_cutoff'],
        first_pass_width=first[1]-first[0],final_width=width(bank['value']),
        width_reduction_after_first_pass=first[1]-first[0]-width(bank['value']),
        first_pass_cpu=info['first_pass_cpu'],first_pass_states=info['first_pass_states'],
        complete_cpu=bank['complete_cost']['process_cpu'],states=bank['counters']['states'],
        after_first_pass_solver_cpu=bank['cost']['process_cpu']-info['first_pass_cpu'],
        reflow_pass_cpu=sum(p['cpu_end']-p['cpu_start'] for p in passes if p['phase']=='reflow'),
        final_stop=info['final_stop'],
        unsupported_passes=sum(p['unsupported_histories']>0 for p in passes))


def matched_action_attribution(results,kind):
    """Intersect two paid bounds of each identical forced-root continuation.

    At a real onboard/platform decision, cheap P0 and one-step P0 both
    execute the forced root action then follow depth-zero cheap P0.
    Their root policy values differ, so those values cannot be intersected.
    A platform_start root has different continuation depths and is excluded.
    """
    if kind not in ('onboard','platform'):
        raise ValueError('Matched root continuations require an onboard or platform decision')
    cheap,step=results['cheap_p0'],results['one_step_p0']
    if set(cheap['actions'])!=set(step['actions']):
        raise ValueError('Matched root continuations have different action sets')
    actions={}
    for action,left in cheap['actions'].items():
        right=step['actions'][action]
        # Check only the common forced-action target; C0 and V1 themselves
        # need not have intersecting ranges.
        compatible(dict(value=left,actions={},certified_action=None),
                   dict(value=right,actions={},certified_action=None))
        lo=max(left['lower'],right['lower']);hi=min(left['upper'],right['upper'])
        complete=left['complete'] or right['complete']
        actions[action]=dict(lower=min(lo,hi),upper=hi,complete=complete,
                             mean=min(lo,hi) if complete else None)
    derived=deepcopy(results)
    cheap_action='wait' if cheap['cheap_action'] is None else str(cheap['cheap_action'])
    best=min(actions,key=lambda action:actions[action]['upper'])
    other=[v['lower'] for action,v in actions.items() if action!=best]
    label=lambda action:action if action in ('wait','alight','continue') else int(action)
    certificate=label(best) if other and actions[best]['upper']<min(other)-1e-9 else None
    for mode in ('cheap_p0','one_step_p0'):
        derived[mode]['actions']=deepcopy(actions)
        derived[mode]['certified_action']=certificate
    derived['cheap_p0']['value']=dict(actions[cheap_action])
    low=min(v['lower'] for v in actions.values());high=min(v['upper'] for v in actions.values())
    complete=all(v['complete'] for v in actions.values())
    derived['one_step_p0'].update(value=dict(lower=low,upper=high,complete=complete,
        mean=low if complete else None),selected_action=label(best))
    for mode in ('cheap_p0','one_step_p0'):
        compatible(results[mode],derived[mode])
    output=joint_attribution(derived)
    output.update(matched_actions=actions,matched_minimizer_certificate=certificate,
        scope='Supplemental attribution from the same five paid queries within one version and budget. '
              'Only cheap P0/one-step forced actions at an onboard/platform root share a target; '
              'their root values are rebuilt separately before five-policy propagation. '
              'No intervals are imported across versions, and no whole-policy equality with optimal is implied.')
    return output


def matched_supplement(versions,budgets):
    records=[];flat=[];aggregates={}
    for version,rows in versions.items():
        aggregates[version]={}
        for budget in budgets:
            level=budget['name'];cells=[]
            for row in rows:
                derived=matched_action_attribution(row['results'][level],row['kind'])
                cell=dict(version=version,budget=level,query_index=row['query_index'],
                    root_index=row['root_index'],kind=row['kind'],**derived)
                cells.append(cell);records.append(cell)
                for name,entry in derived['gaps'].items():
                    flat.append(dict(version=version,budget=level,query_index=row['query_index'],
                        root_index=row['root_index'],kind=row['kind'],comparison=name,
                        **{k:v for k,v in entry.items() if k!='certificate_modes'},
                        certificate_modes=','.join(entry['certificate_modes']),
                        joint_process_cpu=derived['cost']['process_cpu'],charged_queries=derived['cost']['complete_queries']))
            gaps={name:gap_aggregate([dict(query_index=r['query_index'],**r['gaps'][name]) for r in cells])
                  for name,*_ in GAPS}
            for name in ('cheap_p0_minus_optimal','cheap_working_minus_optimal'):
                gaps[name]['upper_bound_at_most_seconds']={str(limit):dict(
                    states=sum(r['gaps'][name]['upper']<=limit+TOL for r in cells),
                    query_indices=[r['query_index'] for r in cells if r['gaps'][name]['upper']<=limit+TOL])
                    for limit in (1,10,60,300)}
            aggregates[version][level]=dict(states=len(cells),queries_per_state=5,
                mean_cpu_per_joint_state=mean(cells,lambda r:r['cost']['process_cpu']),
                total_charged_cpu=sum(r['cost']['process_cpu'] for r in cells),
                certified_cheap_one_step_equalities=sum(bool(r['equalities']) for r in cells),gaps=gaps)
    return dict(aggregate=aggregates,records=records,flat=flat)


def summarize():
    protocol=read(HERE/'contract.json')
    panel=read(HERE.parent/'reconstruction_v52/panel.json.gz')
    baselines={name:read(HERE.parent/'reconstruction_v53'/filename) for name,filename in
               (('impact','summary.json'),('compressed','summary_compressed.json'))}
    require_panel(panel,protocol,baselines['impact'])
    modes,budgets=protocol['modes'],protocol['budgets']
    paths=[HERE/'natural'/f"{r['query_index']:03}_{b['name']}_{m}.json.gz"
           for r in panel['rows'] for b in budgets for m in modes]
    missing=[p for p in paths if not p.exists()]
    if missing:
        raise ValueError(f'Complete panel required: {len(missing)} missing receipts; first={missing[0].name}')
    previous={name:{r['query_index']:r for r in data['records']} for name,data in baselines.items()}
    records,flat,pairs,gaps,choices,joints,budget_pairs,refinements,refinement_rows=[],[],[],[],[],[],[],[],[]
    audit=Counter()
    for item in panel['rows']:
        query,state=item['query_index'],item['state'];root=state['root_index']
        record={k:previous['impact'][query][k] for k in ('query_index','source_state_index',
            'root_index','day','kind','clock','selection_reasons','observed_baseline_actions','source','actual_v51')}
        record.update(results={},gaps={},observed_v51_action_gaps={},joint_attribution={})
        for b in budgets:
            level=b['name'];results=record['results'][level]={}
            for mode in modes:
                source=f'natural/{query:03}_{level}_{mode}.json.gz';bank=read(HERE/source)
                if (bank['query_index'],bank['source_state_index'],bank['root_index'],bank['day'],bank['mode'])!=(
                        query,item['source_state_index'],root,state['day'],mode):
                    raise ValueError('Receipt differs from the frozen state or policy')
                if bank['budget']!=b or bank['counters']['states']>b['states']:
                    raise ValueError('Receipt violates the original state budget')
                if bank['implementation']!='refining_conditional':
                    raise ValueError('Receipt uses the wrong implementation')
                if (bank['clock'],bank['deadline'],bank['future_public_frames'])!=(state['clock'],state['deadline'],0):
                    raise ValueError('Physical clock or information contract changed')
                for value in (bank['value'],*bank['actions'].values()):
                    valid_value(value,state['deadline']-state['clock'])
                valid_certificate(bank)
                if mode=='cheap_working':
                    audit['original_working_root_action_checks']+=check_working_root_action(item['occurrences'],bank)
                if any(not isfinite(bank['complete_cost'][k]) or bank['complete_cost'][k]<0 for k in ('process_cpu','wall')):
                    raise ValueError('Invalid charged query cost')
                result=results[mode]=normalized(bank,source)
                refinement_rows.append(refinement_row(bank))
                for name,old in previous.items():
                    prior=old[query]['results'][level][mode]
                    audit.update(compatible(prior,result));audit['cross_version_cells']+=1
                    pairs.append(paired_cell(query,root,level,mode,name,prior,result))
                flat.append(dict(query_index=query,root_index=root,budget=level,mode=mode,
                    lower=bank['value']['lower'],upper=bank['value']['upper'],width=width(bank['value']),
                    complete=bank['value']['complete'],selected_action=bank['selected_action'],
                    certified_action=bank['certified_action'],states=bank['counters']['states'],
                    cpu=bank['complete_cost']['process_cpu'],wall=bank['complete_cost']['wall'],
                    cpu_threshold=b['process_cpu_seconds'],
                    cpu_overshoot=max(0.,bank['complete_cost']['process_cpu']-b['process_cpu_seconds']),source=source))
                refinements.append(dict(query_index=query,root_index=root,budget=level,mode=mode,
                    source=source,root_refinement=bank['root_refinement']))
            record['gaps'][level]={name:gap(results[left]['value'],results[right]['value'],positive)
                                  for name,left,right,positive in GAPS}
            for name,entry in record['gaps'][level].items():
                gaps.append(dict(query_index=query,root_index=root,budget=level,comparison=name,**entry))
            actual=record['observed_v51_action_gaps'][level]=observed_action_gaps(item['occurrences'],results['optimal'])
            for choice in actual:
                choices.append(dict(query_index=query,root_index=root,budget=level,
                    **{k:v for k,v in choice.items() if k not in ('action_value','optimal_value')},
                    action_lower=choice['action_value']['lower'],action_upper=choice['action_value']['upper'],
                    optimal_lower=choice['optimal_value']['lower'],optimal_upper=choice['optimal_value']['upper']))
            joint=record['joint_attribution'][level]=joint_attribution(results)
            for name,entry in joint['gaps'].items():
                joints.append(dict(query_index=query,root_index=root,budget=level,comparison=name,
                    **{k:v for k,v in entry.items() if k!='certificate_modes'},
                    certificate_modes=','.join(entry['certificate_modes']),
                    joint_process_cpu=joint['cost']['process_cpu'],charged_queries=joint['cost']['complete_queries']))
        for mode in modes:
            a,z=(record['results'][b['name']][mode] for b in budgets)
            audit.update(compatible(a,z));audit['cross_budget_cells']+=1
            budget_pairs.append(dict(query_index=query,root_index=root,mode=mode,
                b2_width=width(a['value']),b10_width=width(z['value']),
                width_reduction=width(a['value'])-width(z['value']),
                b2_cpu=a['cost']['process_cpu'],b10_cpu=z['cost']['process_cpu'],
                b2_certified=a['certified_action'],b10_certified=z['certified_action']))
        records.append(record)
    aggregates={b['name']:{m:aggregate([r['results'][b['name']][m] for r in records],b)
                          for m in modes} for b in budgets}
    comparisons={name:{b['name']:{m:paired_aggregate([r for r in pairs
        if (r['baseline'],r['budget'],r['mode'])==(name,b['name'],m)]) for m in modes}
        for b in budgets} for name in baselines}
    gap_summary={b['name']:{name:gap_aggregate([r for r in gaps if
        (r['budget'],r['comparison'])==(b['name'],name)]) for name,*_ in GAPS} for b in budgets}
    joint_summary={};observed_summary={};alignment={}
    for b in budgets:
        level=b['name']
        joint_summary[level]=dict(states=88,queries_per_state=len(modes),
            mean_cpu_per_joint_state=mean(records,lambda r:r['joint_attribution'][level]['cost']['process_cpu']),
            total_charged_cpu=sum(r['joint_attribution'][level]['cost']['process_cpu'] for r in records),
            certified_cheap_one_step_equalities=sum(bool(r['joint_attribution'][level]['equalities']) for r in records),
            gaps={name:gap_aggregate([r for r in joints if (r['budget'],r['comparison'])==(level,name)])
                  for name,*_ in GAPS})
        observed_summary[level]={}
        for policy in baselines['impact']['observed_v51_action_gaps']['aggregate'][level]:
            rows=[r for r in choices if (r['budget'],r['policy'])==(level,policy)]
            observed_summary[level][policy]=dict(occurrences=len(rows),queries=len({r['query_index'] for r in rows}),
                statuses=dict(Counter(r['status'] for r in rows)),
                certified_positive_queries=sorted({r['query_index'] for r in rows if r['lower']>TOL}),
                certified_positive_roots=sorted({r['root_index'] for r in rows if r['lower']>TOL}),
                certified_zero_queries=sorted({r['query_index'] for r in rows if r['upper']<=TOL}),
                max_proven_loss_lower=max((r['lower'] for r in rows),default=None))
        certified=[r for r in records if r['results'][level]['optimal']['certified_action'] is not None]
        alignment[level]={}
        for mode in ('cheap_p0','cheap_working'):
            same=[r['query_index'] for r in certified if str(r['results'][level]['optimal']['certified_action'])
                  ==str(r['results'][level][mode]['cheap_action'])]
            alignment[level][mode]=dict(optimal_certified_queries=len(certified),same_root_choice=len(same),
                different_root_choice=len(certified)-len(same),same_query_indices=same)
    supplemental=matched_supplement(dict(v54=records,
        v53_impact=baselines['impact']['records'],v53_compressed=baselines['compressed']['records']),budgets)
    result=dict(stage='V54',implementation='refining',queries=len(records),calls=len(flat),modes=modes,budgets=budgets,
        original_root_denominator=72,represented_roots=72,missing_roots=panel['missing_first_nontrivial_roots'],
        all_v51_state_count=panel['all_state_count'],all_same_history_disagreements=panel['same_history_disagreement_states'],
        aggregate=aggregates,gap_summary=gap_summary,records=records,all_root_context=baselines['impact']['all_root_context'],
        observed_v51_action_gaps=dict(aggregate=observed_summary,records='observed_action_gaps.csv',
            scope=baselines['impact']['observed_v51_action_gaps']['scope']),
        joint_attribution=dict(aggregate=joint_summary,records='joint_attribution.csv',
            scope=baselines['impact']['joint_attribution']['scope']),
        matched_action_attribution=dict(aggregate=supplemental['aggregate'],
            records='matched_action_attribution.json',flat_records='matched_action_attribution.csv',
            scope='Supplemental common forced-action intersections, then five-policy propagation, separately '
                  'for V54 and both V53 implementations. Each state/budget charges its own five complete queries; '
                  'raw and original joint results are unchanged. Only onboard/platform roots are admissible.'),
        optimal_certificate_root_choice_alignment=dict(aggregate=alignment,
            scope=baselines['impact']['optimal_certificate_root_choice_alignment']['scope']),
        audit=dict(audit,all_value_and_action_ranges_intersect=True,conflicting_certificates=0,
            state_caps_respected=True,cpu_is_atomic_operation_threshold=True),
        same_budget_comparison=dict(primary_baseline='V53 impact',aggregate=comparisons,records='same_budget_paired.csv',
            scope='Each of both original V53 budgets is compared separately on all 88 states and five modes. No prior query ranges are imported into the new solver without charge.'),
        budget_comparison=dict(records='budget_paired.csv',scope='Two original V53 budgets; no monotonicity assumption.'),
        root_refinement=dict(records='root_refinement.json',flat_records='refinement.csv',
            aggregate={b['name']:{m:dict(queries=88,
                reflowed_queries=sum(r['reflow_calls']>0 for r in refinement_rows if (r['budget'],r['mode'])==(b['name'],m)),
                refined_queries=sum(r['threshold_refinements']>0 for r in refinement_rows if (r['budget'],r['mode'])==(b['name'],m)),
                final_stops=dict(Counter(r['final_stop'] for r in refinement_rows if (r['budget'],r['mode'])==(b['name'],m))),
                mean_width_reduction_after_first_pass=mean([r for r in refinement_rows if (r['budget'],r['mode'])==(b['name'],m)],
                    lambda r:r['width_reduction_after_first_pass']),
                mean_after_first_pass_solver_cpu=mean([r for r in refinement_rows if (r['budget'],r['mode'])==(b['name'],m)],
                    lambda r:r['after_first_pass_solver_cpu']),
                mean_reflow_pass_cpu=mean([r for r in refinement_rows if (r['budget'],r['mode'])==(b['name'],m)],lambda r:r['reflow_pass_cpu']))
                for m in modes} for b in budgets},
            scope='Per-query work and stopping history; cumulative counters are not an additive decomposition of root value uncertainty. First-pass comparisons are checkpoints within the same charged query, not separately timed V53 runs.'),
        gap_scope=baselines['impact']['gap_scope'],g2_passed=False,new_method_contribution_established=False,
        ordinary_method_intrinsic_barrier_established=False,whole_journey_optimality_claim=False,server_transfer_bytes=0)
    save(HERE/'summary.json',result);save(HERE/'root_refinement.json',refinements)
    save(HERE/'matched_action_attribution.json',supplemental['records'])
    for name,rows in [('paired',flat),('same_budget_paired',pairs),('budget_paired',budget_pairs),
                      ('gaps',gaps),('observed_action_gaps',choices),('joint_attribution',joints),('refinement',refinement_rows),
                      ('matched_action_attribution',supplemental['flat'])]:
        write_csv(HERE/f'{name}.csv',rows)
    plot(pairs,modes,budgets)
    print(dict(queries=len(records),calls=len(flat),audit=result['audit'],
        optimal={b['name']:{k:aggregates[b['name']]['optimal'][k]
            for k in ('mean_width','mean_cpu','certified_actions','complete')} for b in budgets}),flush=True)
    return result


def plot(rows,modes,budgets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,b in zip(axes,budgets):
        for mode,color in zip(modes,colors):
            cells=[r for r in rows if (r['baseline'],r['budget'],r['mode'])==('impact',b['name'],mode)]
            x=[mean(cells,lambda r:r[k]) for k in ('baseline_cpu','v54_cpu')]
            y=[mean(cells,lambda r:r[k]) for k in ('baseline_width','v54_width')]
            ax.plot(x,y,color=color,label=mode,linewidth=1)
            ax.scatter(x[0],y[0],color=color,marker='x',s=45)
            ax.scatter(x[1],y[1],color=color,marker='o',s=35)
        ax.set(xlabel='Mean complete query process CPU (s)',ylabel='Mean conditional interval width (s)',
            title=f"{b['name']}: V53 impact x to V54 o; all 88 states")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/'quality_time.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,b in zip(axes,budgets):
        cells=[r for r in rows if (r['baseline'],r['budget'])==('impact',b['name'])]
        for mode,color in zip(modes,colors):
            selected=[r for r in cells if r['mode']==mode]
            ax.scatter([r['baseline_width'] for r in selected],[r['v54_width'] for r in selected],
                color=color,s=15,alpha=.5,label=mode)
        top=max(max(r['baseline_width'],r['v54_width']) for r in cells)*1.03
        ax.plot([0,top],[0,top],'--',color='gray',linewidth=1)
        ax.set(xlabel='V53 impact interval width (s)',ylabel='V54 interval width (s)',
            title=f"{b['name']}: all 440 paired cells; below line is narrower")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/'paired_widths.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    summarize()
