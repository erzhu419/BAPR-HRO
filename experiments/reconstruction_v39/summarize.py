"""Evaluate the predeclared structural retention rule from all paid receipts."""

from collections import Counter
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.reconstruction_v34.discover import read
from experiments.reconstruction_v38.summarize import stats, paired_cost
from experiments.reconstruction_v39.run import HERE, V38, VARIANTS


def structures(rows, online=False):
    planner, engine = Counter(), Counter()
    for row in rows:
        if online:
            planner.update(row['cost']['planner_stats'])
            engine.update(row['cost']['decision_stats'])
            children = row['gates']
        else:
            children = [row]
        for bank in children:
            planner.update(bank['cost']['planner_stats_total'])
            engine.update(bank['cost']['engine_stats_total'])
    queries = engine['onboard_requests']+engine['platform_requests']
    unique = engine['onboard_unique']+engine['platform_unique']
    return dict(planner=dict(planner), engine=dict(engine), decision_counts_instrumented=bool(engine),
        decision_requests=queries if engine else None,
        unique_decisions=unique if engine else None, decision_hits=queries-unique if engine else None,
        mean_key_trips=engine['key_trips']/queries if queries else None,
        mean_key_records=engine['key_record_units']/queries if queries else None)


def less_work(candidate, ordinary):
    return dict(unique_decisions=candidate['unique_decisions'] < ordinary['unique_decisions'],
        ride_compiles=candidate['planner']['ride_compiles'] < ordinary['planner']['ride_compiles'],
        platform_compiles=candidate['planner']['platform_compiles'] < ordinary['planner']['platform_compiles'],
        kernel_compiles=candidate['planner']['kernel_compiles'] < ordinary['planner']['kernel_compiles'])


def main():
    protocol, manifest = read(HERE/'protocol.json'), read(V38/'seed_manifest.json')
    banks = {v: [] for v in VARIANTS}
    state_pairs = []
    for entry in manifest:
        stem = entry['source'].replace('/', '_').removesuffix('.json.gz')
        rows = {v: read(HERE/'banks'/f'{stem}_{v}.json.gz') for v in VARIANTS}
        for variant, row in rows.items():
            banks[variant].append(row)
        state_pairs.append(dict(source=entry['source'], seed=entry['seed'], root_index=entry['state']['root_index'],
            all_exact=all(r['exact_v38'] for r in rows.values()),
            cost={v:r['complete_cost'] for v,r in rows.items()},
            structure={v:structures([r]) for v,r in rows.items()},
            complete_pairs=rows['frontier']['paired']['complete_pairs'],
            unresolved_pairs=rows['frontier']['paired']['unresolved_pairs'],
            selected_action=rows['frontier']['selected_action']))
    bank_cpu = {v:[r['complete_cost']['process_cpu'] for r in rows] for v,rows in banks.items()}
    bank_structure = {v:structures(rows) for v,rows in banks.items()}
    methods = [*VARIANTS, *(f'{impl}__{name}' for impl in ('v38','frontier') for name in protocol['cheap_methods'])]
    roots = {m:[read(HERE/'roots'/f'{i:03}_{m}.json.gz') for i in range(72)] for m in methods}
    controls = {m:dict(loss_seconds=stats([r['outcome']['loss'] for r in rows]),
        process_cpu_seconds=stats([r['cost']['total_process_cpu'] for r in rows]),
        wall_seconds=stats([r['cost']['total_wall'] for r in rows]),
        statuses=dict(Counter(r['outcome']['status'] for r in rows)),
        all_exact_original=all(r['exact_original'] for r in rows)) for m,rows in roots.items()}
    online_structure = {v:structures(roots[v], online=True) for v in VARIANTS}
    comparisons = {}
    for name in methods:
        gains = [b['outcome']['loss']-a['outcome']['loss'] for a,b in zip(roots['frontier'],roots[name])]
        comparisons[name] = dict(baseline_loss_minus_frontier=stats(gains),
            wins=sum(x>0 for x in gains), harms=sum(x<0 for x in gains), ties=sum(x==0 for x in gains),
            cost=paired_cost([r['cost']['total_process_cpu'] for r in roots['frontier']],
                             [r['cost']['total_process_cpu'] for r in roots[name]]))
    root_pairs = [dict(root_index=i, root_id=roots['frontier'][i]['root_id'], day=roots['frontier'][i]['day'],
        losses={m:roots[m][i]['outcome']['loss'] for m in methods},
        process_cpu={m:roots[m][i]['cost']['total_process_cpu'] for m in methods},
        gain_vs_combined=roots['v38__combined_rolling'][i]['outcome']['loss']-roots['frontier'][i]['outcome']['loss'],
        gates=len(roots['frontier'][i]['gates'])) for i in range(72)]
    all_exact = (all(r['all_exact'] for r in state_pairs) and
        all(c['all_exact_original'] for c in controls.values()) and
        all(r['all_banks_exact_v38'] for v in VARIANTS for r in roots[v]))
    work = dict(reference=less_work(bank_structure['frontier'],bank_structure['fullmemo']),
                online=less_work(online_structure['frontier'],online_structure['fullmemo']))
    timing_gate = dict(reference={v:sum(bank_cpu['frontier']) < sum(bank_cpu[v]) for v in ('v38','fullmemo')},
        online={v:controls['frontier']['process_cpu_seconds']['total'] < controls[v]['process_cpu_seconds']['total']
                for v in ('v38','fullmemo')})
    retained = all_exact and all(any(x.values()) for x in work.values()) and all(all(x.values()) for x in timing_gate.values())
    result = dict(stage='V39', development_only=True, scientific_samples_added=0,
        reference=dict(states=148, measurements=444, unique_existing_worlds=sum(e['count'] for e in manifest),
            all_exact=all(r['all_exact'] for r in state_pairs),
            unresolved_pairs_per_variant=sum(r['unresolved_pairs'] for r in state_pairs),
            cpu={v:stats(bank_cpu[v]) for v in VARIANTS},
            wall={v:stats([r['complete_cost']['wall'] for r in rows]) for v,rows in banks.items()},
            frontier_vs_v38=paired_cost(bank_cpu['frontier'],bank_cpu['v38']),
            frontier_vs_fullmemo=paired_cost(bank_cpu['frontier'],bank_cpu['fullmemo']),
            fullmemo_vs_v38=paired_cost(bank_cpu['fullmemo'],bank_cpu['v38']), structures=bank_structure),
        online=dict(roots=72, rollout_replays=216, cheap_replays=864, total_replays=1080,
            controls=controls, comparisons=comparisons, structures=online_structure,
            rollout_calls_per_variant={v:sum(len(r['gates']) for r in roots[v]) for v in VARIANTS},
            all_exact=all_exact, saved_answers_reused=0),
        retention=dict(rule='protocol.json: exactness plus fewer unique decisions or compilations than fullmemo, '
            'and lower full CPU than both comparators in both panels', exact=all_exact,
            work_reduction=work, full_cpu_lower=timing_gate, structural_implementation_retained=retained),
        nonzero_cases=[r for r in root_pairs if r['gain_vs_combined'] != 0],
        by_day={day:dict(count=sum(r['day']==day for r in root_pairs),
            gain_seconds=stats([r['gain_vs_combined'] for r in root_pairs if r['day']==day]))
            for day in sorted({r['day'] for r in root_pairs})},
        root_pairs=root_pairs, state_pairs=state_pairs,
        scope='All three policies execute the same full continuation. Repeated fixed worlds are not independent samples. '
              'The retention rule is neither a novelty claim nor G2 completion; logical travel clocks do not include CPU delay.')
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    decision = dict(stage_complete=True, G2='not_passed', structural_implementation_retained=retained,
        next=('Keep the exact reduction as an optimized standard rollout baseline. A distinct structural algorithm '
              'contribution is still unestablished; no additional experiments are authorized by this stage.' if retained else
              'Close this sufficient-state remedy and the current algorithm-extension route under the frozen closure rule. '
              'Preserve V38, all positive/negative cases and the negative structural finding; do not add another variant.'),
        evidence=['protocol.json','census.json','THEORY.md','summary.json','quality_cpu.png','tests.log','RESULTS.md'])
    (HERE/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    fig, axes = plt.subplots(1,2,figsize=(11,4.3))
    labels = dict(v38='V38 accelerated rollout', fullmemo='Full-history decision memo', frontier='Remaining-menu state')
    for i,name in enumerate(VARIANTS):
        row=controls[name]
        axes[0].scatter(row['process_cpu_seconds']['mean'],row['loss_seconds']['mean'],
            marker='*' if name=='frontier' else 'o',s=130,label=labels[name],color=plt.get_cmap('tab10')(i))
    for impl,marker,color in (('v38','s','gray'),('frontier','^','purple')):
        rows=[controls[f'{impl}__{name}'] for name in protocol['cheap_methods']]
        axes[0].scatter([r['process_cpu_seconds']['mean'] for r in rows],[r['loss_seconds']['mean'] for r in rows],
            marker=marker,s=35,alpha=.7,color=color,label=f'6 cheap controls ({impl})')
    axes[0].set(xscale='log',xlabel='Mean full online CPU (s, log scale)',ylabel='Mean truncated travel loss (s)')
    axes[0].legend(fontsize=7)
    high=max(max(bank_cpu['frontier']),max(bank_cpu['fullmemo']))
    axes[1].plot([0,high],[0,high],'--',color='gray',lw=1)
    axes[1].scatter(bank_cpu['fullmemo'],bank_cpu['frontier'],s=14,alpha=.65)
    axes[1].set(xlabel='Full-history memo bank CPU (s)',ylabel='Remaining-menu bank CPU (s)',title='All 148 original states')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(HERE/'quality_cpu.png',dpi=170)
    plt.close(fig)
    print(json.dumps(dict(retention=result['retention'],controls=controls,
        reference_structure=bank_structure,online_structure=online_structure),indent=2))


if __name__=='__main__':
    main()
