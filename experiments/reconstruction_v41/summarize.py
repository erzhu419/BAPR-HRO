"""Full-cohort V41 accounting and mechanism comparisons from saved receipts."""

from collections import Counter
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reconstruction_v34.summarize import first_divergence, numeric, read, write
from experiments.reconstruction_v40.summarize import cohort_quality, loss_bounds


HERE = Path(__file__).resolve().parent
ARMS = ('legacy', 'physical', 'shallow', 'physical_meeting', 'shallow_meeting')
OLD_CASES = (16, 25, 28, 57, 63, 64)


def status_counts(rows):
    return dict(Counter(row['status'].split(':', 1)[0] for row in rows))


def action_value(rows, cap):
    """Full-bank bounds plus explicitly resolved-only Monte Carlo precision."""
    values = [row['loss'] for row in rows if row['loss'] is not None]
    count, known = len(rows), len(values)
    bounds = [sum(values)/count, (sum(values)+(count-known)*cap)/count]
    return dict(worlds=count, resolved=known, unresolved=count-known,
        full_bank_mean= bounds[0] if known == count else None, full_bank_mean_interval=bounds,
        resolved_sample=dict(n=known, mean=float(np.mean(values)) if known else None,
            se=float(np.std(values, ddof=1)/np.sqrt(known)) if known > 1 else None),
        statuses=status_counts(rows),
        scope='Full-bank empirical missing bounds; SE describes resolved draws only, not missing mass.')


def pair(subject, baseline):
    alo, ahi = loss_bounds(subject)
    blo, bhi = loss_bounds(baseline)
    a, b = subject['outcome']['loss'], baseline['outcome']['loss']
    return dict(root_index=subject['root_index'], day=subject['day'],
        od=subject['origin']+' -> '+subject['goal'], subject_loss=a, baseline_loss=b,
        gain=None if a is None or b is None else b-a, gain_interval=[blo-ahi, bhi-alo],
        subject_cpu=subject['cost']['total_process_cpu'], baseline_cpu=baseline['cost']['total_process_cpu'],
        subject_wall=subject['cost']['total_wall'], baseline_wall=baseline['cost']['total_wall'],
        **first_divergence(subject['outcome'], baseline['outcome']))


def pair_summary(rows):
    gains = [row['gain'] for row in rows if row['gain'] is not None]
    count = len(rows)
    bounds = [sum(row['gain_interval'][j] for row in rows)/count for j in (0, 1)]
    return dict(denominator=count, resolved_pairs=len(gains), unresolved_pairs=count-len(gains),
        wins=sum(g > 0 for g in gains), harms=sum(g < 0 for g in gains), ties=sum(g == 0 for g in gains),
        full_cohort_mean_gain=bounds[0] if count == len(gains) else None,
        full_cohort_mean_gain_interval=bounds,
        resolved_net_gain=sum(gains), resolved_positive_gain=sum(max(g, 0) for g in gains),
        resolved_harm=sum(max(-g, 0) for g in gains),
        subject_minus_baseline_cpu=numeric([r['subject_cpu']-r['baseline_cpu'] for r in rows]),
        subject_minus_baseline_wall=numeric([r['subject_wall']-r['baseline_wall'] for r in rows]),
        aggregate_cpu_ratio=sum(r['subject_cpu'] for r in rows)/sum(r['baseline_cpu'] for r in rows),
        actions_identical=sum(r['actions_identical'] for r in rows),
        common_prefix_history_mismatches=sum(not r['common_prefix_history_equal'] for r in rows))


def grouped_comparison(rows):
    result = pair_summary(rows)
    net = result['resolved_net_gain'] if result['unresolved_pairs'] == 0 else None
    result['concentration'] = {}
    for key in ('day', 'od'):
        groups = {}
        for label in sorted({row[key] for row in rows}):
            subset = [row for row in rows if row[key] == label]
            item = pair_summary(subset)
            item['share_of_full_cohort_net_gain'] = item['resolved_net_gain']/net if net else None
            groups[label] = item
        result['concentration'][key] = groups
    result['nonzero_or_unresolved_roots'] = [row['root_index'] for row in rows if row['gain'] != 0]
    result['rows'] = rows
    return result


def phase_totals(gates):
    phases = {name: 0. for name in ('conditioning', 'generation', 'planner_setup', 'alight_continuation',
                                   'continue_continuation', 'meeting_check')}
    for gate in gates:
        cost = gate['cost']
        for name in ('conditioning', 'generation', 'planner_setup'):
            phases[name] += cost[name]['process_cpu']
        for name in ('alight', 'continue'):
            phases[name+'_continuation'] += cost['continuations'][name]['process_cpu']
        phases['meeting_check'] += gate['meeting_stats']['check_cpu']
    return phases


def compact_root(row, source):
    gates, cost = row['gates'], row['cost']
    partition = cost['decision_cpu_sum']+cost['final_execution_cpu']-cost['total_process_cpu']
    return dict(root_index=row['root_index'], root_id=row['root_id'], day=row['day'],
        origin=row['origin'], goal=row['goal'], loss_cap=row['loss_cap'], source=source,
        outcome={k: row['outcome'][k] for k in ('loss', 'status')},
        cost={k: cost[k] for k in ('total_process_cpu', 'total_wall', 'cold', 'decision_cpu_sum', 'final_execution_cpu')},
        cost_partition_residual=partition, cost_partition_exact=bool(abs(partition) < 1e-9),
        decisions=len(row['decision_costs']), publications=len(row['publications']),
        publications_consumed=sum(p['consumed'] is not None for p in row['publications']),
        publication_cpu_subset=sum(p['cost']['request_to_finish_process_cpu'] for p in row['publications']),
        gate_calls=len(gates), evaluated_pairs=sum(g['count'] for g in gates),
        generated_pairs=sum(g['generated_count'] for g in gates),
        gate_changes=sum(g['selected_action'] != g['base_action'] for g in gates),
        gate_new_states=sum(g['source'] is None for g in gates),
        gate_unresolved_pairs=sum(g['paired']['unresolved_pairs'] for g in gates),
        gate_statuses=dict(Counter(status for g in gates for values in g['rows'].values()
                                  for status in (r['status'].split(':', 1)[0] for r in values))),
        gate_cpu_subset=sum(g['cost']['total_process_cpu'] for g in gates),
        gate_phases_cpu=phase_totals(gates),
        meeting_stats={name:sum(g['meeting_stats'][name] for g in gates)
                       for name in ('checks', 'hits', 'avoided_decisions', 'check_cpu')},
        legacy_physical_exact=row.get('legacy_physical_exact'),
        saved_answers_reused=row['saved_answers_reused'])


def control_summary(rows):
    result = dict(quality=cohort_quality(rows),
        process_cpu=numeric([r['cost']['total_process_cpu'] for r in rows]),
        wall=numeric([r['cost']['total_wall'] for r in rows]),
        cold_cpu_subset=numeric([r['cost']['cold']['process_cpu'] for r in rows]),
        decision_cpu=numeric([r['cost']['decision_cpu_sum'] for r in rows]),
        final_execution_cpu=numeric([r['cost']['final_execution_cpu'] for r in rows]),
        all_cost_partitions_exact=all(r['cost_partition_exact'] for r in rows),
        max_partition_residual=max(abs(r['cost_partition_residual']) for r in rows),
        gates={name:sum(r[name] for r in rows) for name in ('gate_calls', 'evaluated_pairs', 'generated_pairs',
            'gate_changes', 'gate_new_states', 'gate_unresolved_pairs', 'gate_cpu_subset')},
        gate_phases_cpu={name:sum(r['gate_phases_cpu'][name] for r in rows) for name in rows[0]['gate_phases_cpu']},
        meeting_stats={name:sum(r['meeting_stats'][name] for r in rows) for name in rows[0]['meeting_stats']},
        publication_cpu_subset=sum(r['publication_cpu_subset'] for r in rows),
        publication_count=sum(r['publications'] for r in rows),
        all_publications_consumed=all(r['publications'] == r['publications_consumed'] for r in rows),
        legacy_physical_exact=all(r['legacy_physical_exact'] is True for r in rows)
            if rows[0]['legacy_physical_exact'] is not None else None,
        saved_answers_reused=sum(r['saved_answers_reused'] for r in rows))
    result['by_day'] = {day:dict(quality=cohort_quality([r for r in rows if r['day'] == day]),
        process_cpu=numeric([r['cost']['total_process_cpu'] for r in rows if r['day'] == day]))
        for day in sorted({r['day'] for r in rows})}
    return result


def reference_summary(validation):
    states, costs = [], {arm: [] for arm in ARMS}
    all_status = {arm:Counter() for arm in ARMS}
    unresolved = Counter()
    for index, check in enumerate(validation['states']):
        banks = {arm:read(HERE/'banks'/f'{index:03}_{arm}.json.gz') for arm in ARMS}
        tail = read(HERE/'tails'/f'{index:03}.json')
        state = banks['physical']['state']
        cap = state['deadline']-state['clock']
        summaries = {}
        for arm, bank in banks.items():
            if bank['index'] != index or bank['count'] != 128 or bank['generated_count'] != 128:
                raise ValueError(f'Unexpected frozen reference receipt {index}/{arm}')
            values = {action:action_value(rows, cap) for action, rows in bank['rows'].items()}
            lo, hi = bank['paired']['full_bank_mean_gain_interval']
            gain = ([0.,0.] if bank['selected_action'] == bank['base_action'] else
                    [-hi,-lo] if bank['selected_action'] == 'alight' else [lo,hi])
            summaries[arm] = dict(base=bank['base_action'], selected=bank['selected_action'],
                actions=values, paired=bank['paired'], complete_cost=bank['complete_cost'],
                in_bank_selected_vs_base_gain_interval=gain,
                evaluation_cost={k:bank['cost'][k] for k in ('total_process_cpu', 'total_wall')},
                phases_cpu=phase_totals([bank]), meeting_stats=bank['meeting_stats'])
            costs[arm].append(bank['complete_cost'])
            unresolved[arm] += bank['paired']['unresolved_pairs']
            for value in values.values():
                all_status[arm].update(value['statuses'])
        alight = summaries['physical']['actions']['alight']
        low, high = alight['full_bank_mean_interval']
        errors = {name:dict(value=tail[name],
            minus_full_physical_alight_mean=None if alight['full_bank_mean'] is None else tail[name]-alight['full_bank_mean'],
            minus_full_physical_alight_interval=[tail[name]-high, tail[name]-low])
            for name in ('first', 'envelope', 'frozen_combined')}
        states.append(dict(index=index, source=tail['source'], root_index=tail['root_index'],
            day=state['day'], clock=state['clock'], cid=state['cid'], seed=banks['physical']['seed'],
            checks=check, arms=summaries, tail=tail, tail_errors=errors))
    meeting = {}
    for family in ('physical', 'shallow'):
        comparison = [(s['arms'][family+'_meeting']['complete_cost']['process_cpu'],
                       s['arms'][family]['complete_cost']['process_cpu']) for s in states]
        meeting[family] = dict(all_science_equal=all(s['checks'][family+'_meeting_exact'] for s in states),
            process_cpu_difference=numeric([a-b for a,b in comparison]),
            total_cpu_ratio=sum(a for a,_ in comparison)/sum(b for _,b in comparison),
            checks=sum(s['arms'][family+'_meeting']['meeting_stats']['checks'] for s in states),
            hits=sum(s['arms'][family+'_meeting']['meeting_stats']['hits'] for s in states),
            avoided_decisions=sum(s['arms'][family+'_meeting']['meeting_stats']['avoided_decisions'] for s in states))
    return dict(states=len(states), banks=len(states)*len(ARMS), paired_worlds_per_bank=128,
        legacy_worlds_all_exact=all(s['checks']['legacy_worlds_exact'] for s in states),
        unresolved_pairs_by_arm=dict(unresolved), outcome_statuses_by_arm={a:dict(c) for a,c in all_status.items()},
        support_failures_by_arm={a:dict(projection_incompatible=c['projection_incompatible'],
            unsupported=sum(n for status,n in c.items() if status.startswith('unsupported')))
            for a,c in all_status.items()},
        in_bank_improvement_by_arm={a:dict(changed_states=sum(s['arms'][a]['selected'] != s['arms'][a]['base'] for s in states),
            full_state_mean_gain_interval=[sum(s['arms'][a]['in_bank_selected_vs_base_gain_interval'][j]
                for s in states)/len(states) for j in (0,1)],
            scope='Descriptive gain on the same bank used for selection; not independent policy validation.') for a in ARMS},
        process_cpu_by_arm={a:numeric([r['process_cpu'] for r in c]) for a,c in costs.items()},
        wall_by_arm={a:numeric([r['wall'] for r in c]) for a,c in costs.items()},
        ordinary_meeting=meeting,
        changes={name:dict(base_action_changes=sum(s['arms'][a]['base'] != s['arms'][b]['base'] for s in states),
            selected_action_changes=sum(s['arms'][a]['selected'] != s['arms'][b]['selected'] for s in states))
            for name,a,b in (('model_physical_vs_legacy','physical','legacy'),
                             ('tail_shallow_vs_physical','shallow','physical'))},
        tail_error_summary={name:dict(denominator=len(states),
            resolved_mean_error=numeric([s['tail_errors'][name]['minus_full_physical_alight_mean'] for s in states
                if s['tail_errors'][name]['minus_full_physical_alight_mean'] is not None]),
            resolved_absolute_error=numeric([abs(s['tail_errors'][name]['minus_full_physical_alight_mean']) for s in states
                if s['tail_errors'][name]['minus_full_physical_alight_mean'] is not None]),
            full_state_mean_error_interval=[sum(s['tail_errors'][name]['minus_full_physical_alight_interval'][j]
                for s in states)/len(states) for j in (0,1)]) for name in ('first','envelope','frozen_combined')},
        records=states)


def meeting_science_equal(subject, baseline):
    def outcome(row):
        result = row['outcome']
        return dict(loss=result['loss'], status=result['status'], decisions=result['decisions'],
            trace=[{k:v for k,v in event.items() if k != 'receipt'} for event in result['trace']])
    def gates(row):
        return [dict(seed=g['seed'], count=g['count'], paired=g['paired'],
            rows={action:[{k:r[k] for k in ('loss','status','trace','decisions','world_index')}
                         for r in records] for action,records in g['rows'].items()}) for g in row['gates']]
    return outcome(subject) == outcome(baseline) and gates(subject) == gates(baseline)


def draw(controls):
    figure, axis = plt.subplots(figsize=(10, 5.7))
    colors = dict(legacy='#777777', physical='#1f77b4', shallow='#d55e00')
    for label, control in controls.items():
        family, mode = label.split('_', 1)
        cpu = control['process_cpu']['mean']
        low, high = control['quality']['full_cohort_mean_loss_interval']
        marker = 'o' if mode == 'cheap' else 'D' if mode.startswith('meeting') else 's' if mode.endswith('16') else '^'
        if low == high:
            axis.plot(cpu, low, marker=marker, color=colors[family], linestyle='none', label=label, markersize=7)
        else:
            axis.plot([cpu,cpu], [low,high], color=colors[family], label=label)
    axis.set(xscale='log', xlabel='Mean whole-journey process CPU (s)', ylabel='Full 72-root mean truncated loss (s)')
    axis.grid(alpha=.2)
    axis.legend(fontsize=8, loc='best', ncol=2)
    figure.text(.5,.015,'Fixed controls; all missing loss remains in full-cohort intervals. '
        'Ideal logical event time; no hard real-time claim.',ha='center',fontsize=8)
    figure.tight_layout(rect=(0,.04,1,1))
    figure.savefig(HERE/'quality_cpu.png',dpi=180)
    plt.close(figure)


def write_controls(controls, comparisons):
    fields = ('configuration','baseline','mean_journey_cpu','mean_journey_wall','mean_cold_cpu_subset',
        'mean_loss_lower','mean_loss_upper','resolved_roots','wins','harms','ties','unresolved_pairs',
        'mean_gain','mean_gain_lower','mean_gain_upper','gate_calls','evaluated_pairs','generated_pairs',
        'conditional_unresolved_pairs','all_cost_partitions_exact')
    with (HERE/'controls.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for label, control in controls.items():
            baseline=label.split('_',1)[0]+'_cheap'
            comp=comparisons[label+'__vs__'+baseline]
            writer.writerow(dict(configuration=label,baseline=baseline,
                mean_journey_cpu=control['process_cpu']['mean'],mean_journey_wall=control['wall']['mean'],
                mean_cold_cpu_subset=control['cold_cpu_subset']['mean'],
                mean_loss_lower=control['quality']['full_cohort_mean_loss_interval'][0],
                mean_loss_upper=control['quality']['full_cohort_mean_loss_interval'][1],
                resolved_roots=control['quality']['resolved'],wins=comp['wins'],harms=comp['harms'],ties=comp['ties'],
                unresolved_pairs=comp['unresolved_pairs'],mean_gain=comp['full_cohort_mean_gain'],
                mean_gain_lower=comp['full_cohort_mean_gain_interval'][0],mean_gain_upper=comp['full_cohort_mean_gain_interval'][1],
                gate_calls=control['gates']['gate_calls'],evaluated_pairs=control['gates']['evaluated_pairs'],
                generated_pairs=control['gates']['generated_pairs'],
                conditional_unresolved_pairs=control['gates']['gate_unresolved_pairs'],
                all_cost_partitions_exact=control['all_cost_partitions_exact']))


def main():
    protocol=read(HERE/'protocol.json')
    labels=protocol['online_configurations']
    required=[HERE/'reference_validation.json',HERE/'banks_preparation.json',HERE/'online_preparation.json',
        HERE/'publication_receipts.json']
    required += [HERE/'banks'/f'{index:03}_{arm}.json.gz' for index in range(148) for arm in ARMS]
    required += [HERE/'tails'/f'{index:03}.json' for index in range(148)]
    required += [HERE/'roots'/f'{index:03}_{label}.json.gz' for index in range(72) for label in labels]
    absent=[str(path.relative_to(HERE)) for path in required if not path.exists()]
    if absent:
        raise SystemExit(f'Awaiting {len(absent)} declared receipts; first missing: {absent[:5]}')
    validation=read(HERE/'reference_validation.json')
    if validation['completed'] != 148 or len(validation['states']) != 148:
        raise ValueError('All 148 reference states are required')
    compared={(label,label.split('_',1)[0]+'_cheap') for label in labels}
    compared.update((label,'physical_cheap') for label in labels)
    compared.update((('physical_cheap','legacy_cheap'),('physical_rollout128','legacy_rollout128'),('shallow_rollout16','physical_rollout16'),
        ('shallow_rollout128','physical_rollout128'),('physical_meeting128','physical_rollout128'),
        ('shallow_meeting128','shallow_rollout128')))
    pairs={a+'__vs__'+b:[] for a,b in sorted(compared)}
    compact={label:[] for label in labels}
    cases, meeting, graph_preparation=[],[],[]
    for index in range(72):
        roots={label:read(HERE/'roots'/f'{index:03}_{label}.json.gz') for label in labels}
        if any(r['root_index'] != index for r in roots.values()) or len({r['root_id'] for r in roots.values()}) != 1:
            raise ValueError(f'Mismatched frozen root {index}')
        first=roots[labels[0]]
        graph_preparation.append(dict(root_index=index,**first['preparation']))
        for label,row in roots.items():
            compact[label].append(compact_root(row,f'roots/{index:03}_{label}.json.gz'))
        for subject,baseline in sorted(compared):
            pairs[subject+'__vs__'+baseline].append(pair(roots[subject],roots[baseline]))
        meeting.append(dict(root_index=index,**{family:meeting_science_equal(
            roots[family+'_meeting128'],roots[family+'_rollout128']) for family in ('physical','shallow')}))
        cases.append(dict(root_index=index,root_id=first['root_id'],day=first['day'],
            origin=first['origin'],goal=first['goal'],loss_cap=first['loss_cap'],
            loss={label:r['outcome']['loss'] for label,r in roots.items()},
            loss_interval={label:loss_bounds(r) for label,r in roots.items()},
            cpu={label:r['cost']['total_process_cpu'] for label,r in roots.items()}))
    controls={label:control_summary(rows) for label,rows in compact.items()}
    comparisons={name:grouped_comparison(rows) for name,rows in pairs.items()}
    result=dict(stage='V41',protocol='protocol.json',development_only=True,
        model_revision=protocol.get('amendment_before_v2'),
        provisional_disposition=read(HERE/'provisional_query_dependent_v1/DISPOSITION.json'),
        cohort=dict(states=148,conditional_arms=5,conditional_banks=740,physical_roots=72,
            fixed_controls=len(labels),physical_replays=72*len(labels)),
        controls=controls,comparisons=comparisons,reference=reference_summary(validation),
        original_six_cases=[row for row in cases if row['root_index'] in OLD_CASES],root_cases=cases,
        root_receipts=compact,online_meeting=dict(all_exact={family:all(row[family] for row in meeting)
            for family in ('physical','shallow')},rows=meeting),
        preparation=dict(banks=read(HERE/'banks_preparation.json'),online=read(HERE/'online_preparation.json'),
            dated_graphs=graph_preparation),publication_probe=read(HERE/'publication_receipts.json'),
        orientation='Positive actual gain = baseline loss minus subject loss. Conditional paired differences are alight minus continue.',
        cost_scope='Whole-journey total is charged once. Cold is its subset; decision plus final partitions total. '
            'Gate, conditioning, generation, continuation, meeting checks and publication spans are nested costs, not extra fees.',
        limits=['Frozen six-day development cohort; no independent confirmation or per-root oracle.',
            'Conditional future_messages=0. Full G_pi is a fixed-128 sample estimate, not exact integration.',
            'Frozen-Combined reference is exact only for its frozen child marks and independent root-offer laws.',
            'Main travel replay uses ideal logical-time publication; wall-window channel probe is separate.',
            'Missing bounds use all 148 states or all 72 roots; resolved-only SE does not account for missing mass.'])
    write(HERE/'summary.json',result)
    write_controls(controls,comparisons)
    draw(controls)
    print(dict(controls=len(labels),physical_replays=72*len(labels),
        legacy_reference_exact=result['reference']['legacy_worlds_all_exact'],
        meeting_online_exact=result['online_meeting']['all_exact']))


if __name__=='__main__':
    main()
