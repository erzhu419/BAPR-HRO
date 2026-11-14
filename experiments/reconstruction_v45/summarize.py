"""V45 all-state structural evaluation and complete paired control accounting."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reconstruction_v34.summarize import numeric, read, write
from experiments.reconstruction_v38.run import physical_outcome
from experiments.reconstruction_v41.summarize import OLD_CASES, grouped_comparison, meeting_science_equal, pair
from experiments.reconstruction_v43.summarize import difference, distribution_summary, error_aggregate, point
from experiments.reconstruction_v44.summarize import (
    BUDGETS, control_summary, joint_budgets, local_value_equal, net_compute,
    validate_partition,
)

HERE = Path(__file__).resolve().parent
V44 = HERE.parent/'reconstruction_v44'
V41 = HERE.parent/'reconstruction_v41'
BANKS = ('matched128', 'markov128', 'absorbing')
LABELS = ('rolling_fixed', 'full128', 'matched128', 'markov128', 'absorbing')
CANDIDATES = ('markov128', 'absorbing')


def same_bank(fresh, saved):
    """The reduced executor must retain every original sampled prefix and value."""
    for key in ('seed', 'count', 'generated_count', 'future_messages', 'public_history', 'clock', 'deadline'):
        if fresh[key] != saved[key]:
            raise ValueError(f'Same-bank identity changed: {key}')
    if fresh['count'] != 128 or fresh['generated_count'] != 128:
        raise ValueError('The complete original 128 paired worlds are required')
    for arm in ('alight', 'continue'):
        if len(fresh['rows'][arm]) != 128 or len(saved['rows'][arm]) != 128:
            raise ValueError(f'Missing paired worlds: {arm}')
        for index, (a, b) in enumerate(zip(fresh['rows'][arm], saved['rows'][arm], strict=True)):
            if (a['world_index'] != index or b['world_index'] != index or a['prefix'] != b['prefix']
                    or not local_value_equal(a['value'], b['value'])):
                raise ValueError(f'Same-bank prefix or target changed: {arm}/{index}')
    return True


def summary_difference(left, right):
    """All-state bounds, retaining the distinction between population and sample."""
    low, high = left['mean_interval']; other_low, other_high = right['mean_interval']
    return dict(mean=None if left['mean'] is None or right['mean'] is None else left['mean']-right['mean'],
        mean_interval=[low-other_high, high-other_low])


def scalar_summary(value):
    return dict(mean=value['mean'], mean_interval=[value['lower'], value['upper']],
        mean_unresolved_probability_bound=value.get('unresolved_probability', value.get('unresolved_probability_bound',0.)),
        paired_se=None, scope='Exact finite-model population integration; no Monte Carlo confidence interval')


def population_mode(row):
    return row['generated_count']==0


def values_summary(row, label):
    if label == 'absorbing' and population_mode(row):
        actions = {arm: scalar_summary(row['actions'][arm]) for arm in ('alight', 'continue')}
        return actions, scalar_summary(difference(row['actions']['alight'], row['actions']['continue']))
    values = {arm: [r['value'] if label != 'full128' else
        (point(r['loss']) if r['loss'] is not None else
         dict(lower=0., upper=row['deadline']-row['clock'], mean=None, unresolved_probability=1.))
        for r in row['rows'][arm]] for arm in ('alight', 'continue')}
    return ({arm: distribution_summary(v) for arm, v in values.items()},
        distribution_summary([difference(a, b) for a, b in zip(values['alight'], values['continue'], strict=True)]))


def numeric_totals(rows):
    """Sum descriptive numeric leaves, including the evaluator's mechanism counters."""
    answer = {}
    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, (*path, key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            name = '.'.join(path)
            answer[name] = answer.get(name, 0.)+value
    for row in rows:
        visit(row, ())
    return answer


def mechanism(row):
    return {key: row[key] for key in ('cost', 'cache_stats', 'root_law_stats', 'root_independence',
        'structural', 'canonical_stats', 'absorption_counters') if key in row}


def population_diagnostics(row):
    """Keep population mass and full-policy bounds separate from sampled means."""
    atoms=row['atoms']
    mass={arm:sum(a['probability'] for a in atoms[arm]) for arm in ('alight','continue')}
    if any(abs(value-1.)>1e-8 for value in mass.values()) or any(
            a['probability']<0 for values in atoms.values() for a in values):
        raise ValueError('Absorbing population did not conserve all probability mass')
    lower,upper=row['full_pi_difference_interval']
    bounds=row['full_pi_bounds'];selected=row['selected_action']
    other='continue' if selected=='alight' else 'alight'
    return dict(probability_mass=mass,atoms={arm:len(values) for arm,values in atoms.items()},
        platform_probability={arm:sum(a['probability'] for a in values if a['kind']=='platform') for arm,values in atoms.items()},
        terminal_probability={arm:sum(a['probability'] for a in values if a['kind']=='terminal') for arm,values in atoms.items()},
        unresolved_prefix_probability={arm:sum(a['probability'] for a in values if a['kind']=='unresolved_prefix') for arm,values in atoms.items()},
        full_pi_bounds=bounds,full_pi_difference_interval=[lower,upper],
        full_pi_gap_width=upper-lower,residual_interval=row['residual_interval'],
        full_pi_rank_certified=upper < -1e-9 or lower > 1e-9,
        selected_action_regret_upper=max(0.,bounds[selected]['upper']-bounds[other]['lower']))


def absorbing_validation(row):
    if population_mode(row):
        if row['count']!=0:
            raise ValueError('Population count must not describe sampled worlds')
    elif row['count']!=128 or row['generated_count']!=128:
        raise ValueError('A structural fallback must retain the complete original 128 worlds')


def compact_gate(gate, root_index, gate_index, label):
    actions, gap = values_summary(gate, label)
    return dict(root_index=root_index, gate_index=gate_index, cid=gate['cid'], clock=gate['clock'],
        source=gate['source'], seed=gate.get('seed'), count=gate['count'], generated_count=gate['generated_count'],
        base_action=gate['base_action'], selected_action=gate['selected_action'],
        selected_changed=gate['selected_action'] != gate['base_action'], actions=actions, gap=gap,
        cost=gate['cost'], mechanism=mechanism(gate),
        population=population_diagnostics(gate) if label=='absorbing' and population_mode(gate) else None,
        estimator=gate.get('estimator'),structural_fallback=label=='absorbing' and not population_mode(gate))


def gate_comparison(subject, baseline, root_index, subject_label, baseline_label):
    common, used, unmatched = [], set(), []
    for index, gate in enumerate(subject['gates']):
        found = [j for j, other in enumerate(baseline['gates']) if j not in used and
            gate['cid'] == other['cid'] and gate['clock'] == other['clock'] and
            gate['public_history'] == other['public_history']]
        if not found:
            unmatched.append(index)
            continue
        j = found[0]; used.add(j); other = baseline['gates'][j]
        if gate['source'] != other['source']:
            raise ValueError('Common public state changed the frozen source assignment')
        if subject_label != 'absorbing' or not population_mode(gate):
            if any(gate[key] != other[key] for key in ('seed', 'count', 'generated_count')):
                raise ValueError('Common public state changed paired sampling identity')
        if subject_label == 'markov128' and baseline_label == 'matched128':
            same_bank(gate, other)
            if any(gate[k] != other[k] for k in ('base_action', 'selected_action')):
                raise ValueError('Same-bank controller changed an actual action')
        own_actions, own_gap = values_summary(gate, subject_label)
        _, other_gap = values_summary(other, baseline_label)
        common.append(dict(root_index=root_index, subject_gate=index, baseline_gate=j,
            cid=gate['cid'], clock=gate['clock'], source=gate['source'],
            same_base_action=gate['base_action'] == other['base_action'],
            same_selected_action=gate['selected_action'] == other['selected_action'],
            subject_action=gate['selected_action'], baseline_action=other['selected_action'],
            subject_gap=own_gap, baseline_gap=other_gap,
            subject_minus_baseline_gap=summary_difference(own_gap, other_gap)))
    return dict(root_index=root_index, common=common, unmatched_subject_indices=unmatched,
        unmatched_baseline_indices=[j for j in range(len(baseline['gates'])) if j not in used])


def reference_summary():
    records = []
    for index in range(148):
        banks = {label: read(HERE/'banks'/f'{index:03}_{label}.json.gz') for label in BANKS}
        old = read(V44/'references'/f'{index:03}.json.gz')
        full = read(V41/'banks'/f'{index:03}_shallow.json.gz')
        same_bank(banks['matched128'], old)
        same_bank(banks['markov128'], banks['matched128'])
        if any(banks['markov128'][k] != banks['matched128'][k] for k in ('base_action', 'selected_action')):
            raise ValueError(f'Same-bank reference action changed: {index}')
        for label, row in banks.items():
            if row['index'] != index or row['root_index'] != old['root_index']:
                raise ValueError(f'Wrong frozen bank: {index}/{label}')
            if any(row[k] != old[k] for k in ('public_history', 'clock', 'deadline')):
                raise ValueError(f'Population evaluation changed the fixed problem: {index}')
        absorbing_validation(banks['absorbing'])
        result = {}
        for label, row in {**banks, 'full128': full}.items():
            actions, gap = values_summary(row, label)
            result[label] = dict(actions=actions, gap=gap, base_action=row['base_action'],
                selected_action=row['selected_action'], complete_cost=row['complete_cost'], mechanism=mechanism(row))
        differences = {label: dict(actions={arm: summary_difference(result[label]['actions'][arm], result['full128']['actions'][arm])
            for arm in ('alight', 'continue')}, gap=summary_difference(result[label]['gap'], result['full128']['gap'])) for label in BANKS}
        records.append(dict(index=index, root_index=old['root_index'], source=old['source'], seed=old['seed'],
            configurations=result, minus_full_mc=differences,
            population_minus_matched=summary_difference(result['absorbing']['gap'], result['matched128']['gap']),
            population=population_diagnostics(banks['absorbing']) if population_mode(banks['absorbing']) else None,
            absorbing_generated_worlds=banks['absorbing']['generated_count'],
            absorbing_structural_fallback=not population_mode(banks['absorbing'])))
    population_records=[r for r in records if r['population'] is not None]
    return dict(states=148,paired_worlds_per_sample_state=128,all_same_bank_prefixes_and_values_equal=True,
        absorbing_samples_generated=sum(r['absorbing_generated_worlds'] for r in records),
        absorbing_population_states=len(population_records),
        absorbing_structural_fallback_states=[r['index'] for r in records if r['absorbing_structural_fallback']],records=records,
        costs={label:dict(cpu=numeric([r['configurations'][label]['complete_cost']['process_cpu'] for r in records]),
                         wall=numeric([r['configurations'][label]['complete_cost']['wall'] for r in records])) for label in BANKS},
        mechanism_totals={label:numeric_totals([r['configurations'][label]['mechanism'] for r in records]) for label in BANKS},
        structural_eligible_states={label:sum(r['configurations'][label]['mechanism']['structural']['eligible'] for r in records) for label in CANDIDATES},
        errors_vs_full_mc={label:{field:error_aggregate([r['minus_full_mc'][label]['gap'] if field=='gap' else
            r['minus_full_mc'][label]['actions'][field] for r in records]) for field in ('alight','continue','gap')} for label in BANKS},
        population_minus_sample_gap=numeric([r['population_minus_matched']['mean'] for r in records if r['population_minus_matched']['mean'] is not None]),
        population_action_changes=[r['index'] for r in records if r['configurations']['absorbing']['selected_action'] != r['configurations']['matched128']['selected_action']],
        population_bounds=dict(states=len(population_records),rank_certified=sum(r['population']['full_pi_rank_certified'] for r in population_records),
            gap_width=numeric([r['population']['full_pi_gap_width'] for r in population_records]),
            selected_action_regret_upper=numeric([r['population']['selected_action_regret_upper'] for r in population_records])),
        scope='Population κ versus the original finite-bank κ and full-policy Monte Carlo estimates. '
              'Full-policy population bounds are not checked against individual finite-bank Monte Carlo means.')


def online_summary():
    compact = {label: [] for label in LABELS}
    compared = {(a,b) for a in LABELS for b in ('rolling_fixed','full128','matched128') if a != b}
    pairs = {a+'__vs__'+b: [] for a,b in sorted(compared)}
    common = {a+'__vs__'+b: [] for a,b in (('markov128','matched128'),('absorbing','matched128'),('absorbing','full128'))}
    cases = []
    for index in range(72):
        roots = {label: read(HERE/'roots'/f'{index:03}_{label}.json.gz') for label in LABELS}
        for label, row in roots.items():
            if row['root_index'] != index or row['root_id'] != roots['rolling_fixed']['root_id']:
                raise ValueError(f'Wrong frozen root: {index}/{label}')
            if label in ('rolling_fixed','full128','matched128'):
                old = read(V44/'roots'/f'{index:03}_{label}.json.gz')
                if label != 'matched128':
                    if not meeting_science_equal(row, old):
                        raise ValueError(f'Existing V44 comparator science changed: {index}/{label}')
                else:
                    if physical_outcome(row['outcome']) != physical_outcome(old['outcome']) or len(row['gates']) != len(old['gates']):
                        raise ValueError(f'V44 matched controller trajectory changed: {index}')
                    for a,b in zip(row['gates'],old['gates'],strict=True):
                        same_bank(a,b)
                        if any(a[k] != b[k] for k in ('base_action','selected_action')):
                            raise ValueError('V44 matched controller action changed')
            if label == 'absorbing':
                for g in row['gates']:absorbing_validation(g)
            compact[label].append(dict(root_index=index, root_id=row['root_id'], day=row['day'], origin=row['origin'],
                goal=row['goal'], loss_cap=row['loss_cap'], outcome={k:row['outcome'][k] for k in ('loss','status')},
                cost=row['cost'], decision_costs=row['decision_costs'], cost_partition=validate_partition(row),
                saved_answers_reused=row['saved_answers_reused'], source=f'roots/{index:03}_{label}.json.gz',
                gate_records=[compact_gate(g,index,j,label) for j,g in enumerate(row['gates'])]))
        if physical_outcome(roots['markov128']['outcome']) != physical_outcome(roots['matched128']['outcome']):
            raise ValueError(f'Same-bank control changed physical trajectory: {index}')
        for subject, baseline in sorted(compared):
            pairs[subject+'__vs__'+baseline].append(pair(roots[subject],roots[baseline]))
        for key, values in common.items():
            subject, baseline = key.split('__vs__')
            values.append(gate_comparison(roots[subject],roots[baseline],index,subject,baseline))
        first = roots['rolling_fixed']
        cases.append(dict(root_index=index,root_id=first['root_id'],day=first['day'],origin=first['origin'],goal=first['goal'],
            losses={label:r['outcome']['loss'] for label,r in roots.items()}, statuses={label:r['outcome']['status'] for label,r in roots.items()},
            process_cpu={label:r['cost']['total_process_cpu'] for label,r in roots.items()}, wall={label:r['cost']['total_wall'] for label,r in roots.items()}))
    markov_common=common['markov128__vs__matched128']
    if any(r['unmatched_subject_indices'] or r['unmatched_baseline_indices'] for r in markov_common):
        raise ValueError('Same-bank controls no longer visit identical public decision states')
    history=read(HERE.parent/'reconstruction_v43/existing_attribution.json')
    sets=dict(original_six_cases=set(OLD_CASES), original_seven_nonzero_cases=set(history['nonzero_outcome_roots']),
        original_four_changed_ties=set(history['changed_but_zero_outcome_roots']),
        historical_negative_cases={r['root_index'] for r in history['old_negative_roots']})
    controls={label:control_summary(rows) for label,rows in compact.items()}
    for label, rows in compact.items():
        controls[label]['mechanism_totals']=numeric_totals([g['mechanism'] for r in rows for g in r['gate_records']])
        if label in CANDIDATES:
            controls[label]['structural_eligible_gates']=sum(g['mechanism']['structural']['eligible'] for r in rows for g in r['gate_records'])
        if label=='absorbing':
            populations=[g['population'] for r in rows for g in r['gate_records'] if g['population'] is not None]
            controls[label]['population_bounds']=dict(gates=len(populations),
                structural_fallback_gates=sum(g['structural_fallback'] for r in rows for g in r['gate_records']),
                rank_certified=sum(p['full_pi_rank_certified'] for p in populations),
                gap_width=numeric([p['full_pi_gap_width'] for p in populations]),
                selected_action_regret_upper=numeric([p['selected_action_regret_upper'] for p in populations]))
    return dict(controls=controls, comparisons={key:grouped_comparison(rows) for key,rows in pairs.items()},
        net_compute_vs_matched={label:net_compute(compact[label],compact['matched128']) for label in CANDIDATES},
        joint_budgets_vs_matched={label:joint_budgets(compact[label],compact['matched128']) for label in CANDIDATES},
        root_cases=cases,root_receipts=compact, common_gate_comparisons={key:dict(
            common_gates=sum(len(r['common']) for r in rows),
            same_selected_actions=sum(m['same_selected_action'] for r in rows for m in r['common']),
            unmatched_subject_gates=sum(len(r['unmatched_subject_indices']) for r in rows),
            unmatched_baseline_gates=sum(len(r['unmatched_baseline_indices']) for r in rows),records=rows) for key,rows in common.items()},
        **{name:[r for r in cases if r['root_index'] in indices] for name,indices in sets.items()},
        all_original_comparators_exact=True, all_markov_same_bank_trajectories_equal=True)


def csv_output(name, rows):
    with (HERE/name).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def outputs(result):
    csv_output('controls.csv',[dict(configuration=label,
        mean_loss_lower=c['quality']['full_cohort_mean_loss_interval'][0],mean_loss_upper=c['quality']['full_cohort_mean_loss_interval'][1],
        mean_journey_cpu=c['process_cpu']['mean'],mean_journey_wall=c['wall']['mean'],
        decision_cpu_p50=c['decision_cpu']['p50'],decision_cpu_p95=c['decision_cpu']['p95'],
        decision_wall_p50=c['decision_wall']['p50'],decision_wall_p95=c['decision_wall']['p95']) for label,c in result['controls'].items()])
    roots=[]
    for row in result['root_cases']:
        item={key:row[key] for key in ('root_index','day','origin','goal')}
        item.update({f'{label}_{measure}':row[field][label] for label in LABELS for measure,field in
            (('loss','losses'),('cpu','process_cpu'),('wall','wall'))})
        for label in CANDIDATES:
            item[f'{label}_gain_vs_matched']=None if row['losses'][label] is None or row['losses']['matched128'] is None else row['losses']['matched128']-row['losses'][label]
            item[f'{label}_cpu_minus_matched']=row['process_cpu'][label]-row['process_cpu']['matched128']
            item[f'{label}_wall_minus_matched']=row['wall'][label]-row['wall']['matched128']
        roots.append(item)
    csv_output('roots.csv',roots)
    csv_output('budgets.csv',[dict(configuration=label,measure=measure,budget=budget,**row)
        for label,c in result['controls'].items() for measure,values in c['budgets'].items() for budget,row in values.items()])
    refs=[]
    for r in result['reference']['records']:
        item=dict(index=r['index'],root_index=r['root_index'],source=r['source'])
        for label,c in r['configurations'].items():
            item.update({label+'_'+arm:c['actions'][arm]['mean'] for arm in ('alight','continue')})
            item.update({label+'_gap':c['gap']['mean'],label+'_gap_se':c['gap']['paired_se'],
                label+'_selected':c['selected_action'],label+'_cpu':c['complete_cost']['process_cpu'],label+'_wall':c['complete_cost']['wall']})
        refs.append(item)
    csv_output('reference.csv',refs)
    fig,axis=plt.subplots(figsize=(8,5))
    for label,c in result['controls'].items():
        low,high=c['quality']['full_cohort_mean_loss_interval']
        axis.plot([c['process_cpu']['mean']]*2,[low,high],marker='o',label=label)
    axis.set(xscale='log',xlabel='Mean whole-journey process CPU (s)',ylabel='Mean truncated loss over all 72 roots (s)')
    axis.grid(alpha=.2);axis.legend(fontsize=8);fig.tight_layout();fig.savefig(HERE/'quality_time.png',dpi=180);plt.close(fig)


def main():
    required=[HERE/'protocol.json']+[HERE/'banks'/f'{i:03}_{label}.json.gz' for i in range(148) for label in BANKS]
    required += [HERE/'roots'/f'{i:03}_{label}.json.gz' for i in range(72) for label in LABELS]
    absent=[str(path.relative_to(HERE)) for path in required if not path.exists()]
    if absent:raise SystemExit(f'Awaiting {len(absent)} required V45 receipts; first missing: {absent[:5]}')
    result=dict(stage='V45',development_only=True,cohort=dict(states=148,bank_configurations=3,banks=444,roots=72,controls=5,physical_replays=360),
        reference=reference_summary(),**online_summary(),limits='Same-bank Markov evaluation preserves the original 128 worlds. '
        'Absorbing integration evaluates population κ, not full pi. Its full-policy bounds are population statements, not finite-sample confidence intervals. '
        'Budgets describe measured CPU/wall, not physical deadlines. All positive, negative, tied and unresolved cases are retained; all dates remain development data.')
    write(HERE/'summary.json',result);outputs(result)
    print('V45 complete: all 444 banks and 360 online controls retained.')


if __name__=='__main__':
    main()
