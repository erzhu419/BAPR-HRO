"""Summarize every declared V40 configuration, retaining missing-outcome mass."""

from collections import Counter
import csv
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reconstruction_v34.summarize import first_divergence, numeric, read, write


HERE = Path(__file__).resolve().parent
OLD_CASES = (16, 25, 28, 57, 63, 64)


def loss_bounds(row):
    loss = row['outcome']['loss']
    return (0., row['loss_cap']) if loss is None else (loss, loss)


def cohort_quality(rows):
    """The full-cohort denominator includes every unresolved physical replay."""
    bounds = [loss_bounds(row) for row in rows]
    resolved = [row['outcome']['loss'] for row in rows if row['outcome']['loss'] is not None]
    interval = [sum(x[j] for x in bounds)/len(rows) for j in (0, 1)]
    return dict(denominator=len(rows), resolved=len(resolved), unresolved=len(rows)-len(resolved),
        full_cohort_mean_loss=interval[0] if len(resolved) == len(rows) else None,
        full_cohort_mean_loss_interval=interval,
        resolved_only_descriptive=numeric(resolved),
        statuses=dict(Counter(row['outcome']['status'] for row in rows)),
        interval_scope='Empirical unresolved outcomes receive [0, loss_cap]; not a confidence interval.')


def budget_feasibility(rows, allowance):
    """Use complete measured decision CPU, including late work and cold setup."""
    decisions = [decision for row in rows for decision in row['decision_costs']]
    cpu = [decision['process_cpu'] for decision in decisions]
    within = [value <= allowance for value in cpu]
    overruns = [max(0., value-allowance) for value in cpu]
    gates = [decision for decision in decisions if decision['gate']]
    return dict(cpu_budget_seconds=allowance, decisions=len(decisions),
        within=sum(within), exceeded=len(within)-sum(within),
        within_fraction=sum(within)/len(within) if within else None,
        roots_all_decisions_within=sum(all(d['process_cpu'] <= allowance for d in row['decision_costs']) for row in rows),
        roots=len(rows), gate_decisions=len(gates),
        gate_within=sum(d['process_cpu'] <= allowance for d in gates),
        nongate_decisions=len(decisions)-len(gates),
        decision_cpu_seconds=numeric(cpu), overrun_seconds=numeric(overruns),
        maximum_cpu_over_budget_ratio=max(cpu, default=0.)/allowance,
        comparison_scope='Cooperative quota feasibility; unchanged logical travel clocks.')


def gate_summary(rows):
    gates = [gate for row in rows for gate in row['gates']]
    return dict(calls=len(gates), committed_pairs=numeric([g['count'] for g in gates]),
        executed_pairs=numeric([g['executed_count'] for g in gates]),
        late_pairs=numeric([g['late_pair_count'] for g in gates]),
        generated_pairs=numeric([g['generated_count'] for g in gates]),
        sampled_but_unexecuted_pairs=sum(g['generated_count']-g['executed_count'] for g in gates),
        zero_pair_fallbacks=sum(g['count'] == 0 for g in gates),
        ceiling_reached=sum(g['count'] == g['max_count'] for g in gates),
        changed_from_base=sum(g['selected_action'] != g['actions']['base'] for g in gates),
        new_public_states=sum(g['source'] is None for g in gates),
        seed_origins=dict(Counter(str(g['seed_origin']) for g in gates)),
        stop_reasons=dict(Counter(g['stop_reason'] for g in gates)),
        empirical_unresolved_committed_pairs=sum(g['unresolved_pairs'] for g in gates))


def compact_root(row, source):
    gates = [dict(count=g['count'], executed_count=g['executed_count'], late_pair_count=g['late_pair_count'],
        generated_count=sum(b['count'] for b in g['blocks']), max_count=g['max_count'],
        selected_action=g['selected_action'], actions=g['actions'], source=g['source'],
        seed_origin=g['seed_origin'], stop_reason=g['stop_reason'],
        unresolved_pairs=g['paired']['unresolved_pairs']) for g in row['gates']]
    return dict(root_index=row['root_index'], root_id=row['root_id'], day=row['day'],
        loss_cap=row['loss_cap'], outcome={k:row['outcome'][k] for k in ('loss', 'status')},
        cost=row['cost'], decision_costs=row['decision_costs'], gates=gates,
        anchor128_exact=row.get('anchor128_exact'), cheap_exact_original=row.get('cheap_exact_original'),
        saved_answers_reused=row['saved_answers_reused'], source=source)


def pair_row(subject, baseline, allowance=None):
    alo, ahi = loss_bounds(subject)
    blo, bhi = loss_bounds(baseline)
    a, b = subject['outcome']['loss'], baseline['outcome']['loss']
    return dict(root_index=subject['root_index'], root_id=subject['root_id'], day=subject['day'],
        subject_loss=a, baseline_loss=b,
        gain_seconds=None if a is None or b is None else b-a,
        gain_interval=[blo-ahi, bhi-alo], subject_status=subject['outcome']['status'],
        baseline_status=baseline['outcome']['status'],
        subject_cpu=subject['cost']['total_process_cpu'], baseline_cpu=baseline['cost']['total_process_cpu'],
        subject_minus_baseline_cpu=subject['cost']['total_process_cpu']-baseline['cost']['total_process_cpu'],
        subject_minus_baseline_wall=subject['cost']['total_wall']-baseline['cost']['total_wall'],
        cpu_budget=allowance,
        subject_all_decisions_within=None if allowance is None else
            all(d['process_cpu'] <= allowance for d in subject['decision_costs']),
        baseline_all_decisions_within=None if allowance is None else
            all(d['process_cpu'] <= allowance for d in baseline['decision_costs']),
        **first_divergence(subject['outcome'], baseline['outcome']))


def paired_summary(rows):
    gains = [row['gain_seconds'] for row in rows if row['gain_seconds'] is not None]
    interval = [sum(row['gain_interval'][j] for row in rows)/len(rows) for j in (0, 1)]
    return dict(denominator=len(rows), resolved_pairs=len(gains), unresolved_pairs=len(rows)-len(gains),
        roots_with_quota_comparison=sum(r['cpu_budget'] is not None for r in rows),
        roots_both_controllers_within_quota=sum(r['subject_all_decisions_within'] is True and
            r['baseline_all_decisions_within'] is True for r in rows),
        roots_subject_exceeded_quota=sum(r['subject_all_decisions_within'] is False for r in rows),
        roots_baseline_exceeded_quota=sum(r['baseline_all_decisions_within'] is False for r in rows),
        wins=sum(g > 0 for g in gains), harms=sum(g < 0 for g in gains), ties=sum(g == 0 for g in gains),
        certainly_positive=sum(row['gain_interval'][0] > 0 for row in rows),
        certainly_negative=sum(row['gain_interval'][1] < 0 for row in rows),
        full_cohort_mean_gain=interval[0] if len(gains) == len(rows) else None,
        full_cohort_mean_gain_interval=interval, resolved_only_descriptive=numeric(gains),
        subject_minus_baseline_cpu_seconds=numeric([r['subject_minus_baseline_cpu'] for r in rows]),
        aggregate_cpu_ratio=sum(r['subject_cpu'] for r in rows)/sum(r['baseline_cpu'] for r in rows),
        faster=sum(r['subject_minus_baseline_cpu'] < 0 for r in rows),
        slower=sum(r['subject_minus_baseline_cpu'] > 0 for r in rows),
        subject_minus_baseline_wall_seconds=numeric([r['subject_minus_baseline_wall'] for r in rows]),
        actions_identical=sum(r['actions_identical'] for r in rows),
        common_prefix_history_mismatches=sum(not r['common_prefix_history_equal'] for r in rows))


def arm_summary(rows, setting, quotas):
    result = dict(quota_feasibility={name:budget_feasibility(rows, budget) for name, budget in quotas.items()
                                    if setting is None or name == setting},
        quality=cohort_quality(rows),
        process_cpu_seconds=numeric([r['cost']['total_process_cpu'] for r in rows]),
        wall_seconds=numeric([r['cost']['total_wall'] for r in rows]),
        decision_cpu_sum_seconds=numeric([r['cost']['decision_cpu_sum'] for r in rows]),
        final_execution_cpu_seconds=numeric([r['cost']['final_execution_cpu'] for r in rows]),
        gates=gate_summary(rows),
        anchor128_checked=sum(r['anchor128_exact'] is not None for r in rows),
        anchor128_all_exact=all(r['anchor128_exact'] is True for r in rows)
                           if setting == 'pairs128' else None,
        cheap_exact_original=all(r['cheap_exact_original'] is True for r in rows) if setting is None else None,
        saved_answers_reused=sum(r['saved_answers_reused'] for r in rows))
    result['by_day'] = {day:dict(quality=cohort_quality([r for r in rows if r['day'] == day]),
        process_cpu_seconds=numeric([r['cost']['total_process_cpu'] for r in rows if r['day'] == day]))
        for day in sorted({r['day'] for r in rows})}
    return result


def prefix_record(bank, *, count=None, quota=None):
    eligible = [p for p in bank['prefixes'] if
                (count is None or p['count'] <= count) and (quota is None or p['elapsed_cpu'] <= quota)]
    prefix = eligible[-1] if eligible else None
    n = 0 if prefix is None else prefix['count']
    return dict(count=n, elapsed_cpu=None if prefix is None else prefix['elapsed_cpu'],
        selected_action=bank['actions']['base'] if prefix is None else prefix['selected_action'],
        changed_from_base=False if prefix is None else prefix['selected_action'] != bank['actions']['base'],
        mean_difference_interval=None if prefix is None else prefix['full_bank_mean_gain_interval'],
        unresolved_pairs=sum(a['loss'] is None or b['loss'] is None
            for a, b in zip(bank['rows']['alternative'][:n], bank['rows']['base'][:n])))


def reference_summary(manifest, variants, quotas):
    validation = read(HERE/'reference_validation.json')
    if validation['completed'] != len(manifest):
        raise ValueError('All frozen reference states must be measured before summarizing')
    checked = {row['source']:row for row in validation['states']}
    states, costs = [], {v:[] for v in variants}
    checkpoints = {f'pairs{n}':dict(count=n) for n in (16, 32, 64, 128, 256, 512)}
    checkpoints.update({name:dict(quota=quota) for name, quota in quotas.items()})
    for entry in manifest:
        stem = entry['source'].replace('/', '_').removesuffix('.json.gz')
        banks = {v:read(HERE/'banks'/f'{stem}_{v}.json.gz') for v in variants}
        anchor = banks[variants[0]]
        for variant, bank in banks.items():
            costs[variant].append(bank['complete_cost'])
        states.append(dict(source=entry['source'], root_index=entry['state']['root_index'], seed=entry['seed'],
            anchor128_all_exact=checked[entry['source']]['anchor128_exact'],
            implementations512_exact=checked[entry['source']]['full512_implementations_equal'],
            unresolved_pairs=anchor['paired']['unresolved_pairs'],
            costs={v:b['complete_cost'] for v,b in banks.items()},
            prefixes={v:{name:prefix_record(bank, **spec) for name, spec in checkpoints.items()}
                      for v, bank in banks.items()}))
    descriptive = {}
    for variant in variants:
        descriptive[variant] = {}
        for checkpoint in checkpoints:
            rows = [state['prefixes'][variant][checkpoint] for state in states]
            available = [r for r in rows if r['count']]
            descriptive[variant][checkpoint] = dict(states=len(rows), states_with_pairs=len(available),
                committed_pairs=numeric([r['count'] for r in rows]),
                checkpoint_cpu=numeric([r['elapsed_cpu'] for r in available]),
                changed_from_base=sum(r['changed_from_base'] for r in rows),
                unresolved_pairs=sum(r['unresolved_pairs'] for r in rows),
                mean_interval_among_nonempty_prefixes=None if not available else
                    [sum(r['mean_difference_interval'][j] for r in available)/len(available) for j in (0, 1)])
    return dict(states=len(states), measurements=len(states)*len(variants), pairs_per_state=512,
        difference_orientation='alternative action loss minus base action loss; negative favors alternative',
        scientific_validation_source='reference_validation.json: measured-run prefix and full-stream comparisons',
        anchor128_all_exact=all(s['anchor128_all_exact'] for s in states),
        implementations512_all_exact=all(s['implementations512_exact'] for s in states),
        unresolved_pairs_per_implementation=sum(s['unresolved_pairs'] for s in states),
        cpu={v:numeric([r['process_cpu'] for r in rows]) for v,rows in costs.items()},
        wall={v:numeric([r['wall'] for r in rows]) for v,rows in costs.items()},
        descriptive_prefixes=descriptive, state_pairs=states,
        scope='Unrestricted fixed-state 512-bank prefixes are descriptive conditional comparisons, '
              'not actual stopped online replays or population certificates.')


def draw(controls, variants, settings, cheap, quotas):
    colors = dict(zip(variants, ('tab:blue', 'tab:orange', 'tab:green')))
    markers = dict(zip(settings, ('o', 's', '^', 'D', 'P', '*')))
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, value in controls.items():
        variant, setting = name.split('__')
        cheap_arm = name in cheap
        lo, hi = value['quality']['full_cohort_mean_loss_interval']
        x, y = value['process_cpu_seconds']['mean'], (lo+hi)/2
        marker = (('x' if variant == 'cheap_v38' else '+') if cheap_arm else markers[setting])
        ax.errorbar(x, y, yerr=[[y-lo], [hi-y]],
            color='gray' if cheap_arm else colors[variant],
            marker=marker if value['quality']['full_cohort_mean_loss'] is not None else 'None',
            linestyle='none', markersize=6, capsize=3, alpha=.75, label=name)
    ax.set(xscale='log', xlabel='Mean actual whole-journey CPU (s, log scale)',
           ylabel='Full 72-root mean truncated loss (s)', title='All fixed configurations; bars are missing-outcome bounds')
    ax.grid(alpha=.2)
    ax.legend(fontsize=6.5, ncol=2, loc='center left', bbox_to_anchor=(1.01, .5))
    fig.tight_layout()
    fig.savefig(HERE/'quality_budget.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = list(quotas.values())
    for variant in variants:
        rows = [controls[f'{variant}__{name}'] for name in quotas]
        feasible = [r['quota_feasibility'][name] for r,name in zip(rows,quotas)]
        axes[0,0].plot(x, [1-r['within_fraction'] for r in feasible], 'o-', color=colors[variant], label=variant)
        axes[0,1].plot(x, [r['gates']['committed_pairs']['mean'] or 0 for r in rows], 'o-', color=colors[variant])
        axes[1,0].plot(x, [r['overrun_seconds']['mean'] or 0 for r in feasible], 'o-', color=colors[variant])
        axes[1,1].plot(x, [r['gates']['zero_pair_fallbacks']/r['gates']['calls'] if r['gates']['calls'] else 0
                          for r in rows], 'o-', color=colors[variant])
    for name in cheap:
        rows = controls[name]['quota_feasibility']
        axes[0,0].plot(x, [1-rows[q]['within_fraction'] for q in quotas], '--', color='gray', alpha=.35, linewidth=.8)
    axes[0,0].set(ylabel='Decisions exceeding CPU quota / all decisions', ylim=(-.03, 1.03),
                  title='12 cheap configurations shown as dashed curves')
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(ylabel='Mean committed paired worlds per gate')
    axes[1,0].set(ylabel='Mean overrun across all decisions (s)')
    axes[1,1].set(ylabel='Zero-pair fallback gates / all gates', ylim=(-.03, 1.03))
    for ax in axes.flat:
        ax.set(xscale='log', xlabel='Per-decision cooperative CPU quota (s)')
        ax.grid(alpha=.2)
    fig.suptitle('Budget feasibility precedes quality interpretation; all late work is charged')
    fig.tight_layout()
    fig.savefig(HERE/'budget_feasibility.png', dpi=170)
    plt.close(fig)


def write_controls_csv(result, quotas):
    """One fixed configuration per row; unresolved means are never imputed."""
    baseline = 'cheap_frontier__combined_rolling'
    fields = ('configuration', 'cpu_quota_seconds', 'mean_journey_cpu_seconds',
        'mean_journey_wall_seconds', 'cohort_roots', 'mean_loss_lower_seconds',
        'mean_loss_upper_seconds', 'baseline', 'wins', 'harms', 'ties', 'unresolved_pairs',
        'mean_gain_seconds', 'mean_gain_lower_seconds', 'mean_gain_upper_seconds',
        'decision_exceeded', 'decision_count', 'roots_all_decisions_within',
        'mean_committed_pairs_per_gate', 'zero_pair_fallbacks')
    with (HERE/'controls.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, arm in result['controls'].items():
            setting = name.split('__')[1]
            allowance = None if name.startswith('cheap_') else quotas.get(setting)
            feasibility = arm['quota_feasibility'].get(setting) if allowance is not None else None
            gains, intervals = [], []
            for root in result['root_pairs']:
                a, b = root['losses'][name], root['losses'][baseline]
                if a is not None and b is not None:
                    gains.append(b-a)
                alo, ahi = root['loss_intervals'][name]
                blo, bhi = root['loss_intervals'][baseline]
                intervals.append((blo-ahi, bhi-alo))
            count = len(intervals)
            lo, hi = arm['quality']['full_cohort_mean_loss_interval']
            writer.writerow(dict(configuration=name, cpu_quota_seconds=allowance,
                mean_journey_cpu_seconds=arm['process_cpu_seconds']['mean'],
                mean_journey_wall_seconds=arm['wall_seconds']['mean'], cohort_roots=count,
                mean_loss_lower_seconds=lo, mean_loss_upper_seconds=hi, baseline=baseline,
                wins=sum(g > 0 for g in gains), harms=sum(g < 0 for g in gains),
                ties=sum(g == 0 for g in gains), unresolved_pairs=count-len(gains),
                mean_gain_seconds=sum(gains)/count if len(gains) == count else None,
                mean_gain_lower_seconds=sum(pair[0] for pair in intervals)/count,
                mean_gain_upper_seconds=sum(pair[1] for pair in intervals)/count,
                decision_exceeded=None if feasibility is None else feasibility['exceeded'],
                decision_count=sum(len(r['decision_costs']) for r in result['root_receipts'][name]),
                roots_all_decisions_within=None if feasibility is None else feasibility['roots_all_decisions_within'],
                mean_committed_pairs_per_gate=arm['gates']['committed_pairs']['mean'],
                zero_pair_fallbacks=arm['gates']['zero_pair_fallbacks']))


def main():
    protocol = read(HERE/'protocol.json')
    variants = protocol['variants']
    settings = [b['name'] for b in protocol['budgets']]
    quotas = {b['name']:b['cpu_seconds'] for b in protocol['budgets'] if b['cpu_seconds'] is not None}
    rollout = [f'{v}__{setting}' for v in variants for setting in settings]
    cheap = [f'cheap_{impl}__{method}' for impl in ('v38', 'frontier') for method in protocol['cheap_methods']]
    configurations = rollout+cheap
    roots, comparisons, implementation_pairs = {name:[] for name in configurations}, {}, {}
    for subject in rollout:
        comparisons[subject] = {baseline:[] for baseline in cheap}
    for setting in settings:
        for baseline, subject in combinations(variants, 2):
            implementation_pairs[f'{subject}_vs_{baseline}__{setting}'] = []
    root_pairs = []
    for index in range(72):
        current = {}
        for name in configurations:
            source = f'roots/{index:03}_{name}.json.gz'
            row = read(HERE/source)
            if row['root_index'] != index:
                raise ValueError(f'Frozen root mismatch: {source}')
            current[name] = row
            roots[name].append(compact_root(row, source))
        if len({r['root_id'] for r in current.values()}) != 1:
            raise ValueError(f'Configuration root mismatch: {index}')
        for subject in rollout:
            for baseline in cheap:
                comparisons[subject][baseline].append(pair_row(current[subject], current[baseline],
                    quotas.get(subject.split('__')[1])))
        for setting in settings:
            for baseline, subject in combinations(variants, 2):
                implementation_pairs[f'{subject}_vs_{baseline}__{setting}'].append(
                    pair_row(current[f'{subject}__{setting}'], current[f'{baseline}__{setting}'], quotas.get(setting)))
        root_pairs.append(dict(root_index=index, root_id=current[rollout[0]]['root_id'], day=current[rollout[0]]['day'],
            losses={name:r['outcome']['loss'] for name,r in current.items()},
            loss_intervals={name:loss_bounds(r) for name,r in current.items()},
            cpu={name:r['cost']['total_process_cpu'] for name,r in current.items()},
            statuses={name:r['outcome']['status'] for name,r in current.items()}))
    controls = {name:arm_summary(rows, None if name in cheap else name.split('__')[1], quotas)
                for name, rows in roots.items()}

    def pack(rows):
        return dict(**paired_summary(rows), rows=rows,
            by_day={day:paired_summary([r for r in rows if r['day'] == day])
                    for day in sorted({r['day'] for r in rows})})

    manifest = read(HERE.parent/'reconstruction_v38/seed_manifest.json')
    reference = reference_summary(manifest, variants, quotas)
    result = dict(stage='V40', development_only=True, protocol='protocol.json',
        cohort=dict(roots=72, rollout_configurations=len(rollout), cheap_configurations=len(cheap),
                    physical_replays=72*len(configurations), all_declared_receipts_loaded=True),
        budget_semantics='Soft cooperative admission at complete-pair boundaries, not hard real time. '
            'Late pairs are charged but cannot update the incumbent. Inspect quota_feasibility before quality.',
        controls=controls, reference=reference,
        rollout_vs_each_cheap={subject:{baseline:pack(rows) for baseline,rows in baselines.items()}
                              for subject,baselines in comparisons.items()},
        same_setting_implementation_pairs={name:pack(rows) for name,rows in implementation_pairs.items()},
        original_six_cases=[row for row in root_pairs if row['root_index'] in OLD_CASES], root_pairs=root_pairs,
        root_receipts=roots, gain_orientation='baseline loss minus subject loss; positive favors subject',
        anchor128_online_all_exact=all(controls[f'{v}__pairs128']['anchor128_all_exact'] for v in variants),
        cheap_all_exact_original=all(controls[name]['cheap_exact_original'] for name in cheap),
        timing='All root cost.total_process_cpu is used once. Gate costs already overlap these totals and are '
               'not added again. Wall times are descriptive, not uncertainty or confidence estimates.',
        limitations=['Missing physical outcomes keep all 72 roots with [0, loss_cap] bounds; they are not imputed as failures.',
            'Conditional-bank empirical missing-outcome bounds are not population or anytime confidence bounds.',
            'CPU stopping can depend on trajectory complexity. Larger banks are new simulated draws, not new dates.',
            'No per-root method, budget or implementation oracle is reported.',
            'Novelty and route closure are evaluated separately from measured quality and throughput.'])
    write(HERE/'summary.json', result)
    write_controls_csv(result, quotas)
    draw(controls, variants, settings, cheap, quotas)
    print(dict(configurations=len(controls), physical_replays=result['cohort']['physical_replays'],
        anchor128_online_all_exact=result['anchor128_online_all_exact'],
        anchor128_reference_all_exact=reference['anchor128_all_exact'],
        implementations512_all_exact=reference['implementations512_all_exact']))


if __name__ == '__main__':
    main()
