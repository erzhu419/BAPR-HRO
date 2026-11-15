"""P1 attribution and complete-denominator new-domain quality/cost comparisons."""
import argparse
from collections import Counter
import csv
from math import isfinite
from pathlib import Path
from statistics import mean

from experiments.decision_gap_v1.panel import (
    NEW_LABELS, build_panel, legacy_extension, public_state, read, state_identity, write,
)
from experiments.reconstruction_v44.matched import choose
from experiments.reconstruction_v50.summarize import bounds, first_divergence

HERE = Path(__file__).resolve().parent
LABELS = ('old_old_cheap', 'old_new_cheap', *NEW_LABELS)
CHEAP_LABELS = NEW_LABELS[:6]
TRACE_FIELDS = ('kind', 'cid', 'stop', 'entry', 'ready', 'clock', 'choice', 'visible',
                'observed', 'observed_visible', 'observation_cids', 'public_history')


def event_signature(outcome):
    # JSON canonicalization reconciles in-memory tuples/int keys with saved JSON.
    return read_json([{key:event[key] for key in TRACE_FIELDS if key in event}
                      for event in outcome['trace']])


def read_json(value):
    import json
    return json.loads(json.dumps(value))


def validate_old_replay(row, old):
    validation = dict(loss_equal=row['outcome']['loss'] == old['outcome']['loss'],
        status_equal=row['outcome']['status'] == old['outcome']['status'],
        actions_equal=row['actions'] == old['actions'],
        public_physical_trace_equal=event_signature(row['outcome']) == event_signature(old['outcome']),
        observation_contract_equal=row['observation'] == old['observation'])
    if not all(validation.values()):
        raise AssertionError(f"Frozen V51 cheap replay changed at root {row['root_index']}: {validation}")
    return validation


def validate_rows(rows, index):
    cheap = rows['new_cheap']
    for label in LABELS:
        row = rows[label]
        for key in ('root_id', 'day', 'origin', 'goal', 'loss_cap'):
            if row[key] != cheap[key]:
                raise AssertionError(f'Root context changed: {index}/{label}/{key}')
        if row['root_index'] != index or row['label'] != label:
            raise AssertionError(f'Receipt identity changed: {index}/{label}')
        if label != 'old_old_cheap' and row['observation'] != cheap['observation']:
            raise AssertionError(f'New observation contract changed: {index}/{label}')
        cost = row['cost']
        for metric in ('process_cpu', 'wall'):
            values = (cost[metric], cost['shared_menu'][metric], cost[metric+'_including_menu'])
            if not all(isfinite(value) and value >= 0 for value in values):
                raise AssertionError('Nonfinite or negative computation charge')
            if abs(values[0]+values[1]-values[2]) > 1e-9:
                raise AssertionError('Menu preparation missing from total query cost')


def paired_rows(banks, left, right):
    result = []
    for rows in banks:
        a, b = rows[left], rows[right]
        ab, bb = bounds(a), bounds(b)
        low, high = ab[0]-bb[1], ab[1]-bb[0]
        complete = a['outcome']['loss'] is not None and b['outcome']['loss'] is not None
        status = 'win' if low > 0 else 'harm' if high < 0 else 'tie' if low == high == 0 else 'unresolved'
        result.append(dict(root_index=a['root_index'], day=a['day'], baseline=left, challenger=right,
            baseline_loss=a['outcome']['loss'], challenger_loss=b['outcome']['loss'],
            baseline_status=a['outcome']['status'], challenger_status=b['outcome']['status'],
            gain_lower=low, gain_upper=high, complete_pair=complete, classification=status,
            cpu_delta=b['cost']['process_cpu_including_menu']-a['cost']['process_cpu_including_menu'],
            wall_delta=b['cost']['wall_including_menu']-a['cost']['wall_including_menu'],
            actions_equal=a['actions'] == b['actions']))
    return result


def comparison(banks, left, right):
    pairs = paired_rows(banks, left, right)
    complete = [row for row in pairs if row['complete_pair']]
    counts = Counter(row['classification'] for row in pairs)
    return dict(roots=len(pairs), complete_pairs=len(complete), unresolved_pairs=len(pairs)-len(complete),
        wins=counts['win'], harms=counts['harm'], ties=counts['tie'], undecided=counts['unresolved'],
        all_root_mean_gain_bounds=[mean(row['gain_lower'] for row in pairs),
                                   mean(row['gain_upper'] for row in pairs)],
        complete_pair_mean_gain=mean(row['gain_lower'] for row in complete) if complete else None,
        mean_cpu_delta=mean(row['cpu_delta'] for row in pairs),
        mean_wall_delta=mean(row['wall_delta'] for row in pairs),
        action_equal_roots=sum(row['actions_equal'] for row in pairs),
        positive_roots=[row['root_index'] for row in pairs if row['classification']=='win'],
        negative_roots=[row['root_index'] for row in pairs if row['classification']=='harm'],
        unresolved_roots=[row['root_index'] for row in pairs if row['classification']=='unresolved'],
        orientation=f'{left} loss minus {right} loss')


def audit_gates(banks, worlds):
    same_full, modes, fallback, unfinished = {}, Counter(), Counter(), Counter()
    comparisons = total = 0
    for rows in banks:
        for label in NEW_LABELS:
            row = rows[label]
            for sequence, gate in enumerate(row['gates']):
                total += 1
                if gate['future_messages'] != 0 or gate['count'] != worlds:
                    raise AssertionError('Rollout information or sample budget changed')
                if set(gate['rows']) != {'alight', 'continue'} or any(
                        len(values) != worlds for values in gate['rows'].values()):
                    raise AssertionError('Unresolved worlds were removed from the bank')
                low, high = gate['paired']['full_bank_mean_gain_interval']
                if not all(isfinite(value) for value in (low, high)) or low > high+1e-9:
                    raise AssertionError('Invalid action-difference interval')
                if gate['selected_action'] != choose(gate, gate['base_action']):
                    raise AssertionError('Rollout action violates its frozen selection rule')
                state = row['states'][gate['state_index']]
                if state['gate_index'] != sequence or state['selected_action'] != gate['selected_action']:
                    raise AssertionError('Gate does not match its executed public decision')
                modes[label, gate['used_kernel']] += 1
                if gate['fallback'] is not None:
                    fallback[gate['fallback']] += 1
                unfinished[label] += int(gate['completed_pairs'] < worlds)
                if gate['used_kernel'] == 'full_common_observation_pi' and gate['completed_pairs'] == worlds:
                    key = state_identity(public_state(state, row['day']))
                    science = {name:gate[name] for name in ('rows', 'paired', 'common_tape')}
                    if key in same_full:
                        if same_full[key] != science:
                            raise AssertionError('Same public-state full rollout bank differs across policies')
                        comparisons += 1
                    else:
                        same_full[key] = science
    return dict(total=total, same_state_full128_comparisons=comparisons,
        kernels=[dict(policy=label, kernel=kernel, count=count) for (label,kernel),count in modes.items()],
        fallback_counts=dict(fallback), cpu_unfinished_by_policy=dict(unfinished))


def aggregate_rows(banks, label):
    rows = [bank[label] for bank in banks]
    intervals = [bounds(row) for row in rows]
    complete = [row['outcome']['loss'] for row in rows if row['outcome']['loss'] is not None]
    return dict(roots=len(rows), complete=len(complete), unresolved=len(rows)-len(complete),
        all_root_mean_loss_bounds=[mean(x[0] for x in intervals), mean(x[1] for x in intervals)],
        complete_mean_loss=mean(complete) if complete else None,
        mean_cpu=mean(row['cost']['process_cpu_including_menu'] for row in rows),
        mean_wall=mean(row['cost']['wall_including_menu'] for row in rows),
        total_states=sum(len(row['states']) for row in rows),
        total_gates=sum(len(row['gates']) for row in rows),
        unresolved_roots=[row['root_index'] for row in rows if row['outcome']['loss'] is None])


def residual_diagnostic(banks):
    """Keep each fixed baseline separate; their per-root minimum is not a policy."""
    result = {}
    for label in ('new_matched128', 'new_full128'):
        against = {cheap:comparison(banks, cheap, label) for cheap in CHEAP_LABELS}
        positive = {cheap:set(row['positive_roots']) for cheap,row in against.items()}
        result[label] = dict(versus_each_fixed_cheap_policy=against,
            strictly_better_than_every_cheap_policy_roots=sorted(set.intersection(*positive.values())),
            interpretation='The intersection describes residual archive cases, not a deployable per-root baseline oracle')
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(directory=HERE, *, make_plot=True):
    directory = Path(directory)
    protocol = read(directory/'contract.json')
    if tuple(protocol['labels']) != LABELS or protocol['roots'] != 72:
        raise AssertionError('P1 frozen labels or 72-root denominator changed')
    banks, records, flat, regressions = [], [], [], []
    for index in range(protocol['roots']):
        rows = {label:read(directory/'roots'/f'{index:03}_{label}.json.gz') for label in LABELS}
        validate_rows(rows, index)
        old = read(HERE.parent/'reconstruction_v51/roots'/f'{index:03}_cheap.json.gz')
        regressions.append(dict(root_index=index, **validate_old_replay(rows['old_old_cheap'], old)))
        banks.append(rows)
        cheap = rows['new_cheap']
        results = {}
        for label,row in rows.items():
            low, high = bounds(row)
            results[label] = dict(loss=row['outcome']['loss'], status=row['outcome']['status'],
                loss_bounds=[low,high], cost=row['cost'], states=len(row['states']), gates=len(row['gates']),
                first_divergence_from_new_cheap=first_divergence(cheap['actions'], row['actions']))
            flat.append(dict(root_index=index, day=row['day'], origin=row['origin'], goal=row['goal'],
                label=label, loss=row['outcome']['loss'], status=row['outcome']['status'],
                loss_lower=low, loss_upper=high, cpu=row['cost']['process_cpu'], wall=row['cost']['wall'],
                cpu_including_menu=row['cost']['process_cpu_including_menu'],
                wall_including_menu=row['cost']['wall_including_menu'],
                states=len(row['states']), gates=len(row['gates'])))
        records.append(dict(root_index=index, day=cheap['day'], results=results))
    aggregate = {label:aggregate_rows(banks, label) for label in LABELS}
    pairs = [('old_old_cheap','old_new_cheap'), ('old_new_cheap','new_cheap'),
             ('old_old_cheap','new_cheap')]
    # Full same-domain pair matrix, including the matched/full comparison.
    pairs += [(left,right) for i,left in enumerate(NEW_LABELS) for right in NEW_LABELS[i+1:]]
    comparisons = {f'{left}__{right}':comparison(banks,left,right) for left,right in pairs}
    all_pairs = [row for left,right in pairs for row in paired_rows(banks,left,right)]
    gate_audit = audit_gates(banks, protocol['rollout_limits']['worlds_per_query'])
    panel = build_panel(banks)
    legacy = legacy_extension()
    historical = []
    for index in (4,62):
        old_full = read(HERE.parent/'reconstruction_v51/roots'/f'{index:03}_full128.json.gz')
        historical.append(dict(root_index=index, old_full128_loss=old_full['outcome']['loss'],
            fixed_policy_comparisons=[dict(policy=label, new_loss=banks[index][label]['outcome']['loss'],
                new_loss_bounds=bounds(banks[index][label]),
                matches_or_improves_old_full128=(bounds(banks[index][label])[1] <= old_full['outcome']['loss']))
                for label in CHEAP_LABELS]))
    result = dict(stage='decision_gap_v1', archive_calls=len(banks)*len(LABELS), aggregate=aggregate,
        attribution={name:comparisons[f'{left}__{right}'] for name,(left,right) in zip(
            ('observation_change', 'menu_change_given_new_observation', 'total_domain_change'), pairs[:3])},
        same_domain_comparisons={key:value for key,value in comparisons.items() if key.startswith('new_')},
        records=records, old_v51_cheap_regression=dict(roots=len(regressions), rows=regressions),
        gate_audit=gate_audit, state_panel=panel['summary'],
        legacy_panel=dict(original_queries=88, supplemental_queries=1, total_queries=89),
        historical_signals=dict(scope='Cross-domain descriptive context only', rows=historical),
        residual_diagnostic=residual_diagnostic(banks),
        interpretation='Loss ranges retain missing endpoints; these are archive-case comparisons, not confidence intervals or independent confirmation',
        g2_passed=False, new_solver_admitted=False)
    write(directory/'new_panel.json.gz', panel)
    write(directory/'legacy_panel_extension.json.gz', legacy)
    write(directory/'summary.json', result)
    write_csv(directory/'paired_journeys.csv', flat)
    write_csv(directory/'paired_comparisons.csv', all_pairs)
    if make_plot:
        plot(aggregate, directory/'quality_time.png')
    return result


def plot(aggregate, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5))
    for i,label in enumerate(NEW_LABELS):
        row = aggregate[label]
        low,high = row['all_root_mean_loss_bounds']
        x, color = row['mean_cpu'], f'C{i}'
        ax.vlines(x, low, high, color=color)
        ax.plot([x,x], [low,high], marker='_' if row['unresolved'] else 'o',
                linestyle='none', color=color)
        suffix = f" ({row['unresolved']} unresolved)" if row['unresolved'] else ''
        offset = (-8, -16) if label == 'new_cheap' else (7, 8 if i%2 else -16)
        ax.annotate(label.removeprefix('new_')+suffix, (x,low),
            xytext=offset, ha='right' if label == 'new_cheap' else 'left',
            textcoords='offset points', fontsize=8, color=color)
    ax.set(xlabel='Mean whole-journey CPU including menu preparation (s)',
           ylabel='All-root mean truncated loss / bounds (s)',
           title='P1: same new menu and public observations, all 72 roots', xscale='log')
    ax.margins(x=.15,y=.15)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=HERE)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    result = summarize(args.directory, make_plot=not args.no_plot)
    print({key:result[key] for key in ('archive_calls', 'attribution', 'state_panel', 'gate_audit')})
