"""Summarize fixed-control physical replays, without selecting a rootwise oracle."""

from collections import Counter
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PUBLIC, LOCAL = 'public_messages', 'own_and_local_only'


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else path.open()) as handle:
        return json.load(handle)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def mean(values):
    return float(np.mean(values)) if values else None


def numeric(values):
    return dict(count=len(values), total=float(sum(values)), mean=mean(values),
                median=float(np.median(values)) if values else None,
                minimum=min(values, default=None), maximum=max(values, default=None))


def status_class(outcome):
    status = outcome['status']
    if status.startswith('unsupported_'):
        return 'unsupported'
    if status.startswith('unresolved:'):
        return 'missing_or_ambiguous'
    return 'arrived' if status == 'arrived' else 'failure'


def arm_summary(rows):
    return dict(denominator=len(rows), scored=sum(r['outcome']['loss'] is not None for r in rows),
        loss_seconds=numeric([r['outcome']['loss'] for r in rows if r['outcome']['loss'] is not None]),
        status_counts=dict(Counter(r['outcome']['status'] for r in rows)),
        status_classes=dict(Counter(status_class(r['outcome']) for r in rows)),
        cold_total_process_cpu_seconds=numeric([r['cost']['total_process_cpu'] for r in rows]),
        cold_total_wall_seconds=numeric([r['cost']['total_wall'] for r in rows]))


def action(row):
    return row['kind'], row.get('cid', row.get('stop')), row['clock'], row.get('choice')


def first_divergence(subject, baseline):
    """Compare complete public actions and their shared prefix histories."""
    a, b = subject['trace'], baseline['trace']
    equal_history, history_pairs, changed = True, 0, None
    for index, (left, right) in enumerate(zip(a, b)):
        if left['kind'] in ('platform', 'onboard') and right['kind'] in ('platform', 'onboard'):
            same_history = left['public_history'] == right['public_history']
            equal_history &= same_history
            history_pairs += 1
        else:
            same_history = None
        if action(left) != action(right):
            changed = dict(trace_index=index, subject_action=action(left), baseline_action=action(right),
                same_decision_context=action(left)[:3] == action(right)[:3],
                same_public_history=same_history)
            break
    if changed is None and len(a) != len(b):
        index = min(len(a), len(b))
        changed = dict(trace_index=index,
            subject_action=action(a[index]) if index < len(a) else None,
            baseline_action=action(b[index]) if index < len(b) else None,
            same_decision_context=False, same_public_history=None, truncated_trace=True)
    return dict(actions_identical=changed is None, first=changed,
                common_prefix_history_equal=equal_history, compared_history_pairs=history_pairs)


def paired(subjects, baselines):
    gains, cpu, wall, missing, action_differences, history_mismatches = [], [], [], [], 0, 0
    rows = []
    for root_index, a, b in zip(subjects['indices'], subjects['rows'], baselines):
        left, right = a['outcome'], b['outcome']
        cpu.append(a['cost']['total_process_cpu']-b['cost']['total_process_cpu'])
        wall.append(a['cost']['total_wall']-b['cost']['total_wall'])
        divergence = first_divergence(left, right)
        action_differences += not divergence['actions_identical']
        history_mismatches += not divergence['common_prefix_history_equal']
        gain = None if left['loss'] is None or right['loss'] is None else right['loss']-left['loss']
        if gain is None:
            missing.append(dict(root_index=root_index, subject_status=left['status'], baseline_status=right['status']))
        else:
            gains.append(gain)
        rows.append(dict(root_index=root_index, gain_seconds=gain,
            subject_loss=left['loss'], baseline_loss=right['loss'],
            subject_status=left['status'], baseline_status=right['status'],
            cpu_increment=cpu[-1], wall_increment=wall[-1],
            actions_identical=divergence['actions_identical'],
            common_prefix_history_equal=divergence['common_prefix_history_equal']))
    return dict(denominator=len(baselines), scored_pairs=len(gains),
        positive=sum(x > 0 for x in gains), negative=sum(x < 0 for x in gains), tie=sum(x == 0 for x in gains),
        gain_seconds=numeric(gains), subject_minus_baseline_cpu_seconds=numeric(cpu),
        subject_minus_baseline_wall_seconds=numeric(wall), missing_pairs=missing,
        public_action_divergences=action_differences, common_prefix_history_mismatches=history_mismatches,
        pairs=rows)


def receipt_link(root, arm, method, trace_index):
    row = root['rows'][arm][method]['outcome']['trace']
    event = row[trace_index] if trace_index < len(row) else None
    link = dict(file=f'roots/{root["root_index"]:03}.json.gz',
                json_location=f'rows.{arm}.{method}.outcome.trace[{trace_index}]')
    if event is None:
        return link
    message = event.get('message_receipt', {})
    link.update(message_frame_range=[message.get('first_frame_index'), message.get('next_frame_index')],
                message_source=f'message_days/{root["day"]}.json.gz',
                latest_consumed_publications=[r.get('publication', {}) for r in message.get('receipts', [])[-4:]])
    return link


def divergence_catalog(roots, methods, baseline):
    records, references, candidates = [], [], []
    for root in roots:
        for arm in (PUBLIC, LOCAL):
            for method in methods:
                if method == baseline:
                    continue
                subject = root['rows'][arm][method]['outcome']
                control = root['rows'][arm][baseline]['outcome']
                comparison = first_divergence(subject, control)
                if comparison['actions_identical']:
                    continue
                first = comparison['first']
                row = dict(kind='same_information_policy', root_index=root['root_index'], root_id=root['root_id'],
                    day=root['day'], arm=arm, method=method, baseline=baseline, **comparison,
                    gain_seconds=None if subject['loss'] is None or control['loss'] is None else control['loss']-subject['loss'],
                    subject_source=receipt_link(root, arm, method, first['trace_index']),
                    baseline_source=receipt_link(root, arm, baseline, first['trace_index']))
                records.append(row)
                if arm == PUBLIC and first['same_decision_context'] and first['same_public_history']:
                    event = subject['trace'][first['trace_index']]
                    if event['kind'] in ('platform', 'onboard'):
                        candidates.append((root['day'], event['clock'], root['root_index'], method, row, event))
        for method in methods:
            subject, control = (root['rows'][arm][method]['outcome'] for arm in (PUBLIC, LOCAL))
            comparison = first_divergence(subject, control)
            if not comparison['actions_identical']:
                first = comparison['first']
                records.append(dict(kind='same_policy_information', root_index=root['root_index'],
                    root_id=root['root_id'], day=root['day'], method=method,
                    subject_arm=PUBLIC, baseline_arm=LOCAL, **comparison,
                    gain_seconds=None if subject['loss'] is None or control['loss'] is None else control['loss']-subject['loss'],
                    subject_source=receipt_link(root, PUBLIC, method, first['trace_index']),
                    baseline_source=receipt_link(root, LOCAL, method, first['trace_index'])))
    histories = []
    for _, _, _, _, row, event in sorted(candidates, key=lambda item: item[:4]):
        history = event['public_history']
        if any(history == old for old in histories):
            continue
        histories.append(history)
        references.append(dict(root_index=row['root_index'], root_id=row['root_id'], day=row['day'],
            method=row['method'], baseline=row['baseline'], kind=event['kind'], clock=event['clock'],
            first=row['first'], subject_source=row['subject_source'], baseline_source=row['baseline_source'],
            public_history=history, message_state=event['message_state']))
        if len(references) == 4:
            break
    return dict(selection_rule='At most four earliest (date, decision clock, root index, method) '
        'distinct public histories among same-information action disagreements. No gain-based selection.',
        reference_count=len(references), reference_states=references, divergences=records,
        compared_root_method_pairs=len(roots)*(2*(len(methods)-1)+len(methods)))


def draw(roots, methods):
    colors = plt.get_cmap('tab10').colors
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, arm in zip(axes, (PUBLIC, LOCAL)):
        unresolved = 0
        for index, method in enumerate(methods):
            rows = [root['rows'][arm][method] for root in roots]
            complete = [r for r in rows if r['outcome']['loss'] is not None]
            unresolved += len(rows)-len(complete)
            x = [r['cost']['total_process_cpu'] for r in complete]
            y = [r['outcome']['loss'] for r in complete]
            ax.scatter(x, y, color=colors[index], s=16, alpha=.42, label=method)
            if complete:
                ax.scatter([mean(x)], [mean(y)], color=colors[index], marker='X', s=90,
                           edgecolor='black', linewidth=.5)
        ax.set_xscale('log')
        ax.set_xlabel('Complete cold process CPU (seconds, log scale)')
        ax.set_title(f'{arm}\n{len(roots)} fixed roots; {unresolved} unscored runs')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Truncated journey loss (seconds)')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .99), ncol=3, frameon=False)
    figure.text(.5, .015, 'Small dots: individual root/control. X: fixed-control mean. '
        'Warm identical-history repeats are excluded. CPU does not advance physical clocks.', ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .06, 1, .85))
    figure.savefig(HERE/'quality_cpu.png', dpi=180)
    plt.close(figure)


def main():
    selection, protocol = read(HERE/'selection.json'), read(HERE/'protocol.json')
    methods, baseline = list(protocol['methods']), protocol['primary']
    roots = []
    missing = []
    for index, selected in enumerate(selection['selected']):
        path = HERE/'roots'/f'{index:03}.json.gz'
        if not path.exists():
            missing.append(index)
            continue
        root = read(path)
        if root['root_index'] != index or root['root_id'] != selected['root_id']:
            raise ValueError(f'Root receipt does not match frozen selection: {path}')
        roots.append(root)
    if missing:
        raise SystemExit(f'Awaiting {len(missing)}/{len(selection["selected"])} fixed roots: {missing}')
    by_day = {day: [r for r in roots if r['day'] == day] for day in protocol['main_dates']}
    aggregate, matrix, information, day_summaries = {}, {}, {}, {}
    groups = {'all': roots, **by_day}
    for group, chosen in groups.items():
        indices = [r['root_index'] for r in chosen]
        arm_rows, comparisons, information_rows = {}, {}, {}
        for arm in (PUBLIC, LOCAL):
            arm_rows[arm] = {method: arm_summary([r['rows'][arm][method] for r in chosen]) for method in methods}
            comparisons[arm] = {}
            for method in methods:
                comparisons[arm][method] = {}
                for control in methods:
                    value = paired(dict(indices=indices, rows=[r['rows'][arm][method] for r in chosen]),
                                   [r['rows'][arm][control] for r in chosen])
                    if control != baseline:
                        value.pop('pairs')  # Keep the full paired ledger for the declared primary baseline.
                    comparisons[arm][method][control] = value
        for method in methods:
            information_rows[method] = paired(dict(indices=indices,
                rows=[r['rows'][PUBLIC][method] for r in chosen]), [r['rows'][LOCAL][method] for r in chosen])
        if group == 'all':
            aggregate, matrix, information = arm_rows, comparisons, information_rows
        else:
            day_summaries[group] = dict(denominator=len(chosen), controls=arm_rows,
                policy_vs_declared_combined={arm: {method: comparisons[arm][method][baseline]
                    for method in methods} for arm in (PUBLIC, LOCAL)}, information_gains=information_rows)
    warm = [dict(root_index=r['root_index'], day=r['day'], arm=arm, method=method,
            **r['rows'][arm][method]['cost']['identical_history_warm'])
        for r in roots for arm in (PUBLIC, LOCAL) for method in methods
        if 'identical_history_warm' in r['rows'][arm][method]['cost']]
    summary = dict(scope='Fixed controls on a frozen development discovery cohort; no rootwise oracle baseline.',
        denominator=len(selection['selected']), roots_completed=len(roots), root_runs=len(roots)*2*len(methods),
        method_order=methods, declared_primary=baseline,
        sign_convention='Positive gain = baseline loss minus subject loss. Positive compute increment = subject costs more.',
        timing_scope='Cold process CPU and wall include constructor, menu setup, posterior updates, candidate/tail '
            'compilation, choice, execution and trace. Common file/graph preparation is separate. '
            'Physical clocks are logical replay clocks; computation does not shift boarding.',
        controls=aggregate, same_information_policy_matrix=matrix, same_method_information_gains=information,
        by_day=day_summaries, selection_coverage=selection['by_day'],
        selection_contract='Static menus precede outcome inspection. Eligibility uses actual endpoint '
            'coverage and the observed own-arrival clock to require a message received before the decision. '
            'Root selection does not use realized policy loss or predicted advantage.',
        missing_source_eligibility_count=sum(not e['complete_labels'] for e in selection['eligibility']),
        static_root_candidates=sum(day['qualifying_roots'] for day in selection['by_day'].values()),
        preparation_wall_seconds=numeric([r['preparation_wall'] for r in roots]),
        identical_history_warm=dict(scope='Root0 only: same already computed history replayed with the same '
            'model/planner. This measures repeated-history reuse, not acceleration on new journeys.', rows=warm),
        limitations=['Discovery dates were used for development; these are not independent confirmation estimates.',
            'Loss means use scored pairs; all missing and unsupported runs remain in the fixed denominator.',
            'Logical replay CPU comparisons do not establish real-time feasibility.'])
    catalog = divergence_catalog(roots, methods, baseline)
    summary['divergence_catalog'] = dict(file='divergence_catalog.json',
        total_disagreements=len(catalog['divergences']), conditional_reference_states=catalog['reference_count'])
    write(HERE/'summary.json', summary)
    write(HERE/'divergence_catalog.json', catalog)
    draw(roots, methods)
    print(json.dumps(dict(roots=len(roots), runs=summary['root_runs'],
        divergence_records=len(catalog['divergences']), references=catalog['reference_count'])))


if __name__ == '__main__':
    main()
