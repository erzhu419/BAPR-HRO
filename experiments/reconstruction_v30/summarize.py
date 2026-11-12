"""Matched-model summaries of fixed-history diagnostic records."""

from collections import Counter, defaultdict
from statistics import mean


LAWS = ('original20', 'route_empirical20')
METRICS = ('first_arrival', 'remaining_wait', 'tail_wait', 'final_cost')


def average(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def numeric_scores(rows):
    return {label: average([row[key] for row in rows]) for label, key in
        [('bias', 'bias'), ('mae', 'absolute_error'), ('crps', 'crps'),
         ('coverage90', 'covered90'), ('width90', 'width90'),
         ('conditional_mass', 'conditional_mass')]}


def forecast_summary(rows):
    matched = defaultdict(dict)
    for row in rows:
        key = row['history_id'], row['role'], row['cid']
        if row['law'] in matched[key]:
            raise AssertionError('Repeated forecast would inflate paired evidence')
        matched[key][row['law']] = row
    paired = [group for group in matched.values() if all(law in group for law in LAWS)]
    metrics = {}
    for name in METRICS:
        pairs = [group for group in paired if all(group[law]['scores'].get(name) is not None for law in LAWS)]
        if any(group[LAWS[0]]['scores'][name]['actual'] != group[LAWS[1]]['scores'][name]['actual'] for group in pairs):
            raise AssertionError('Matched models must use the same observed label')
        by_law = {law: numeric_scores([group[law]['scores'][name] for group in pairs]) for law in LAWS}
        difference = {key: None if not pairs else by_law[LAWS[1]][key]-by_law[LAWS[0]][key]
                      for key in ('mae', 'crps', 'coverage90', 'width90')}
        metrics[name] = dict(n=len(pairs), by_law=by_law, empirical_minus_original=difference)
    binary_names = sorted({key for group in paired for law in LAWS for key in group[law]['scores'].get('binary', {})})
    binary = {}
    for name in binary_names:
        pairs = [group for group in paired if all(name in group[law]['scores'].get('binary', {}) for law in LAWS)]
        if any(group[LAWS[0]]['scores']['binary'][name]['actual'] !=
               group[LAWS[1]]['scores']['binary'][name]['actual'] for group in pairs):
            raise AssertionError('Matched binary models must share the same event label')
        binary[name] = dict(n=len(pairs), by_law={law: {
            key: average([group[law]['scores']['binary'][name][source] for group in pairs])
            for key, source in [('predicted_probability', 'probability'), ('actual_rate', 'actual'), ('brier', 'brier')]}
            for law in LAWS})
    projection_artifacts = {}
    for law in LAWS:
        assessed = [r for r in rows if r['law'] == law and r['role'] == 'board'
                    and r['forecast']['status'] == 'supported']
        masses, cases = [], []
        for row in assessed:
            distribution = row['forecast']['distributions']['first_arrival']
            mass = sum(p for t, p in zip(distribution['values'], distribution['probabilities'])
                       if t < row['clock'])
            masses.append(mass)
            if mass > 1e-12:
                cases.append(dict(forecast_id=row['forecast_id'], mass_before_public_clock=mass))
        projection_artifacts[law] = dict(cases=cases, n=len(cases), evaluated=len(assessed),
            max_mass=max(masses) if masses else None, mean_mass=average(masses),
            mean_scope='All supported BOARD forecasts in this summary; threshold for cases is 1e-12')
    return dict(forecast_records=len(rows), unique_history_actions=len(matched),
        paired_history_actions=len(paired), unique_histories=len({r['history_id'] for r in rows}),
        unique_roots=len({r['root_id'] for r in rows}),
        unique_day_trips=len({(r['day'], r['trip']) for r in rows}),
        roles=dict(Counter(group[next(iter(group))]['role'] for group in matched.values())),
        actual_statuses=dict(Counter(group[next(iter(group))]['actual']['status'] for group in matched.values())),
        support={law: dict(Counter(r['forecast']['status'] for r in rows if r['law'] == law)) for law in LAWS},
        unsupported_reasons={law: dict(Counter(r['forecast'].get('reason') for r in rows
            if r['law'] == law and r['forecast']['status'] != 'supported')) for law in LAWS},
        projection_methods={law: dict(Counter(r['forecast'].get('method') for r in rows
            if r['law'] == law and r['forecast']['status'] == 'supported')) for law in LAWS},
        paired_metrics=metrics, paired_binary=binary, projection_artifacts=projection_artifacts)


def pair_summary(rows, forecasts, histories):
    """Missing action losses retain the public-clock-to-deadline range."""
    board_only = [row for row in rows if row['target_cid'] is None]
    pairs = [row for row in rows if row['target_cid'] is not None]
    lower, upper, realized = [], [], []
    by_law = {law: [] for law in LAWS}
    for row in pairs:
        history = histories[row['history_id']]
        interval = history['clock'], history['root']['deadline']
        # Both models share physical labels. Use either stored action pair.
        ids = next(iter(row['by_law'].values()))
        board = forecasts[ids['board_id']]['actual']['cost']
        target = forecasts[ids['target_id']]['actual']['cost']
        board_range = (board, board) if board is not None else interval
        target_range = (target, target) if target is not None else interval
        lower.append(board_range[0]-target_range[1])
        upper.append(board_range[1]-target_range[0])
        actual_gain = row['actual_gain']
        if board is not None and target is not None:
            if actual_gain != board-target:
                raise AssertionError('Actual action difference must match fixed-route labels')
            realized.append(actual_gain)
        elif actual_gain is not None:
            raise AssertionError('An unresolved fixed-route label cannot yield a known action difference')
        for law, scores in row['by_law'].items():
            by_law[law].append(dict(nominal=scores['nominal_gain'], integrated=scores['integrated_gain'],
                                    actual=actual_gain))
    comparisons = {}
    for law, values in by_law.items():
        supported = [v for v in values if v['nominal'] is not None and v['integrated'] is not None]
        paired = [v for v in supported if v['actual'] is not None]
        # Positive gain selects target WAIT; ties retain the current BOARD.
        wait = lambda gain: gain > 1e-9
        comparisons[law] = dict(cells=len(values), supported=len(supported), paired_actual=len(paired),
            mean_nominal_gain=average([v['nominal'] for v in supported]),
            mean_integrated_gain=average([v['integrated'] for v in supported]),
            mean_integral_change=average([v['integrated']-v['nominal'] for v in supported]),
            ranking_flips=sum(wait(v['nominal']) != wait(v['integrated']) for v in supported),
            nominal_waits=sum(wait(v['nominal']) for v in supported),
            integrated_waits=sum(wait(v['integrated']) for v in supported),
            gain_prediction_bias=average([v['integrated']-v['actual'] for v in paired]),
            gain_prediction_mae=average([abs(v['integrated']-v['actual']) for v in paired]),
            realized_mean_of_nominal_pair_choice=average([v['actual'] if wait(v['nominal']) else 0. for v in paired]),
            realized_mean_of_integrated_pair_choice=average([v['actual'] if wait(v['integrated']) else 0. for v in paired]))
    return dict(records=len(rows), n=len(pairs), board_only=len(board_only),
        unique_histories=len({r['history_id'] for r in pairs}),
        paired_actual=len(realized), missing_actual=len(pairs)-len(realized),
        actual_wait_gain=dict(mean=average(realized), wins=sum(v > 0 for v in realized),
                              harms=sum(v < 0 for v in realized), ties=sum(v == 0 for v in realized)),
        all_pair_mean_gain_identification_interval=[average(lower), average(upper)],
        missing_cost_interval='[public clock, root deadline]; known costs retained', by_law=comparisons)


def summarize(result):
    forecasts = result['forecasts']
    forecast_by_id = {r['forecast_id']: r for r in forecasts}
    histories = {r['history_id']: r for r in result['histories']}
    scopes = {}
    for tag in ('first_offer', 'v29_disagreement'):
        rows = [r for r in forecasts if tag in r['tags']]
        pairs = [r for r in result['pairs'] if tag in r['tags']]
        group_keys = sorted({(r['day'], r['root_kind']) for r in rows})
        by_group = {}
        for day, kind in group_keys:
            selected = [r for r in rows if r['day'] == day and r['root_kind'] == kind]
            selected_pairs = [r for r in pairs if r['day'] == day and r['root_kind'] == kind]
            by_group[f'{day}/{kind}'] = dict(forecasts=forecast_summary(selected),
                pairs=pair_summary(selected_pairs, forecast_by_id, histories))
        scopes[tag] = dict(forecasts=forecast_summary(rows),
            by_role={role: forecast_summary([r for r in rows if r['role'] == role])
                     for role in sorted({r['role'] for r in rows})},
            pairs=pair_summary(pairs, forecast_by_id, histories), by_date_root_kind=by_group)
    return dict(complete=result['complete'], coverage=dict(Counter(r['status'] for r in result['coverage'])),
        unique_histories=len(histories), unique_roots=len({r['root_id'] for r in result['histories']}),
        dates=sorted({r['day'] for r in result['histories']}), scopes=scopes,
        interpretation='Development-only matched forecasts. First-offer and V29 scopes may overlap; '
            'history/action and same-vehicle counts are not independent sample sizes. '
            'Component timing scores condition on their recorded successful mass; final-cost scores '
            'include the failure atom at the deadline. Identification intervals describe missing labels, '
            'not sampling confidence intervals. Pair choices are fixed-action diagnostics, not a replay '
            'of an altered rolling policy. No statistical significance or new G2 claim.',
        wall=result.get('wall'))


if __name__ == '__main__':
    import gzip
    import json
    from pathlib import Path

    here = Path(__file__).resolve().parent
    with gzip.open(here/'results.json.gz', 'rt') as handle:
        result = json.load(handle)
    summary = summarize(result)
    (here/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(complete=summary['complete'], coverage=summary['coverage'],
        histories=summary['unique_histories'],
        paired_first_offer=summary['scopes']['first_offer']['pairs']['paired_actual'],
        first_offer_flips={law: row['ranking_flips'] for law, row in
            summary['scopes']['first_offer']['pairs']['by_law'].items()}), indent=2))
