"""One predeclared segment-conditioned candidate, with frozen policy controls."""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from experiments.reconstruction_v23.run_transfer_screen import TRAINING
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from
from experiments.reconstruction_v28.run import run_root, assert_same
from experiments.reconstruction_v30.forecast import forecast_board, forecast_target
from experiments.reconstruction_v30.metrics import score_forecast
from experiments.reconstruction_v31.model import SegmentDurationBelief, fit_segment_durations
from src.transfer_opportunity import TransferOpportunityScores


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'
BASELINE = HERE.parent/'reconstruction_v30'


def avg(values):
    return mean(values) if values else None


def forecast_summary(rows):
    output = dict(n=len(rows), support=dict(Counter(x['forecast']['status'] for x in rows)), metrics={}, binary={})
    for metric in ('first_arrival', 'remaining_wait', 'tail_wait', 'final_cost'):
        paired = [x for x in rows if x['scores'].get(metric) is not None and x['baseline_scores'].get(metric) is not None]
        output['metrics'][metric] = dict(n=len(paired),
            baseline_n=sum(x['baseline_scores'].get(metric) is not None for x in rows),
            candidate_n=sum(x['scores'].get(metric) is not None for x in rows), **{name: {
            k: avg([x[key][metric][k] for x in paired]) for k in ('bias', 'absolute_error', 'crps', 'covered90', 'width90')}
            for name, key in (('baseline', 'baseline_scores'), ('candidate', 'scores'))})
    names = sorted({key for x in rows for key in x['baseline_scores'].get('binary', {})})
    for metric in names:
        paired = [x for x in rows if metric in x['scores'].get('binary', {}) and metric in x['baseline_scores'].get('binary', {})]
        output['binary'][metric] = dict(n=len(paired),
            baseline_n=sum(metric in x['baseline_scores'].get('binary', {}) for x in rows),
            candidate_n=sum(metric in x['scores'].get('binary', {}) for x in rows),
            **{name: avg([x[key]['binary'][metric]['brier'] for x in paired])
            for name, key in (('baseline', 'baseline_scores'), ('candidate', 'scores'))})
    return output


def policy_summary(rows, baseline):
    output = {}
    for kind in ('origin', 'first_transfer'):
        selected = [x for x in rows if x['root_kind'] == kind]
        methods = {}
        for method in ('one_step_nominal', 'single_target_nominal'):
            pairs = [(baseline[x['root_id']]['outcomes'][method], x['outcomes'][method]) for x in selected]
            known = [(a, b) for a, b in pairs if a['loss'] is not None and b['loss'] is not None]
            methods[method] = dict(roots=len(pairs), paired=len(known),
                baseline_known=sum(a['loss'] is not None for a, b in pairs),
                candidate_known=sum(b['loss'] is not None for a, b in pairs),
                baseline_loss=avg([a['loss'] for a, b in known]), candidate_loss=avg([b['loss'] for a, b in known]),
                mean_gain=avg([a['loss']-b['loss'] for a, b in known]),
                wins=sum(a['loss'] > b['loss'] for a, b in known), harms=sum(a['loss'] < b['loss'] for a, b in known),
                ties=sum(a['loss'] == b['loss'] for a, b in known),
                baseline_arrivals=sum(a['arrived'] for a, b in pairs), candidate_arrivals=sum(b['arrived'] for a, b in pairs),
                candidate_statuses=dict(Counter(b['status'] for a, b in pairs)))
        output[kind] = methods
    return output


def admission(first, by_day, policy, all_forecasts):
    # These are frozen development retention conditions, not significance tests.
    quality = [first]+list(by_day.values())
    cost = all(x['metrics']['final_cost']['n'] > 0 and
        x['metrics']['final_cost']['candidate']['crps'] <= x['metrics']['final_cost']['baseline']['crps']+1e-9
        for x in quality)
    binary = first['binary']['legal_target_appears']
    control = policy['origin']['single_target_nominal']
    conditions = dict(support=all_forecasts['support'].get('supported', 0) == all_forecasts['n'],
        forecast_labels=all(x['candidate_n'] >= x['baseline_n']
            for group in [all_forecasts]+quality for kind in ('metrics', 'binary') for x in group[kind].values()),
        policy_labels=all(x['candidate_known'] >= x['baseline_known']
            for methods in policy.values() for x in methods.values()),
        total_cost_crps_overall_and_each_date=cost,
        legal_target_brier=binary['candidate'] is not None and binary['candidate'] <= binary['baseline']+1e-9,
        origin_single_cost=control['paired'] == 48 and control['mean_gain'] >= -1e-9,
        origin_single_arrivals=control['candidate_arrivals'] >= control['baseline_arrivals'])
    return dict(status='retain_development_candidate' if all(conditions.values()) else 'stop_segment_candidate',
        conditions=conditions, failed=[k for k, v in conditions.items() if not v],
        scope='One fixed-shrinkage development test. No tuning, further variable, independent confirmation or G2 admission.')


def main():
    started = perf_counter()
    selection = json.loads((SOURCE/'selection.json').read_text())
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    with gzip.open(BASELINE/'results.json.gz', 'rt') as handle:
        reference = json.load(handle)
    with gzip.open(HERE.parent/'reconstruction_v28/results.json.gz', 'rt') as handle:
        old_policy = {r['root_id']: r for r in json.load(handle)['results'] if r['law'] == 'route_empirical20'}
    training, _ = load_city(TRAINING)
    segment_fit = fit_segment_durations(training)
    (HERE/'segment_fit.json').write_text(json.dumps(segment_fit, indent=2)+'\n')
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    preparation = perf_counter()-started
    by_history = defaultdict(list)
    for row in reference['forecasts']:
        if row['law'] == 'route_empirical20':
            by_history[row['history_id']].append(row)
    histories = {x['history_id']: x for x in reference['histories']}
    forecasts, policies, walls = [], [], {}
    checks = dict(board_costs=0, unchanged_first_offer=0)
    for day, group in selection['by_day'].items():
        tick = perf_counter()
        graph, _, stop_ids, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        params = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
        model = SegmentDurationBelief(graph, duration_fit=fit['duration_fit'], segment_fit=segment_fit,
            stop_identities={v: k for k, v in stop_ids.items()}, public_occurrences=False,
            grid_step_seconds=20, **fit['increments'])
        walls[day] = dict(setup=perf_counter()-tick, forecast=0., validation=0., policy=0.)
        for hid, source_rows in by_history.items():
            history = histories[hid]
            if history['day'] != day:
                continue
            root = history['root']
            compiler = TransferOpportunityScores(graph, model, params, root['deadline'], options_from(root['full_options']))
            for previous in source_rows:
                role, cid, clock = previous['role'], previous['cid'], history['clock']
                tick = perf_counter()
                forecast = forecast_board(compiler, cid, clock) if role == 'board' else forecast_target(
                    compiler, cid, history['entry'], clock, history['ready'])
                walls[day]['forecast'] += perf_counter()-tick
                tick = perf_counter()
                if role == 'board' and forecast['status'] == 'supported':
                    old = compiler.board_score(cid, clock)
                    if old is None or abs(old.exact-forecast['cost']) > 1e-6:
                        raise AssertionError('Segment distribution and backward integral disagree')
                    checks['board_costs'] += 1
                walls[day]['validation'] += perf_counter()-tick
                scores = score_forecast(forecast, previous['actual'], root['deadline'], previous['reference_tail'])
                forecasts.append(dict(history_id=hid, root_id=root['root_id'], day=day, root_kind=root['root_kind'],
                    role=role, cid=cid, tags=history['tags'], actual=previous['actual'], baseline_scores=previous['scores'],
                    baseline_cost=previous['forecast']['cost'], forecast=forecast, scores=scores))
            print('forecast', hid, len(source_rows), flush=True)
        for root in group['roots']:
            tick = perf_counter()
            row = run_root(graph, metadata, model, params, root)
            walls[day]['policy'] += perf_counter()-tick
            assert_same(row['outcomes']['full_first_offer'], old_policy[root['root_id']]['outcomes']['full_first_offer'], root['root_id'])
            checks['unchanged_first_offer'] += 1
            policies.append(row)
            print('policy', root['root_id'], {m: row['outcomes'][m]['loss'] for m in ('one_step_nominal', 'single_target_nominal')}, flush=True)
    first_rows = [r for r in forecasts if 'first_offer' in r['tags']]
    first = forecast_summary(first_rows)
    by_day = {day: forecast_summary([r for r in first_rows if r['day'] == day]) for day in selection['by_day']}
    policy = policy_summary(policies, old_policy)
    all_forecasts = forecast_summary(forecasts)
    summary = dict(all_forecasts=all_forecasts, first_offer=first, first_offer_by_day=by_day,
        first_offer_by_role={role: forecast_summary([r for r in first_rows if r['role'] == role]) for role in ('board', 'target')},
        v29=forecast_summary([r for r in forecasts if 'v29_disagreement' in r['tags']]),
        policy=policy, decision=admission(first, by_day, policy, all_forecasts), checks=checks,
        wall=dict(training_and_loading=preparation, by_day=walls, total=perf_counter()-started))
    with gzip.open(HERE/'results.json.gz', 'wt') as handle:
        json.dump(dict(complete=True, forecasts=forecasts, policies=policies, summary=summary), handle, separators=(',', ':'), allow_nan=False)
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (HERE/'decision.json').write_text(json.dumps(summary['decision'], indent=2)+'\n')
    print(json.dumps(dict(decision=summary['decision'], checks=checks, policy=policy, wall=summary['wall']), indent=2), flush=True)


if __name__ == '__main__':
    main()
