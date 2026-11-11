"""Evaluate rider-known relative delays under fixed and actual-clock menus."""
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_run import (DiscreteLaw, ForecastEvaluator, data_inputs, law_summary,
                             load, summarize_endpoints, write)
from src.calibration_model import PublicForecasts
from src.calibration_relative import RelativeDelayModel

METHODS = ('evolving_incoming', 'evolving_public_incoming', 'frozen_public_mixture', 'relative_incoming')


def candidate_pairs(graph, actual_source=None, window=300):
    """Without actual_source the menu is a function of the timetable only."""
    pairs = []
    for incoming in graph.connections:
        transfer = graph.stops[incoming.arr_stop].min_transfer_time
        if actual_source is None:
            ready = incoming.arr_time + transfer
        else:
            before = actual_source[incoming.id]
            if before['outcome'] != 'MEASURED':
                continue
            ready = before['arrival']['time_seconds'] + transfer
        for outgoing in graph.get_connections_from(incoming.arr_stop):
            if incoming.trip_id != outgoing.trip_id and outgoing.pickup_allowed and abs(outgoing.dep_time - ready) <= window:
                pairs.append((incoming, outgoing))
    return pairs


def build_day(data, day, fit, frames):
    public = PublicForecasts(data['dates'][day]['connections'], frames)
    evaluator = ForecastEvaluator(data, day, fit, public)
    return evaluator, public


def training_pairs(data, dates, fit):
    result = {}
    for day in dates:
        evaluator, _ = build_day(data, day, fit, [])
        result[day] = [(evaluator.source[a.id], evaluator.source[b.id]) for a, b in candidate_pairs(evaluator.graph)
                       if evaluator.source[a.id]['outcome'] == 'MEASURED']
    return result


def evolving_after_arrival(evaluator, incoming, outgoing, now, use_public):
    """Expose the same rider observation to the original independent-trip law."""
    after = evaluator.source[outgoing.id]
    evaluator.evolving(after['id'], 'departure', now, use_public)
    receipt = evaluator.public.at(now).receipt if use_public else -1
    branch = evaluator.frame_models[receipt].copy()
    branch.observe_connection_event(incoming.id, 'arrival', now - incoming.arr_time)
    times, weights, cancel = branch.event_distribution(outgoing.id, 'departure', evaluator.priors)
    return DiscreteLaw(times, weights, cancel)


def evaluate(data, day, fit, frames, marginal, relative, protocol):
    evaluator, public = build_day(data, day, fit, frames)
    graph, source = evaluator.graph, evaluator.source
    primary = candidate_pairs(graph, window=protocol['scheduled_pair_window_seconds'])
    secondary = candidate_pairs(graph, source, protocol['scheduled_pair_window_seconds'])
    fixed_keys = {(a.id, b.id) for a, b in primary}
    actual_keys = {(a.id, b.id) for a, b in secondary}
    outputs = {}
    cache = {}
    for name, pairs in (('fixed_schedule', primary), ('actual_current', secondary)):
        counts = Counter({'planned_pairs': len(pairs)})
        rows = []
        for incoming, outgoing in pairs:
            before, after = source[incoming.id], source[outgoing.id]
            if before['outcome'] != 'MEASURED':
                counts['unobserved_incoming'] += 1
                continue
            key = incoming.id, outgoing.id
            if key not in cache:
                now = before['arrival']['time_seconds']
                ready = now + graph.stops[incoming.arr_stop].min_transfer_time
                outcome = after['departure']
                actual = outcome['time_seconds'] if outcome['status'] == 'REAL' else None
                known = outcome['status'] in ('REAL', 'CANCELED')
                caught = bool(actual is not None and ready <= actual <= ready + 720) if known else None
                mixed, features = marginal.predict(after, 'departure', now, public)
                laws = {'evolving_incoming': evolving_after_arrival(evaluator, incoming, outgoing, now, False),
                        'evolving_public_incoming': evolving_after_arrival(evaluator, incoming, outgoing, now, True),
                        'frozen_public_mixture': mixed, 'relative_incoming': relative.predict(before, after, now)}
                cache[key] = {'incoming': before['id'], 'outgoing': after['id'],
                              'incoming_route': before['route_id'], 'outgoing_route': after['route_id'],
                              'clock': now, 'ready': ready, 'scheduled': outgoing.dep_time,
                              'incoming_delay': now - incoming.arr_time, 'actual': actual,
                              'status': outcome['status'], 'caught': caught, 'features': features,
                              'methods': {method: law_summary(law, actual) for method, law in laws.items()},
                              'probabilities': {method: law.board_probability(ready) for method, law in laws.items()},
                              'scheduled_probability': float(ready <= outgoing.dep_time <= ready + 720)}
            row = cache[key]
            rows.append(row)
            counts[row['status']] += 1
        outputs[name] = {'coverage_counts': dict(counts), 'rows': rows}
    known_fixed = {(a.id, b.id) for a, b in primary if source[a.id]['outcome'] == 'MEASURED'}
    outputs['menu_comparison'] = {'fixed_known_incoming_pairs': len(known_fixed), 'actual_current_pairs': len(actual_keys),
                                  'intersection': len(known_fixed & actual_keys),
                                  'added_by_actual_clock': len(actual_keys - known_fixed),
                                  'removed_by_actual_clock': len(known_fixed - actual_keys)}
    return outputs


def metrics(rows, method, offset=0):
    probabilities = [row['probabilities'][method] for row in rows if row['caught'] is not None]
    truth = [row['caught'] for row in rows if row['caught'] is not None]
    p, y = np.asarray(probabilities), np.asarray(truth, dtype=float)
    return {'interval': summarize_endpoints(rows, method, offset),
            'probability': {'n': len(p), 'brier': float(np.mean((p - y) ** 2)),
                            'mean_predicted': float(p.mean()), 'actual_caught_fraction': float(y.mean()),
                            'signed_error': float(np.mean(p - y))}}


def select_menu(rows, method):
    menus = defaultdict(list)
    for row in rows:
        menus[row['incoming']].append(row)
    # Unknown outcomes remain candidates; they are censored only after selection.
    return [min(menu, key=lambda row: (-row['probabilities'][method], row['scheduled'], row['outgoing']))
            for menu in menus.values()]


def offsets_by_menu(calibration, dates, alpha, methods=METHODS):
    offsets, scores = {}, {}
    rank = math.ceil((len(dates) + 1) * (1 - alpha))
    for menu in ('fixed_schedule', 'actual_current'):
        offsets[menu], scores[menu] = {}, {}
        for method in methods:
            by_date = {}
            for day in dates:
                values = sorted(max(row['methods'][method]['lower'] - row['actual'],
                                    row['actual'] - row['methods'][method]['upper'])
                                for row in calibration[day][menu]['rows'] if row['actual'] is not None)
                by_date[day] = values[math.ceil(.9 * len(values)) - 1]
            offsets[menu][method] = max(0., sorted(by_date.values())[rank - 1])
            scores[menu][method] = by_date
    return offsets, scores


def summarize(output, offsets, methods=METHODS):
    result = {'menu_comparison': output['menu_comparison']}
    for menu in ('fixed_schedule', 'actual_current'):
        rows = output[menu]['rows']
        result[menu] = {'coverage_counts': output[menu]['coverage_counts'], 'methods': {}}
        for method in methods:
            selected = select_menu(rows, method)
            result[menu]['methods'][method] = {'raw': metrics(rows, method),
                'date_calibrated': metrics(rows, method, offsets[menu][method]),
                'greedy_probability_selected': {'n': len(selected),
                    'unresolved_selected': sum(row['caught'] is None for row in selected),
                    **metrics(selected, method, offsets[menu][method])}}
    return result


def main():
    frozen = json.loads((HERE / 'calibration_frozen_fit.json').read_text())
    original, fit = frozen['protocol'], frozen['evolving_fit']
    protocol = {'parameter_training_dates': original['parameter_training_dates'],
                'calibration_dates': original['calibration_dates'], 'development_dates': original['heldout_dates'],
                'scheduled_pair_window_seconds': 300, 'patience_seconds': 720,
                'route_pair_pseudocount': 20, 'kernel_sd_seconds': original['kernel_sd_seconds'],
                'methods': METHODS, 'selection': 'Primary pairs use scheduled incoming arrival + GTFS station transfer '
                'and different-trip departures within +/-300 seconds, before inspecting outcomes. Secondary pairs '
                'center the same menu window on the actual traveler-ready time. Unresolved outcomes remain menu candidates.',
                'information': 'Only the supplied incoming measured arrival and numeric forecasts received by that time. '
                'No other same-day actual endpoint is observed by the predictor.',
                'theory': 'A shared additive day location with equal incoming/outgoing loadings cancels in the '
                'relative delay. Relative residual exchangeability across selected date blocks is still required. '
                'The residual includes minus the incoming idiosyncratic error, so cancellation does not establish '
                'coverage conditional on every particular incoming delay.'}
    write(HERE / 'calibration_relative_protocol.json', protocol)
    data = data_inputs()
    frames = load(HERE / 'calibration_public_frames.json.gz')['dates']
    with (HERE / 'calibration_frozen_model.pkl').open('rb') as handle:
        marginal = pickle.load(handle)
    relative = RelativeDelayModel(marginal, training_pairs(data, protocol['parameter_training_dates'], fit))
    relative._cache.clear()
    with (HERE / 'calibration_relative_frozen_model.pkl').open('wb') as handle:
        pickle.dump(relative, handle)
    write(HERE / 'calibration_relative_frozen_fit.json', {'protocol': protocol,
        'training_pair_counts': dict(relative.training_counts),
        'route_pair_counts': [{'incoming_route': key[0], 'outgoing_route': key[1], 'n': len(value)}
                              for key, value in relative.samples.items() if key != 'pooled']})
    print('RELATIVE FIT FROZEN', dict(relative.training_counts), flush=True)
    outcomes = {}
    for day in protocol['calibration_dates'] + protocol['development_dates']:
        outcomes[day] = evaluate(data, day, fit, frames[day]['frames'], marginal, relative, protocol)
        print(day, {menu: value['coverage_counts'] for menu, value in outcomes[day].items() if 'coverage_counts' in value}, flush=True)
    offsets, scores = offsets_by_menu(outcomes, protocol['calibration_dates'], .1)
    write(HERE / 'calibration_relative_frozen_offsets.json', {'offset_seconds': offsets, 'date_scores': scores})
    summaries = {day: summarize(outcomes[day], offsets) for day in protocol['development_dates']}
    result = {'protocol': protocol, 'offset_seconds': offsets, 'date_scores': scores,
              'development': summaries, 'rows_by_date': outcomes}
    with gzip.open(HERE / 'calibration_relative_results.json.gz', 'wt') as handle:
        json.dump(result, handle, separators=(',', ':'), allow_nan=False)
    write(HERE / 'calibration_relative_summary.json', {key: result[key] for key in ('protocol', 'offset_seconds', 'development')})
    for day, value in summaries.items():
        for menu in ('fixed_schedule', 'actual_current'):
            print('RESULT', day, menu, {method: {'coverage': round(row['date_calibrated']['interval']['coverage'], 4),
                  'width': round(row['date_calibrated']['interval']['mean_width_seconds'], 1),
                  'brier': round(row['date_calibrated']['probability']['brier'], 5)}
                  for method, row in value[menu]['methods'].items()}, flush=True)


if __name__ == '__main__':
    main()
