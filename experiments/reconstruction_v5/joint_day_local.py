"""Prior endpoints and fixed every-twentieth incoming-known transfer pair."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_run import DiscreteLaw, law_summary, summarize_endpoints, write
from calibration_relative_run import candidate_pairs, training_pairs
from joint_day_run import DATES, inputs
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from src.evolving_trip_belief import EvolvingTripBelief
from src.joint_day_belief import JointDayBelief

METHODS = ('independent', 'joint_day')


def setup(data, day, fit):
    graph, _, _, _, source, _ = dated_graph(data, day, 'gtfs_station_max')
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    beliefs = {'independent': EvolvingTripBelief(graph, public_occurrences=False, **fit['increments']),
               'joint_day': JointDayBelief(graph, public_occurrences=False, **fit['increments'])}
    return graph, source, parameters, beliefs


def endpoints(data, day, fit):
    graph, source, parameters, beliefs = setup(data, day, fit)
    rows = []
    for c in graph.connections:
        for kind in ('departure', 'arrival'):
            outcome = source[c.id][kind]
            if outcome['status'] != 'REAL':
                continue
            actual = outcome['time_seconds']
            estimates = {}
            for name, belief in beliefs.items():
                law = DiscreteLaw(*belief.event_distribution(c.id, kind, parameters))
                estimates[name] = law_summary(law, actual)
            rows.append({'source_connection': source[c.id]['id'], 'kind': kind, 'actual': actual, 'methods': estimates})
    return rows


def transfers(data, day, fit):
    graph, source, parameters, beliefs = setup(data, day, fit)
    ordered = sorted(candidate_pairs(graph), key=lambda pair: (pair[0].dep_time, pair[0].id, pair[1].dep_time, pair[1].id))
    fixed = ordered[::20]
    rows, censored = [], Counter()
    for incoming, outgoing in fixed:
        before, after = source[incoming.id], source[outgoing.id]
        if before['outcome'] != 'MEASURED':
            censored['unobserved_incoming'] += 1
            continue
        outcome = after['departure']
        now = before['arrival']['time_seconds']
        ready = now + graph.stops[incoming.arr_stop].min_transfer_time
        actual = outcome['time_seconds'] if outcome['status'] == 'REAL' else None
        caught = bool(actual is not None and ready <= actual <= ready + 720) if outcome['status'] in ('REAL', 'CANCELED') else None
        values, probabilities = {}, {}
        for name, belief in beliefs.items():
            branch = belief.copy()
            branch.observe_connection_event(incoming.id, 'arrival', now - incoming.arr_time)
            law = DiscreteLaw(*branch.event_distribution(outgoing.id, 'departure', parameters))
            values[name] = law_summary(law, actual)
            probabilities[name] = law.board_probability(ready)
        rows.append({'incoming': before['id'], 'outgoing': after['id'], 'clock': now, 'ready': ready,
                     'incoming_delay': now - incoming.arr_time, 'actual': actual, 'status': outcome['status'],
                     'caught': caught, 'methods': values, 'probabilities': probabilities})
    summary = {}
    for name in METHODS:
        known = [row for row in rows if row['caught'] is not None]
        p = np.asarray([row['probabilities'][name] for row in known])
        y = np.asarray([row['caught'] for row in known], dtype=float)
        summary[name] = {'raw_interval': summarize_endpoints(rows, name), 'n_probability': len(known),
                         'brier': float(np.mean((p - y) ** 2)), 'signed_probability_error': float(np.mean(p - y))}
    return {'full_scheduled_pairs': len(ordered), 'preselected_every_20th': len(fixed), 'censored': dict(censored),
            'target_statuses': dict(Counter(row['status'] for row in rows)), 'methods': summary, 'rows': rows}


def training_grid_reference(data, fit):
    pairs = training_pairs(data, fit['train_dates'], fit)
    choices = [(day, before, after) for day, rows in pairs.items() for before, after in rows
               if before['arrival']['status'] == after['departure']['status'] == 'REAL']
    choices.sort(key=lambda item: item[1]['arrival']['time_seconds'] - item[1]['scheduled_arrival_seconds'])
    records = []
    for index in (0, len(choices) // 2, len(choices) - 1):
        day, before, after = choices[index]
        graph, source, parameters, _ = setup(data, day, fit)
        mapping = {row['id']: cid for cid, row in source.items()}
        results = {}
        for step in (60, 30):
            model = JointDayBelief(graph, public_occurrences=False, deltas_seconds=np.arange(-360, 361, step), **fit['increments'])
            model.observe_connection_event(mapping[before['id']], 'arrival', before['arrival']['time_seconds'] - before['scheduled_arrival_seconds'])
            law = DiscreteLaw(*model.event_distribution(mapping[after['id']], 'departure', parameters))
            delta, weights = model.day_posterior(parameters)
            results[str(step)] = {'mean': law.mean, 'lower': law.quantile(.05), 'upper': law.quantile(.95),
                                  'day_posterior_mean': float(delta @ weights),
                                  'board_probability': law.board_probability(before['arrival']['time_seconds'] + 120)}
        records.append({'date': day, 'incoming_source': before['id'], 'outgoing_source': after['id'], 'steps': results})
    return records


def main():
    data, _, fit = inputs()
    calibration_dates = [f'2023-10-{day:02}' for day in range(3, 12)]
    protocol = {'training': fit, 'calibration_dates': calibration_dates, 'development_dates': DATES,
                'endpoint_information': 'Prior only, with the same fixed route parameters and increments.',
                'transfer_selection': 'Sort schedule-only pairs by incoming departure/id then outgoing departure/id; '
                'retain positions 0,20,40,... before inspecting actual outcomes. Both beliefs observe only incoming arrival.',
                'transfer_intervals': 'Raw central90 intervals; no conditional-envelope calibration or independence claim.',
                'delta_grid': list(range(-360, 361, 60)), 'date_status': 'Already observed development dates.'}
    write(HERE / 'joint_day_local_v2_protocol.json', protocol)
    write(HERE / 'joint_day_training_grid_v2.json', training_grid_reference(data, fit))
    rows = {}
    started = perf_counter()
    for day in calibration_dates + list(DATES):
        rows[day] = endpoints(data, day, fit)
        print('ENDPOINTS', day, len(rows[day]), round(perf_counter() - started, 2), flush=True)
    offsets, scores = {}, {}
    for name in METHODS:
        score = {}
        for day in calibration_dates:
            values = sorted(max(row['methods'][name]['lower'] - row['actual'], row['actual'] - row['methods'][name]['upper']) for row in rows[day])
            score[day] = values[math.ceil(.9 * len(values)) - 1]
        scores[name] = score
        offsets[name] = max(0., max(score.values()))
    test, paired = {}, {}
    for day in DATES:
        test[day] = {name: {'raw': summarize_endpoints(rows[day], name),
                           'date_calibrated': summarize_endpoints(rows[day], name, offsets[name])} for name in METHODS}
        paired[day] = transfers(data, day, fit)
        print('TRANSFER', day, {name: value['brier'] for name, value in paired[day]['methods'].items()}, flush=True)
    result = {'protocol': protocol, 'offset_seconds': offsets, 'date_scores': scores, 'endpoints': test,
              'transfers': paired, 'endpoint_rows': rows, 'elapsed_seconds': perf_counter() - started}
    with gzip.open(HERE / 'joint_day_local_v2_results.json.gz', 'wt') as handle:
        json.dump(result, handle, separators=(',', ':'), allow_nan=False)
    write(HERE / 'joint_day_local_v2_summary.json', {key: result[key] for key in ('protocol', 'offset_seconds', 'endpoints', 'elapsed_seconds')} |
          {'transfers': {day: {key: value for key, value in row.items() if key != 'rows'} for day, row in paired.items()}})


if __name__ == '__main__':
    main()
