"""Diagnose ordered-kernel duration drift and the July27 finite-bank miss."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city, past_forecast_fit
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from src.evolving_trip_belief import EvolvingTripBelief


def context(path, day, fit=None):
    data, _ = load_city(path)
    fit = fit or fit_training(data, retain_no_pickup=True)
    graph, actual, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
    priors = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
              for c in graph.connections}
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    return data, fit, graph, actual, metadata, priors, belief


def catch_metrics(cid, ready, belief, priors, scenes, graph, actual):
    c = graph.connections[cid]
    times, weights, cancel = belief.event_distribution(cid, 'departure', priors)
    caught = (ready <= times) & (times <= ready + 720)
    event = actual.events.get(cid)
    cdf = weights.cumsum()
    quantiles = [int(times[min(int(cdf.searchsorted(q)), len(times) - 1)]) for q in (.05, .5, .95)]
    draws = [scene.events[cid] for scene in scenes]
    probability = float((1 - cancel) * weights[caught].sum())
    successes = sum(not e.canceled and ready <= e.departure <= ready + 720 for e in draws)
    return {'connection': cid, 'trip': c.trip_id, 'route': c.route, 'ready': ready,
            'scheduled_departure': c.dep_time, 'model_mean_departure': float(times @ weights),
            'model_departure_q05_q50_q95': quantiles,
            'posterior_board_probability': probability,
            'posterior_all_worlds_catch_probability': probability ** len(draws),
            'bank_departures': [e.departure for e in draws],
            'bank_caught': successes, 'bank_catch_fraction': successes / len(draws),
            'bank_worlds': len(draws), 'actual_departure': None if event is None else event.departure,
            'actual_caught': None if event is None else bool(not event.canceled and ready <= event.departure <= ready + 720)}


def profile_trip(trip, fit, graph, metadata, priors, belief, include_slots):
    slots = belief._trips[trip]
    law = belief._law(trip, priors)
    route = graph.connections[slots[0].cid].route
    rows, previous_mean = [], None
    for index, slot in enumerate(slots):
        times, weights, _ = belief.event_distribution(slot.cid, slot.kind, priors)
        mean = float(times @ weights) if len(times) else None
        observed = metadata[slot.cid][slot.kind]
        actual = observed['time_seconds'] if observed['status'] == 'REAL' else None
        row = {'slot': index, 'connection': slot.cid, 'event': slot.kind, 'scheduled': slot.scheduled,
               'model_delay': None if mean is None else mean - slot.scheduled,
               'observed_delay': None if actual is None else actual - slot.scheduled}
        if index:
            before = slots[index - 1]
            kind = ('travel' if slot.kind == 'arrival' else 'gap' if
                    graph.connections[before.cid].arr_stop != graph.connections[slot.cid].dep_stop else 'dwell')
            duration = slot.scheduled - before.scheduled
            target = duration + fit['increments'][kind + '_drift'] * 60
            indices = np.flatnonzero(~law.canceled[index - 1])
            desired = before.scheduled + priors[route]['mean'] * 60
            chosen = int(indices[np.argmin(abs(law.times[index - 1][indices] - desired))])
            probabilities = np.exp(law._transition(index - 1)[chosen])
            operating = ~law.canceled[index]
            probabilities = probabilities[operating]
            expected = float(probabilities @ (law.times[index][operating] - law.times[index - 1][chosen])
                             / probabilities.sum())
            row.update(transition=kind, scheduled_duration=duration, raw_duration_target=target,
                       kernel_expected_duration=expected, kernel_bias_vs_raw_target=expected - target,
                       nominal_increment=None if mean is None or previous_mean is None else mean - previous_mean)
        previous_mean = mean
        rows.append(row)
    zero = [r for r in rows if r.get('transition') == 'dwell' and r['scheduled_duration'] == 0]
    return {'trip': trip, 'route': route, 'slots': len(slots), 'zero_dwell_count': len(zero),
            'route_prior_mean_delay': priors[route]['mean'] * 60,
            'model_first_delay': rows[0]['model_delay'], 'observed_first_delay': rows[0]['observed_delay'],
            'model_last_delay': rows[-1]['model_delay'], 'observed_last_delay': rows[-1]['observed_delay'],
            'zero_dwell_mean_kernel_duration': float(np.mean([r['kernel_expected_duration'] for r in zero])) if zero else None,
            'zero_dwell_raw_duration_target': fit['increments']['dwell_drift'] * 60,
            'slot_comparison': rows if include_slots else [rows[0], rows[-1]]}


def training_duration_bins(data):
    groups = defaultdict(list)
    for day in data['train_dates']:
        trips = defaultdict(list)
        for c in data['dates'][day]['connections']:
            if not c['dropoff_allowed']:
                continue
            trips[c['trip_id']].append(c)
            if c['outcome'] == 'MEASURED':
                scheduled = c['scheduled_arrival_seconds'] - c['scheduled_departure_seconds']
                actual = c['arrival']['time_seconds'] - c['departure']['time_seconds']
                groups['travel', 'zero' if scheduled == 0 else 'short_le60' if scheduled <= 60 else 'long'].append(actual - scheduled)
        for rows in trips.values():
            rows.sort(key=lambda c: c['dep_sequence'])
            for before, after in zip(rows, rows[1:]):
                if (before['arr_stop'] != after['dep_stop'] or before['arr_sequence'] != after['dep_sequence']
                        or before['arrival']['status'] != 'REAL' or after['departure']['status'] != 'REAL'):
                    continue
                scheduled = after['scheduled_departure_seconds'] - before['scheduled_arrival_seconds']
                actual = after['departure']['time_seconds'] - before['arrival']['time_seconds']
                groups['dwell', 'zero' if scheduled == 0 else 'short_le60' if scheduled <= 60 else 'long'].append(actual - scheduled)
    return [{'kind': kind, 'scheduled_bin': bucket, 'n': len(values),
             'mean_observed_residual': float(np.mean(values)), 'std_observed_residual': float(np.std(values))}
            for (kind, bucket), values in sorted(groups.items())]


def main():
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, fit, graph, actual, metadata, priors, belief = context(source, '2023-07-27')
    scenes = belief.sample(priors, np.random.default_rng(np.random.SeedSequence([20230727, 0])), 4)
    initial = [catch_metrics(cid, 29100, belief, priors, scenes, graph, actual) for cid in (7588, 8341)]
    after_wait = belief.copy()
    after_wait.observe_wait(7588, 29100, 29820)
    next_scenes = after_wait.sample(priors, np.random.default_rng(np.random.SeedSequence([20230727, 1])), 4)
    second = catch_metrics(11065, 29820, after_wait, priors, next_scenes, graph, actual)
    focused = [profile_trip(graph.connections[cid].trip_id, fit, graph, metadata, priors, belief, True)
               for cid in (7588, 8341, 11065)]
    # Select July26 examples from static zero-dwell counts, without ranking actual delay errors.
    _, _, g26, a26, m26, p26, b26 = context(source, '2023-07-26', fit)
    order = []
    for trip, slots in b26._trips.items():
        zero = sum(slots[i].scheduled == slots[i - 1].scheduled
                   and g26.connections[slots[i - 1].cid].arr_stop == g26.connections[slots[i].cid].dep_stop
                   for i in range(2, len(slots), 2))
        order.append((-zero, trip))
    july26 = [profile_trip(trip, fit, g26, m26, p26, b26, False) for _, trip in sorted(order)[:3]]
    national_source = ROOT / 'experiments/reconstruction_data_audit/city_dated_replay.json.gz'
    national_fit = past_forecast_fit('2023-10-25')
    _, _, gn, an, mn, pn, bn = context(national_source, '2023-10-25', national_fit)
    order = []
    for trip, slots in bn._trips.items():
        zero = sum(slots[i].scheduled == slots[i - 1].scheduled
                   and gn.connections[slots[i - 1].cid].arr_stop == gn.connections[slots[i].cid].dep_stop
                   for i in range(2, len(slots), 2))
        order.append((-zero, trip))
    national = [profile_trip(trip, national_fit, gn, mn, pn, bn, False) for _, trip in sorted(order)[:3]]
    result = {'scope': 'Read-only diagnosis using past fits, static selection and realized endpoint comparison. No policy or model change.',
              'units': 'seconds except fit increments, whose stored units are minutes',
              'july_train_dates': fit['train_dates'], 'july_increment_fit': fit['increments'],
              'july_training_residual_bins': training_duration_bins(data),
              'july27_first_decision': initial, 'july27_after_wait_7588': second,
              'july27_focused_trip_slots': focused, 'july26_static_most_zero_dwell_trips': july26,
              'national_prior_dates': national_fit['train_dates'], 'national_increment_fit': national_fit['increments'],
              'national_static_most_zero_dwell_trips': national}
    output = Path(__file__).with_name('runtime_transition_bias_results.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Saved', output, output.stat().st_size, 'bytes')
    print(json.dumps({'july_first': initial, 'july_second': second, 'july26': july26, 'national': national}, indent=2))


if __name__ == '__main__':
    main()
