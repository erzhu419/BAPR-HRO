"""Read-only diagnosis of ordered-clock drift on training and V24 failures."""

from collections import defaultdict
import gzip
import json
from statistics import mean

import numpy as np
from scipy.special import log_ndtr

from experiments.reconstruction_v23.run_transfer_screen import TRAINING, model
from experiments.reconstruction_v24.screen import HERE, SOURCE
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from src.ordered_duration_belief import OrderedDurationBelief


def clock_stats(belief, params, cid):
    times, mass, cancel = belief.event_distribution(cid, 'departure', params)
    scheduled = belief.graph.connections[cid].dep_time
    return dict(mean_delay=float(times @ mass-scheduled), canceled=cancel), (times, mass, cancel)


def run():
    training, _ = load_city(TRAINING)
    fit = fit_training(training, retain_no_pickup=True)
    data, _ = load_city(SOURCE)
    with gzip.open(HERE/'results.json.gz', 'rt') as handle:
        results = json.load(handle)['results']
    with (HERE/'selection.json').open() as handle:
        roots = {r['root_id']: r for day in json.load(handle)['by_day'].values() for r in day['roots']}
    targets, routes = [], set()
    for day in data['dates']:
        graph, actual, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        base, params = model(graph, fit)
        for result in results:
            outcome = result['actual']['outcomes']
            nominal, first = outcome['clock_nominal'], outcome['first_offer']
            if (result['day'] != day or nominal.get('loss') is None
                    or nominal['arrived'] or not first['arrived']):
                continue
            root = roots[result['root_id']]
            cid = nominal['action']
            trip, slot = base._positions[cid, 'departure']
            connection = graph.connections[cid]
            routes.add(connection.route)
            entry, now = root['scheduled_entry'], result['actual']['public']['clock']
            grids = []
            for step in (None, 60, 20, 10):
                belief = base.with_grid_step(step, trips={trip})
                stats, (times, mass, cancel) = clock_stats(belief, params, cid)
                before = (1-cancel)*mass[times < entry].sum()
                future = (1-cancel)*mass[times > now].sum()
                stats.update(grid_step=step, before_entry_mass=float(before), future_mass=float(future),
                             future_given_no_local_departure=float(future/(before+future+cancel)))
                grids.append(stats)
            ordered = OrderedDurationBelief(graph, public_occurrences=False, **fit['increments'])
            ordered_stats, _ = clock_stats(ordered, params, cid)
            targets.append(dict(root_id=root['root_id'], route=connection.route, cid=cid,
                departure_index=slot//2, scheduled=connection.dep_time,
                actual_departure=actual.events[cid].departure, entry=entry, decision_clock=now,
                prior=fit['route_priors'][connection.route], grids=grids,
                existing_ordered_duration=ordered_stats))
    training_summary = []
    increments = defaultdict(list)
    transition_means = defaultdict(list)
    for day in fit['train_dates']:
        graph, actual, _, _, metadata, _ = dated_graph(training, day, 'gtfs_station_max', True)
        base, params = model(graph, fit)
        selected = [cid for cid, row in metadata.items() if row['route_id'] in routes and row['outcome'] == 'MEASURED']
        trips = {base._trip_by_cid[cid] for cid in selected}
        for step in (None, 60):
            belief = base.with_grid_step(step, trips=trips)
            phases = defaultdict(list)
            all_rows = []
            for cid in selected:
                trip, slot = belief._positions[cid, 'departure']
                stats, _ = clock_stats(belief, params, cid)
                observed = actual.events[cid].departure-graph.connections[cid].dep_time
                row = stats['mean_delay'], observed
                phases[(slot//2)//5].append(row)
                all_rows.append(row)
            summarize = lambda rows: dict(n=len(rows), predicted_delay=mean(x for x, _ in rows),
                                         measured_delay=mean(y for _, y in rows),
                                         mean_error=mean(x-y for x, y in rows))
            training_summary.append(dict(day=day, grid_step=step, all=summarize(all_rows),
                phases={f'{phase*5}-{phase*5+4}': summarize(rows) for phase, rows in sorted(phases.items())}))
            for trip in trips:
                law, slots = belief._law(trip, params), belief._trips[trip]
                for index in range(1, len(slots)):
                    before, after = slots[index-1], slots[index]
                    if before.cid not in actual.events or after.cid not in actual.events:
                        continue
                    if actual.events[before.cid].canceled or actual.events[after.cid].canceled:
                        continue
                    kind = ('travel' if after.kind == 'arrival' else
                            'gap' if graph.connections[before.cid].arr_stop != graph.connections[after.cid].dep_stop
                            else 'dwell')
                    duration = after.scheduled-before.scheduled
                    weights = law.marginal(index-1)
                    edge = np.exp(law._transition(index-1))
                    elapsed = law.times[index][None, :]-law.times[index-1][:, None]
                    transition_means[step, kind, duration].append(float(weights @ (edge*elapsed).sum(axis=1)))
        for trip in trips:
            slots = base._trips[trip]
            for index in range(1, len(slots)):
                before, after = slots[index-1], slots[index]
                if before.cid not in actual.events or after.cid not in actual.events:
                    continue
                a, b = actual.events[before.cid], actual.events[after.cid]
                if a.canceled or b.canceled:
                    continue
                previous = a.departure if before.kind == 'departure' else a.arrival
                following = b.departure if after.kind == 'departure' else b.arrival
                duration = after.scheduled-before.scheduled
                kind = ('travel' if after.kind == 'arrival' else
                        'gap' if graph.connections[before.cid].arr_stop != graph.connections[after.cid].dep_stop
                        else 'dwell')
                increments[kind, duration].append(following-previous)
    def increment_row(kind, duration, rows):
        mu = duration+fit['increments'][kind+'_drift']*60
        sd = fit['increments'][kind+'_sd']*60
        normal_mean = mu+sd*np.exp(-.5*(mu/sd)**2-.5*np.log(2*np.pi)-log_ndtr(mu/sd))
        return dict(kind=kind, scheduled_seconds=duration, n=len(rows),
                    measured_elapsed=mean(rows), measured_sd=float(np.std(rows)),
                    measured_zero_fraction=float(np.mean(np.array(rows) == 0)),
                    measured_negative_fraction=float(np.mean(np.array(rows) < 0)),
                    measured_residual=mean(rows)-duration, continuous_ordered_mean=float(normal_mean),
                    grid60_mean=mean(transition_means[60, kind, duration]),
                    legacy_mean=mean(transition_means[None, kind, duration]))
    increment_summary = [increment_row(kind, duration, rows)
                         for (kind, duration), rows in sorted(increments.items())]
    result = dict(scope='Descriptive in-sample calibration diagnosis; no refitting or evaluation-based threshold. '
                        'Target failures were selected retrospectively; route training summaries use all measured '
                        'July 25 connections of those routes. Slot index is within the retained trip prefix.',
                  increments=fit['increments'], routes=sorted(routes), targets=targets,
                  training=training_summary, training_elapsed_by_schedule=increment_summary)
    (HERE/'calibration_probe.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(routes=sorted(routes), targets=len(targets), training=training_summary,
                         training_elapsed_by_schedule=increment_summary), indent=2))


if __name__ == '__main__':
    run()
