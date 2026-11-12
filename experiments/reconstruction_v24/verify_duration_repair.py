"""Measure the fitted elapsed repair on the original training route and failures."""

from collections import defaultdict
import json
from statistics import mean

from experiments.reconstruction_v23.run_transfer_screen import TRAINING, model
from experiments.reconstruction_v24.calibration_probe import clock_stats
from experiments.reconstruction_v24.screen import HERE, SOURCE
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from src.empirical_duration_belief import EmpiricalDurationBelief, fit_empirical_durations


def main():
    old = json.loads((HERE/'calibration_probe.json').read_text())
    training, _ = load_city(TRAINING)
    fit = fit_training(training, retain_no_pickup=True)
    empirical = fit_empirical_durations(training, retain_no_pickup=True)
    rows = []
    for day in fit['train_dates']:
        graph, actual, _, _, metadata, _ = dated_graph(training, day, 'gtfs_station_max', True)
        base, params = model(graph, fit)
        cids = [cid for cid, row in metadata.items()
                if row['route_id'] in old['routes'] and row['outcome'] == 'MEASURED']
        trips = {base._trip_by_cid[cid] for cid in cids}
        laws = dict(original=base.with_grid_step(20, trips=trips),
                    empirical=EmpiricalDurationBelief(graph, duration_fit=empirical,
                        public_occurrences=False, grid_step_seconds=20, **fit['increments']))
        for name, belief in laws.items():
            phases = defaultdict(list)
            for cid in cids:
                prediction, _ = clock_stats(belief, params, cid)
                observed = actual.events[cid].departure-graph.connections[cid].dep_time
                phase = belief._positions[cid, 'departure'][1]//10
                phases[phase].append((prediction['mean_delay'], observed))
            def summarize(values):
                return dict(n=len(values), predicted_delay=mean(x for x, _ in values),
                            actual_delay=mean(y for _, y in values),
                            mean_error=mean(x-y for x, y in values))
            rows.append(dict(law=name, grid=20, day=day,
                all=summarize([v for values in phases.values() for v in values]),
                phases={f'{p*5}-{p*5+4}': summarize(values) for p, values in sorted(phases.items())}))
    data, _ = load_city(SOURCE)
    targets = []
    for day in {t['root_id'][:10] for t in old['targets']}:
        graph, _, _, _, _, _ = dated_graph(data, day, 'gtfs_station_max', True)
        _, params = model(graph, fit)
        belief = EmpiricalDurationBelief(graph, duration_fit=empirical, public_occurrences=False,
                                        grid_step_seconds=20, **fit['increments'])
        for target in old['targets']:
            if target['root_id'][:10] != day:
                continue
            stats, (times, mass, cancel) = clock_stats(belief, params, target['cid'])
            before = (1-cancel)*mass[times < target['entry']].sum()
            future = (1-cancel)*mass[times > target['decision_clock']].sum()
            stats['future_given_no_local_departure'] = float(future/(before+future+cancel))
            targets.append(dict(root_id=target['root_id'], cid=target['cid'], empirical=stats,
                original=next(v for v in target['grids'] if v['grid_step'] == 20),
                actual_delay=target['actual_departure']-target['scheduled']))
    result = dict(scope='All July 25 training data fit the repair. The route and five failures were '
                  'selected retrospectively for diagnosis; training errors are in-sample, not validation.',
                  training=rows, targets=targets, fit_exclusions=empirical['excluded'])
    (HERE/'route_calibration_repair.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
