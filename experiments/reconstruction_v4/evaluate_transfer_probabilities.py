"""Score catchability after first-hand arrival, without future departure truth."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import numpy as np

from run_dated_replay import DATA_PATH, dated_graph, fit_training, read_inputs
from src.evolving_trip_belief import EvolvingTripBelief


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DATA_PATH)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('transfer_probability_calibration.json'))
    args = parser.parse_args()
    data, _ = read_inputs(data_path=args.data)
    fit = fit_training(data)
    dates = {}
    for day in data['holdout_dates']:
        graph, world, _, _, source, _ = dated_graph(data, day, 'gtfs_station_max')
        priors = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                  for c in graph.connections}
        base = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
        base.mean_events(priors)
        scores = defaultdict(list)
        pairs = arrivals = 0
        for incoming in graph.connections:
            if source[incoming.id]['outcome'] != 'MEASURED':
                continue
            ridden = world.events[incoming.id]
            belief = base.copy()
            belief.observe_connection_event(incoming.id, 'departure', ridden.departure - incoming.dep_time)
            belief.observe_connection_event(incoming.id, 'arrival', ridden.arrival - incoming.arr_time)
            arrivals += 1
            for outgoing in graph.get_connections_from(incoming.arr_stop):
                if outgoing.id == incoming.id or source[outgoing.id]['outcome'] == 'UNRESOLVED':
                    continue
                same = outgoing.trip_id == incoming.trip_id
                if not outgoing.pickup_allowed and not same:
                    continue
                ready = ridden.arrival + (0 if same else graph.stops[incoming.arr_stop].min_transfer_time)
                if outgoing.dep_time > ridden.arrival + 25 * 60:
                    continue
                clocks, weights, cancel = belief.event_distribution(outgoing.id, 'departure', priors)
                p = (1 - cancel) * float(weights[(clocks >= ready) & (clocks <= ready + 720)].sum())
                actual = world.events[outgoing.id]
                caught = not actual.canceled and ready <= actual.departure <= ready + 720
                scheduled = ready <= outgoing.dep_time <= ready + 720
                scores['evolving'].append((p - caught) ** 2)
                scores['schedule'].append((float(scheduled) - caught) ** 2)
                scores['actual_caught'].append(int(caught))
                if not same and abs(outgoing.dep_time - ready) <= 300:
                    scores['critical_evolving'].append((p - caught) ** 2)
                    scores['critical_schedule'].append((float(scheduled) - caught) ** 2)
                pairs += 1
        dates[day] = {'first_hand_arrival_states': arrivals, 'scored_candidate_pairs': pairs,
                      'actual_caught': sum(scores['actual_caught']),
                      'evolving_brier': float(np.mean(scores['evolving'])),
                      'schedule_brier': float(np.mean(scores['schedule'])),
                      'different_trip_within_5min_scheduled_margin': {
                          'pairs': len(scores['critical_evolving']),
                          'evolving_brier': float(np.mean(scores['critical_evolving'])),
                          'schedule_brier': float(np.mean(scores['critical_schedule']))}}
    result = {'data': str(args.data), 'training': fit,
              'transfer_rule': 'gtfs_station_max, absent station rules retain 120-second assumption',
              'patience_seconds': 720, 'by_date': dates,
              'scope': 'Each measured incoming segment creates a local arrival state. Only its ridden endpoints condition the trip law. Outgoing realized outcomes are used exclusively for scoring. Unknown outgoing outcomes are excluded. Public forecast information is absent. Earlier scheduled candidates remain eligible as in the executor.',
              'dependence': 'Candidate pairs share arrivals, trips and dates. These Brier means are diagnostics, without independent-pair statistical inference.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dates, indent=2))


if __name__ == '__main__':
    main()
