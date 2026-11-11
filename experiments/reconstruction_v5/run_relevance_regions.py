"""Hard-support relevance on complete city graphs and archived causal histories.

No city policy is run here. Past actions are replayed only to condition support;
held-out endpoint clocks are read only after their archived ride has completed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import ROOT, dated_graph, fit_training
from src.evolving_trip_belief import EvolvingTripBelief
from src.transit_relevance import RelevanceIndex


def count_laws(belief):
    return sum(memo.law is not None for memo in belief._cache.values())


def memory_peak_kib():
    # VmHWM belongs to this Linux process image; ru_maxrss can include the
    # launcher image's pre-exec high-water mark in this workspace.
    return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmHWM:')))


def run(data, archive):
    started = perf_counter()
    fit = fit_training(data, retain_no_pickup=True)
    fit_ms = (perf_counter() - started) * 1000
    rows = []
    for day in data['holdout_dates']:
        started = perf_counter()
        graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        graph_ms = (perf_counter() - started) * 1000
        pooled = fit['route_priors']['pooled']
        parameters = {c.route: fit['route_priors'].get(c.route, pooled) for c in graph.connections}
        total_trips = len({c.trip_id for c in graph.connections})
        for qi, query in enumerate(data['od_queries']):
            archived = next(row for row in archive['rows'] if row['date'] == day and row['query'] == qi
                            and row['method'] == 'evolving_rollout')
            belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
            last_arrival = query['ready_seconds']
            for number, decision in enumerate(archived['decisions']):
                before = perf_counter()
                index = RelevanceIndex(graph, lambda cid, kind: belief.hard_support(cid, kind, parameters))
                support_ms = (perf_counter() - before) * 1000
                before = perf_counter()
                result = index.solve(decision['stop'], stops[query['destination']],
                                     decision['clock_seconds'], query['deadline_seconds'])
                closure_ms = (perf_counter() - before) * 1000
                # Public-feed extension is a different synthetic contract. This
                # conservatively quantifies it; it is not the actual city feed.
                before = perf_counter()
                public = index.solve(decision['stop'], stops[query['destination']],
                                     decision['clock_seconds'], query['deadline_seconds'],
                                     public_occurrences=True, publication_lookback=600)
                public_ms = (perf_counter() - before) * 1000
                assert count_laws(belief) == 0
                rows.append({'date': day, 'query': qi, 'decision': number,
                             'stop': decision['stop'], 'clock_seconds': decision['clock_seconds'],
                             'deadline_seconds': query['deadline_seconds'],
                             'total_connections': len(graph.connections), 'total_trips': total_trips,
                             'transport_edges': len(result.transport_edge_ids),
                             'transport_trips': len(result.transport_trip_ids),
                             'required_trip_count': len(result.required_trip_ids),
                             'retained_connections': len(result.retained_connection_ids),
                             'synthetic_public_required_trips': len(public.required_trip_ids),
                             'synthetic_public_retained_connections': len(public.retained_connection_ids),
                             'same_trip_offpath_connections_retained':
                                 len(result.retained_connection_ids - result.transport_edge_ids),
                             'fit_ms_once': fit_ms, 'graph_build_ms_once_per_date': graph_ms,
                             'support_metadata_ms': support_ms, 'closure_ms': closure_ms,
                             'synthetic_public_closure_ms': public_ms,
                             'process_image_peak_kib_so_far': memory_peak_kib(),
                             'inference_laws_constructed': count_laws(belief), 'work': result.work,
                             'retained_trip_ids': sorted(result.required_trip_ids),
                             'retained_connection_ids': sorted(result.retained_connection_ids),
                             'observed_exact_event_count': len(belief._known_times)})

                # Only the previous, completed action can supply the next query's evidence.
                cid = decision['action']
                if cid is None or cid == 'wait':
                    continue
                connection = graph.connections[cid]
                if decision['outcome'] == 'board':
                    source = metadata[cid]
                    departure = source['departure']['time_seconds']
                    arrival = source['arrival']['time_seconds']
                    assert arrival == decision['end_seconds']
                    belief.observe_connection_event(cid, 'departure', departure - connection.dep_time)
                    belief.observe_connection_event(cid, 'arrival', arrival - connection.arr_time)
                    last_arrival = arrival
                elif decision['outcome'] == 'wait_late':
                    transfer = (graph.stops[decision['stop']].min_transfer_time
                                if decision['current_trip'] is not None
                                and decision['current_trip'] != connection.trip_id else 0)
                    belief.observe_wait(cid, max(decision['clock_seconds'], last_arrival + transfer),
                                        decision['end_seconds'])
                elif decision['outcome'] not in ('deadline', 'unresolved'):
                    raise ValueError(f"Unexpected receipt-free outcome: {decision['outcome']}")
    return fit, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
    parser.add_argument('--archive', type=Path, default=ROOT / 'experiments/reconstruction_v4/city_replay_results.json')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('relevance_regions_results.json.gz'))
    args = parser.parse_args()
    before = perf_counter()
    initial_resource_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    initial_image_peak = memory_peak_kib()
    data, _ = load_city(args.input)
    archive = json.loads(args.archive.read_text())
    fit, rows = run(data, archive)
    output = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'input': str(args.input.relative_to(ROOT)), 'archived_histories': str(args.archive.relative_to(ROOT)),
              'scope': 'Complete declared city graph; support-only diagnostic on all archived rollout decision histories. No new policy journeys or held-out fitting.',
              'training_dates': data['train_dates'], 'holdout_dates': data['holdout_dates'],
              'fit': fit, 'public_occurrences': False, 'future_external_receipts': [],
              'minimum_travel_relaxation_seconds': 0,
              'retained_trip_identity_includes_all_slots_and_evidence': True,
              'timing_scope': 'Separate training, graph build, support metadata, closure. VmHWM is the current Linux process-image peak; getrusage is retained separately, including the startup image.',
              'elapsed_ms': (perf_counter() - before) * 1000,
              'process_image_peak_kib': memory_peak_kib(),
              'process_image_peak_kib_at_start': initial_image_peak,
              'getrusage_peak_kib_at_start': initial_resource_peak,
              'getrusage_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'source_texts': {path: (ROOT / path).read_text() for path in
                               ['src/transit_relevance.py', 'src/evolving_trip_belief.py',
                                'experiments/reconstruction_v5/run_relevance_regions.py']},
              'rows': rows}
    with gzip.open(args.output, 'wt') as handle:
        json.dump(output, handle)
    print(json.dumps({'output': str(args.output), 'states': len(rows),
                      'initial_states': [{key: row[key] for key in
                                          ['date', 'query', 'total_trips', 'required_trip_count',
                                           'total_connections', 'retained_connections',
                                           'synthetic_public_required_trips', 'support_metadata_ms', 'closure_ms']}
                                         for row in rows if row['decision'] == 0]}), flush=True)


if __name__ == '__main__':
    main()
