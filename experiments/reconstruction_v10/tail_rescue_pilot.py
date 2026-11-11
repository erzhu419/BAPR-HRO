"""Value a downstream failure branch on the eight frozen V9 histories."""

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter, process_time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from experiments.reconstruction_v8.budget_run import SOURCE, write_json
from experiments.reconstruction_v9.search_reference import ARCHIVE, HISTORIES
from src.recourse_itinerary_router import build_trip_blocks
from src.tail_rescue import TailRescueBounds, compile_tail_rescue


def case_key(row):
    return row['date'], row['query'], row['decision']


def main():
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    archive = json.loads(ARCHIVE.read_text())['rows']
    journeys = {(r['date'], r['query']): r for r in archive if r['method'] == 'evolving_rollout'}
    histories = json.loads(HISTORIES.read_text())['rows']
    old_rows = json.loads((HERE.parent / 'reconstruction_v9' / 'witness_reference_results.json').read_text())['rows']
    old_witnesses = {case_key(r): r for r in old_rows}
    graphs, rows = {}, []
    for history in histories:
        day, query, decision = case_key(history)
        if day not in graphs:
            graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
            parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                          for c in graph.connections}
            graphs[day] = graph, stops, metadata, parameters, build_trip_blocks(graph)
        graph, stops, metadata, parameters, blocks = graphs[day]
        journey = journeys[day, query]
        destination, deadline = stops[journey['od']['destination']], journey['od']['deadline_seconds']
        old = old_witnesses[day, query, decision]
        candidates = [old['same_primary']]
        if old['uct128_alternative'] and old['uct128_alternative']['primary']:
            candidates.append(old['uct128_alternative'])
        measures = []
        for candidate in candidates:
            belief, state = context(graph, metadata, journey, decision, fit)
            if (state.stop, state.time, destination) != (history['stop'], history['clock'], history['destination']):
                raise AssertionError('The reconstructed root differs from the archived history')
            started, cpu = perf_counter(), process_time()
            bounds = TailRescueBounds(belief, parameters, graph, destination, deadline)
            plan, value = compile_tail_rescue(bounds, state, candidate['primary'], trip_blocks=blocks)
            if abs(value.primary.upper - candidate['primary_only_upper_seconds']) > 1e-8:
                raise AssertionError('Fixed-primary replay did not reproduce the original upper')
            measures.append(dict(plan=asdict(plan), upper=value.upper, primary_upper=value.primary.upper,
                failure_probability=value.failed_probability, rescue_gain=value.rescue_success_gain,
                rescue_completion_probability=value.rescue_completion_probability,
                wall_seconds=perf_counter()-started, cpu_seconds=process_time()-cpu))
        upper = min([old['best_upper_seconds']] + [m['upper'] for m in measures])
        rows.append(dict(date=day, query=query, decision=decision, original_upper=history['upper'],
                         v9_best_upper=old['best_upper_seconds'], upper=upper,
                         fixed_primary_tail_rescue=measures[0], alternative_tail_rescue=measures[1:]))
        print(day, query, decision, round(upper, 3), [(m['plan']['failure_index'],
              round(m['rescue_gain'], 3), round(m['wall_seconds'], 3)) for m in measures], flush=True)
    summary = dict(mean_v9_best_upper=mean(r['v9_best_upper'] for r in rows),
        mean_v10_best_upper=mean(r['upper'] for r in rows),
        improves_v9=sum(r['upper'] < r['v9_best_upper'] - 1e-8 for r in rows),
        fixed_primary_with_rescue=sum(bool(r['fixed_primary_tail_rescue']['plan']['rescue']) for r in rows),
        fixed_primary_median_wall_seconds=median(r['fixed_primary_tail_rescue']['wall_seconds'] for r in rows))
    write_json(HERE / 'tail_rescue_results.json', dict(summary=summary, rows=rows,
        scope='Same eight public histories and prior as V9. One cold search for an independent downstream rescue per archived primary. Plan scoring uses the exact committed no-publication law.'))


if __name__ == '__main__':
    main()
