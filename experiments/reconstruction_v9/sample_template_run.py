"""Matched cold sampling profile with translated unobserved HMM templates."""

import json
from statistics import median
from time import perf_counter, process_time

import numpy as np

from sample_run import ROOT, HERE, context, dated_graph, fit_training, load_city, rng
from src.temporal_scan import sample_clocks, template_trip_map


def sample(graph, metadata, journey, row, fit, parameters, n, reuse):
    belief, _ = context(graph, metadata, journey, row['decision'], fit)
    before, cpu = perf_counter(), process_time()
    templates = template_trip_map(belief, parameters) if reuse else None
    build_seconds = perf_counter() - before
    clocks = sample_clocks(belief, parameters, rng(row), n, templates=templates)
    timings = dict(index_wall_seconds=build_seconds, wall_seconds=perf_counter() - before,
                   cpu_seconds=process_time() - cpu)
    if templates is not None:
        representatives = {rep for rep, _ in templates.values()} | set(belief._revision)
        timings.update(trip_chains=len(belief._trips), distinct_chains=len(representatives),
                       event_slots=sum(map(len, belief._trips.values())),
                       representative_slots=sum(len(belief._trips[t]) for t in representatives))
    return clocks, timings


def main():
    protocol = json.loads((HERE / 'sample_protocol.json').read_text())
    n = protocol['draws_per_root']
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    v8 = json.loads((ROOT / 'experiments/reconstruction_v8/opportunity_city_results.json').read_text())['rows']
    archive = json.loads((ROOT / 'experiments/reconstruction_v4/city_replay_results.json').read_text())
    journeys = {(j['date'], j['query']): j for j in archive['rows'] if j['method'] == 'evolving_rollout'}
    day, rows = None, []
    for i, old in enumerate(v8):
        if day != old['date']:
            day = old['date']
            graph, _, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
            parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                          for c in graph.connections}
        journey = journeys[day, old['query']]
        order = ('full', 'template') if i % 2 == 0 else ('template', 'full')
        values = {arm: sample(graph, metadata, journey, old, fit, parameters, n, arm == 'template')
                  for arm in order}
        full, full_timings = values['full']
        reused, reuse_timings = values['template']
        if any(not np.array_equal(a, b) for a, b in zip(full, reused)):
            raise AssertionError('Translated template changed a clock or cancellation')
        rows.append(dict(date=day, query=old['query'], decision=old['decision'], arms=list(order),
                         draws=n, connections=len(graph.connections), full=full_timings,
                         template=reuse_timings, all_events_equal=True))
        print(day, old['query'], old['decision'], 'full', round(full_timings['wall_seconds'], 3),
              'template', round(reuse_timings['wall_seconds'], 3), flush=True)
    summary = {key: {metric: median(row[key][metric] for row in rows)
                     for metric in ('wall_seconds', 'cpu_seconds', 'index_wall_seconds')}
               for key in ('full', 'template')}
    summary['paired_clock_and_service_triples'] = sum(row['draws'] * row['connections'] for row in rows)
    result = dict(scope='Same 32-world tapes and eight roots as sample_protocol.json. Each arm uses a fresh conditioned belief, excludes history reconstruction and includes template construction where used. Arm order alternates by root.',
                  template_method='Reuse representative forward messages and batch all member tapes per representative.',
                  rows=rows, summary=summary)
    (HERE / 'sample_template_results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
