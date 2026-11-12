"""Match public first-ride policies and hindsight on two frozen first menus."""

from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v13.platform_run import (
    ARCHIVE, CASES, INVENTORY, SOURCE, V10, WORLDS, build_model, compile_arms,
    dated_graph, fit_training, key, load_city, load_plan, prepare_waiting_context,
    replay_arm, write_json)
from src.continuation_confidence import continuation_lower
from src.public_rescue_candidates import PublicRescueCandidates, screen_public_itineraries
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map


BASE = 'first_visible'
ARMS = {BASE: 0, 'expanded_guarded': 0, 'expanded_first_visible': None}
WINDOW = 720
V13 = ROOT / 'experiments/reconstruction_v13/platform_results.json'


def read(path):
    return json.loads(path.read_text())


def hindsight(graph, scan, clocks, root, goal, deadline, menus):
    """Exact forced first connections plus the common relaxed clairvoyant suffix."""
    departures, arrivals, canceled = clocks
    transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
    ready = max(root.time, root.last_arrival + graph.stops[root.stop].min_transfer_time)
    values = {}
    for cid in menus['expanded']:
        c = graph.connections[cid]
        values[cid] = np.full(len(departures), deadline, dtype=np.int32)
        if not c.pickup_allowed or c.id in root.used_connections:
            continue
        for i in np.flatnonzero((departures[:, cid] >= ready) & (~canceled[:, cid].astype(bool))
                                & (arrivals[:, cid] <= deadline)):
            arrival = int(arrivals[i, cid])
            after = replace(root, stop=c.arr_stop, time=arrival, current_trip=c.trip_id,
                            last_arrival=arrival, last_connection=cid,
                            used_connections=root.used_connections | {cid}, local_since=arrival)
            values[cid][i] = scan.platform_arrivals(*(x[i:i+1] for x in clocks), after, goal,
                                                    deadline, transfers)[0]
    result = {}
    for menu, ids in menus.items():
        choices = np.stack([values[cid] for cid in ids])
        best = np.argmin(choices, axis=0)
        result[menu] = [dict(loss=int(choices[j, i] - root.time),
                             first=ids[j] if choices[j, i] < deadline else None)
                        for i, j in enumerate(best)]
    return result, values


def summarize(rows, cap):
    losses = {method: [row['policies'][method]['loss_upper'] for row in rows] for method in ARMS}
    refs = {menu: [row['hindsight'][menu]['loss'] for row in rows] for menu in ('ab', 'expanded')}
    summary = {}
    for method, values in losses.items():
        menu = 'ab' if method == BASE else 'expanded'
        delta = [a-b for a, b in zip(losses[BASE], values)]
        alpha = .05 / (len(CASES) * (len(ARMS)-1))
        summary[method] = dict(mean_loss=mean(values),
            arrivals=sum(row['policies'][method]['status'] == 'arrived' for row in rows),
            mean_gain_vs_base=mean(delta),
            improved=sum(d > 0 for d in delta), worsened=sum(d < 0 for d in delta),
            gain_lower_model_only=(None if method == BASE else
                continuation_lower(np.array(delta)+cap, 2*cap, alpha)-cap),
            mean_hindsight_gap=mean(value-ref for value, ref in zip(values, refs[menu])),
            mean_rule_wall_seconds=mean(row['policies'][method]['total_rule_pipeline_seconds']
                                        for row in rows),
            first_boardings=dict(Counter(str(row['policies'][method]['first']) for row in rows)))
    summary['hindsight_ab'] = dict(mean_loss=mean(refs['ab']))
    summary['hindsight_expanded'] = dict(mean_loss=mean(refs['expanded']),
        mean_gain_vs_ab=mean(a-b for a, b in zip(refs['ab'], refs['expanded'])))
    summary['full_hindsight'] = dict(mean_loss=mean(row['full_hindsight'] for row in rows))
    return summary


def row_receipt(row):
    keep = ('status', 'loss_lower', 'loss_upper', 'published', 'rejected', 'publish_at',
            'worker_seconds', 'construction_cpu_seconds', 'binding_cpu_seconds', 'rule_work',
            'total_rule_pipeline_seconds', 'missing_connections', 'trigger_trace')
    receipt = {k: row[k] for k in keep}
    receipt['rides'] = [(r['ridden_connection'], r['time'], r['next_time']) for r in row['rides']]
    receipt['first'] = receipt['rides'][0][0] if receipt['rides'] else None
    return receipt


def main():
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    old = {(*key(row), row['world']): row for row in read(V13)['rows']
           if row['panel'] == 'waiting' and row['method'] == BASE}
    inventory = {key(r): r for r in read(INVENTORY)['rows']}
    plans = {key(r): r for r in read(V10)['rows']}
    journeys = {(r['date'],r['query']):r for r in read(ARCHIVE)['rows']
                if r['method'] == 'evolving_rollout'}
    before = perf_counter()
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    result = dict(shared_fit_seconds=perf_counter()-before, cpu=cpu, roots=[])
    with TemporaryDirectory(prefix='bapr_v14_menu_') as tmp:
        before = perf_counter()
        kernel = build_temporal_kernel(Path(tmp)/'scan.so')
        result['shared_compile_seconds'] = perf_counter()-before
        for identity in CASES:
            day, query, decision = identity
            before = perf_counter()
            graph, actual, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
            journey, record = journeys[day, query], inventory[identity]
            prefix = tuple(record['next_transfer_boundary']['prefix'])
            belief, root = prepare_waiting_context(graph, metadata, journey, decision, fit, prefix,
                                                   record['recorded_incumbent']['rides'])
            goal, deadline = record['root']['destination'], journey['od']['deadline_seconds']
            plan = load_plan(plans[identity]['fixed_primary_tail_rescue']['plan'], prefix)
            scan = TemporalScan(graph, kernel)
            parameters = {c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled'])
                          for c in graph.connections}
            setup = perf_counter()-before
            ab = tuple(sorted({plan.primary[0], plan.rescue[0]}))
            before, before_cpu = perf_counter(), process_time()
            options = screen_public_itineraries(graph, root, goal, deadline, ab, WINDOW)
            screen, screen_cpu = perf_counter()-before, process_time()-before_cpu
            menus = {'ab': ab, 'expanded': tuple(sorted((*ab, *options)))}
            controllers = {BASE: compile_arms(graph, plan, root, deadline)[BASE]}
            for method, margin in ARMS.items():
                if method == BASE:
                    continue
                before, before_cpu = perf_counter(), process_time()
                controller = PublicRescueCandidates(graph, plan, root, deadline, options, margin=margin)
                controllers[method] = dict(controller=controller,
                    construction_seconds=screen+perf_counter()-before,
                    construction_cpu_seconds=screen_cpu+process_time()-before_cpu)
            root_row = dict(case=identity, stop=root.stop, time=root.time, deadline=deadline,
                            setup_seconds=setup, candidate_screen_seconds=screen,
                            original_plan_seconds=plans[identity]['fixed_primary_tail_rescue']['wall_seconds'],
                            menus=menus, extra_paths={cid: dict(trip=graph.connections[cid].trip_id,
                               path=option.path, scheduled_arrival=option.scheduled_arrival)
                               for cid, option in options.items()})
            before = perf_counter()
            rng = np.random.default_rng(np.random.SeedSequence(
                [20261113, int(day.replace('-', '')), query, decision]))
            clocks = sample_clocks(belief, parameters, rng, WORLDS,
                                  templates=template_trip_map(belief, parameters))
            root_row['draw_seconds'] = perf_counter()-before
            rows, actual_row = [], None
            for world in range(-1, WORLDS):
                if world == -1:
                    scene = actual
                else:
                    dep, arr, cnl = (x[world] for x in clocks)
                    scene = JourneyScenario({c.id: VehicleEvent(int(dep[c.id])-c.dep_time, bool(cnl[c.id]),
                        int(dep[c.id]), int(arr[c.id])) for c in graph.connections}, minute=60, publications={})
                model = build_model(graph, scene, root, goal, deadline)
                order = list(ARMS)
                offset = (world + decision) % len(order)
                policies = {}
                for method in order[offset:] + order[:offset]:
                    policies[method] = row_receipt(replay_arm(graph, plan, root, goal, deadline, model,
                        model.scenarios[0], method, controllers[method],
                        actual=actual if world == -1 else None, metadata=metadata if world == -1 else None))
                baseline = policies[BASE]
                archived = old[(*identity, world)]
                for field in ('status', 'loss_upper', 'loss_lower', 'publish_at'):
                    assert baseline[field] == archived[field], (identity, world, field, baseline[field], archived[field])
                if world == -1:
                    actual_row = policies
                else:
                    rows.append(dict(world=world, policies=policies))
            before = perf_counter()
            ready = max(root.time, root.last_arrival + graph.stops[root.stop].min_transfer_time)
            publish = sorted({p['publish_at'] for row in rows for p in row['policies'].values()})
            assert publish[-1] <= ready, 'A policy could miss an opportunity during computation'
            refs, first_values = hindsight(graph, scan, clocks, root, goal, deadline, menus)
            transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
            full = scan.platform_arrivals(*clocks, root, goal, deadline, transfers)-root.time
            for i, row in enumerate(rows):
                row['hindsight'] = {menu: refs[menu][i] for menu in menus}
                row['full_hindsight'] = int(full[i])
                assert full[i] <= refs['expanded'][i]['loss'] <= refs['ab'][i]['loss']
                for method, policy in row['policies'].items():
                    menu = 'ab' if method == BASE else 'expanded'
                    assert policy['first'] is None or policy['first'] in menus[menu]
                    assert refs[menu][i]['loss'] <= policy['loss_upper'], (identity, i, method)
                    first = policy['first']
                    policy['hindsight_same_first'] = (deadline-root.time if first is None else
                                                      int(first_values[first][i]-root.time))
                    assert refs[menu][i]['loss'] <= policy['hindsight_same_first'] <= policy['loss_upper']
            root_row.update(rows=rows, actual=actual_row, publication_clocks=publish, earliest_board=ready,
                            hindsight_seconds=perf_counter()-before,
                            summary=summarize(rows, deadline-root.time))
            result['roots'].append(root_row)
            write_json(HERE/'menu_info.json', result)
            print(identity, 'extra first connections', len(options),
                  {m: round(s['mean_loss'], 2) for m, s in root_row['summary'].items()}, flush=True)


if __name__ == '__main__':
    main()
