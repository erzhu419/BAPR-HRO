"""Paired conditional-model and actual-clock tests of downstream transfers."""

from collections import Counter
import argparse
from dataclasses import asdict
from functools import cache
import gzip
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from experiments.reconstruction_v23.run_transfer_screen import TRAINING, model, same_vehicle
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v24.pair_decision import METHODS, PairDecision
from experiments.reconstruction_v24.screen import HERE, SOURCE
from src.evolving_trip_belief import _SampleBank
from src.recourse_itinerary_router import trip_key
from src.transfer_opportunity import RouteOption, TransferOpportunityScores


N_WORLDS = 64


def options_from(rows):
    return {row['first_cid']: RouteOption(tuple(row['first_path']),
                {path[0]: tuple(path) for path in row['tail_paths']}) for row in rows}


def ranks_from(graph, rows):
    return {row['first_cid']: (row['scheduled_arrival']-graph.connections[row['first_cid']].dep_time,
                               row['first_cid']) for row in rows}


def first_event(events, cids, ready, deadline, ranks):
    if any(cid not in events for cid in cids):
        return None, 'unresolved_departure'
    offers = [(events[cid].departure, ranks[cid], cid) for cid in cids
              if not events[cid].canceled and ready <= events[cid].departure < deadline]
    return (min(offers)[2], 'offer') if offers else (None, 'no_offer')


def same_trip_replay(graph, events, path, ready, deadline, metadata=None):
    if metadata is not None and not same_vehicle(metadata, path):
        return None, 'ambiguous_vehicle'
    time = ready
    for cid in path:
        event = events.get(cid)
        if event is None:
            return None, 'unresolved_endpoint'
        if event.canceled or event.departure < time or event.departure >= deadline or event.arrival > deadline:
            return deadline, 'deadline'
        time = event.arrival
    return time, 'arrived'


def route_replay(graph, events, option, ready, deadline, metadata=None):
    clock, status = same_trip_replay(graph, events, option.first_path, ready, deadline, metadata)
    if status != 'arrived' or not option.tails:
        return clock, status, None
    stop = graph.connections[option.first_path[-1]].arr_stop
    ready = clock+graph.stops[stop].min_transfer_time
    ranks = {cid: (graph.connections[path[-1]].arr_time-graph.connections[cid].dep_time, cid)
             for cid, path in option.tails.items()}
    tail, status = first_event(events, option.tails, ready, deadline, ranks)
    if status != 'offer':
        return (None if status.startswith('unresolved') else deadline), status, None
    clock, status = same_trip_replay(graph, events, option.tails[tail], ready, deadline, metadata)
    return clock, status, tail


def public_case(events, cids, entry, ready, deadline, ranks):
    first, status = first_event(events, cids, ready, deadline, ranks)
    if status != 'offer':
        return dict(status=status)
    clock = events[first].departure
    visible = tuple(cid for cid in cids if not events[cid].canceled and events[cid].departure == clock)
    # This is an observation filter. Actual clocks before entry or after now
    # never enter the scorer, including for canceled or not-yet-seen vehicles.
    seen = {cid for cid in cids if not events[cid].canceled
            and entry <= events[cid].departure <= clock}
    other = next((cid for cid in cids if cid not in seen), None)
    return dict(status='offer', clock=clock, visible=visible, other=other)


def replay(graph, events, root, pair, full, decision, entry, ready, metadata=None):
    deadline, start = root['deadline'], root['start']
    case = public_case(events, tuple(pair), entry, ready, deadline, decision.ranks)
    out, charge = {}, {}
    for method in METHODS:
        tick = perf_counter()
        if case['status'] == 'offer':
            cid, fallback = decision.choose(case['visible'], case['other'], case['clock'], method)
        else:
            cid, fallback = None, None
        charge[method] = perf_counter()-tick
        if cid is None:
            clock = None if case['status'].startswith('unresolved') else deadline
            status, tail = case['status'], None
        else:
            # Waiting rejects the visible vehicle. The chosen unseen vehicle
            # may already have left before entry: it must not be boarded then.
            attempt = case['clock'] if cid in case['visible'] else case['clock']+1
            clock, status, tail = route_replay(graph, events, pair[cid], attempt, deadline, metadata)
        out[method] = dict(action=cid, tail=tail, loss=None if clock is None else clock-start,
                           arrived=status == 'arrived', status=status, fallback=fallback)
    tick = perf_counter()
    cid, status = first_event(events, tuple(full), ready, deadline, ranks_from(graph, root['full_options']))
    charge['full_first_offer'] = perf_counter()-tick
    if cid is None:
        clock = None if status.startswith('unresolved') else deadline
        tail = None
    else:
        clock, status, tail = route_replay(graph, events, full[cid], ready, deadline, metadata)
    out['full_first_offer'] = dict(action=cid, tail=tail, loss=None if clock is None else clock-start,
                                  arrived=status == 'arrived', status=status)
    return dict(public=case, outcomes=out, action_wall=charge)


def selected_worlds(belief, params, trips, n, seed):
    rng = np.random.default_rng(seed)
    uniforms = {trip: rng.random((n, len(belief._trips[trip])))[:, ::-1]
                for trip in sorted(trips, key=str)}
    bank = _SampleBank(belief, params, uniforms)
    worlds = [{} for _ in range(n)]
    for trip in uniforms:
        for events, draw in zip(worlds, bank.trip_events(trip), strict=True):
            events.update(draw)
    return worlds


def compare(rows, left, right):
    paired = [row for row in rows if all(row['outcomes'][m]['loss'] is not None for m in (left, right))]
    differences = [row['outcomes'][left]['loss']-row['outcomes'][right]['loss'] for row in paired]
    return dict(pairs=len(paired), mean_gain=mean(differences) if differences else None,
                wins=sum(x > 0 for x in differences), harms=sum(x < 0 for x in differences),
                changed=sum(row['outcomes'][left]['action'] != row['outcomes'][right]['action'] for row in paired),
                mean_loss={m: mean(row['outcomes'][m]['loss'] for row in paired) if paired else None
                           for m in (left, right)},
                arrivals={m: sum(row['outcomes'][m]['arrived'] for row in paired) for m in (left, right)})


def groups(rows):
    return {f'{right}_vs_{left}': compare(rows, left, right)
            for left, right in (('first_offer', 'clock_nominal'), ('first_offer', 'exact'),
                                ('clock_nominal', 'exact'), ('mean_arrival', 'exact'),
                                ('full_first_offer', 'exact'), ('first_offer', 'full_first_offer'))}


def run_root(graph, actual, metadata, belief, params, root, index, n_worlds=N_WORLDS):
    started = perf_counter()
    pair, full = options_from(root['options']), options_from(root['full_options'])
    ranks = ranks_from(graph, root['options'])
    compiler = TransferOpportunityScores(graph, belief, params, root['deadline'], pair)
    decision = PairDecision(compiler, ranks, root['scheduled_entry'], root['scheduled_ready'], root['deadline'])
    setup = perf_counter()-started
    # Charge the independent nominal branch before constructing exact tables.
    started = perf_counter()
    for cid in pair:
        times, mass, _ = decision.distributions[cid]
        for t, p in zip(times, mass, strict=True):
            if p > 0 and root['scheduled_entry'] <= t < root['deadline']:
                decision.score(cid, int(t), 'clock_nominal')
    nominal_compile = perf_counter()-started
    started = perf_counter()
    for cid in pair:
        compiler.board_score(cid, graph.connections[cid].dep_time)
    exact_compile = perf_counter()-started
    started = perf_counter()
    exact_model = decision.expectation()
    expectation_wall = perf_counter()-started
    exact_model['expected_loss'] = {m: value-root['start'] for m, value in exact_model['expected_arrival'].items()}
    # This test detects a policy/evaluator disagreement on the exhaustive
    # two-offer finite law, for which exact board-or-wait must be optimal.
    if any(exact_model['expected_loss']['exact'] > value+1e-6
           for value in exact_model['expected_loss'].values()):
        raise AssertionError('Exact pair decision is worse in its own enumerated model')
    trips = {trip_key(graph.connections[cid]) for option in full.values()
             for path in (option.first_path, *option.tails.values()) for cid in path}
    started = perf_counter()
    worlds = selected_worlds(belief, params, trips, n_worlds, [20261112, 24, index]) if n_worlds else []
    sample_wall = perf_counter()-started
    started = perf_counter()
    sampled = [replay(graph, events, root, pair, full, decision, root['scheduled_entry'], root['scheduled_ready'])
               for events in worlds]
    sample_replay_wall = perf_counter()-started
    entry, status = (root['start'], 'arrived')
    if root['prefix']:
        entry, status = same_trip_replay(graph, actual.events, root['prefix'], root['start']+30,
                                         root['deadline'], metadata)
    if status == 'arrived':
        ready = entry+max(30, graph.stops[root['stop']].min_transfer_time) if root['prefix'] else entry+30
        actual_decision = PairDecision(compiler, ranks, entry, ready, root['deadline'])
        actual_result = replay(graph, actual.events, root, pair, full, actual_decision, entry, ready, metadata)
    else:
        actual_result = dict(prefix_status=status, outcomes={m: dict(action=None, loss=None, arrived=False)
            for m in (*METHODS, 'full_first_offer')})
    return dict(root_id=root['root_id'], day=root['day'], root_kind=root['root_kind'],
        pair_omits_best_direct=root.get('pair_omits_best_direct'), pair_menu=[asdict(o) for o in pair.values()],
        model=exact_model, sample_summary=groups(sampled), samples=sampled, actual=actual_result,
        wall=dict(shared_setup=setup, nominal_compile=nominal_compile, exact_compile_increment=exact_compile,
                  exact_history_integration=expectation_wall, sampling=sample_wall, sample_replay=sample_replay_wall))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=60, choices=(60, 20, 10))
    parser.add_argument('--worlds', type=int, default=N_WORLDS)
    parser.add_argument('--law', choices=('original', 'empirical'), default='original')
    args = parser.parse_args()
    output_dir = (HERE / f'route_empirical{args.step}' if args.law == 'empirical'
                  else HERE if args.step == 60 else HERE / f'grid{args.step}')
    output_dir.mkdir(exist_ok=True)
    total = perf_counter()
    selection = json.loads((HERE/'selection.json').read_text())
    data, _ = load_city(SOURCE)
    training, _ = load_city(TRAINING)
    fit = fit_training(training, retain_no_pickup=True)
    if args.law == 'empirical':
        from src.empirical_duration_belief import EmpiricalDurationBelief, fit_empirical_durations
        duration_fit = fit_empirical_durations(training, retain_no_pickup=True)
        (output_dir/'duration_fit.json').write_text(json.dumps(duration_fit, indent=2)+'\n')
    shared = perf_counter()-total
    results, skipped, setup, duration_support = [], [], {}, {}
    for day, selected in selection['by_day'].items():
        tick = perf_counter()
        graph, actual, _, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        base, params = model(graph, fit)
        if args.law == 'empirical':
            base = EmpiricalDurationBelief(graph, duration_fit=duration_fit, minute=60,
                public_occurrences=False, grid_step_seconds=args.step, **fit['increments'])
        roots = [r for r in selected['roots'] if len(r['options']) == 2]
        trips = {trip_key(graph.connections[cid]) for root in roots for option in root['full_options']
                 for path in (option['first_path'], *option['tail_paths']) for cid in path}
        belief = base.with_grid_step(args.step, trips=trips) if args.step == 60 else None
        setup[day] = perf_counter()-tick
        skipped += [dict(root_id=r['root_id'], reason='no_downstream_transfer_pair')
                    for r in selected['roots'] if len(r['options']) != 2]
        for root in roots:
            # Fine-grid sensitivity needs only this root's small trip menu;
            # discard its dense message cache before advancing to the next.
            if args.step != 60:
                local_trips = {trip_key(graph.connections[cid]) for option in root['full_options']
                               for path in (option['first_path'], *option['tail_paths']) for cid in path}
                belief = base.with_grid_step(args.step, trips=local_trips)
            result = run_root(graph, actual, metadata, belief, params, root, len(results), args.worlds)
            results.append(result)
            print(root['root_id'], result['model']['expected_loss'], result['actual']['outcomes'], flush=True)
            with gzip.open(output_dir/'results.json.gz', 'wt') as handle:
                json.dump(dict(results=results, skipped=skipped, shared_setup_wall=shared,
                               day_setup_wall=setup, complete=False), handle, separators=(',', ':'))
        if args.law == 'empirical':
            duration_support[day] = dict(used_groups=sorted(base.used_duration_groups),
                fallback_groups=sorted(base.fallback_duration_groups),
                pooled_groups=sorted(base.pooled_duration_groups),
                residual_groups=sorted(base.residual_duration_groups))
    common = [r['actual'] for r in results if all(o['loss'] is not None for o in r['actual']['outcomes'].values())]
    summary = dict(roots=len(results), worlds_per_root=args.worlds, grid_step_seconds=args.step, law=args.law,
        model_expected_loss={m: mean(r['model']['expected_loss'][m] for r in results) for m in METHODS},
        model_gain_by_root=[dict(root_id=r['root_id'], exact_vs_nominal=
                r['model']['expected_loss']['clock_nominal']-r['model']['expected_loss']['exact'],
                exact_vs_first=r['model']['expected_loss']['first_offer']-r['model']['expected_loss']['exact'])
                for r in results],
        actual=groups([r['actual'] for r in results]),
        actual_common=dict(roots=len(common), mean_loss={m: mean(r['outcomes'][m]['loss'] for r in common)
                           for m in (*METHODS, 'full_first_offer')},
                           arrivals={m: sum(r['outcomes'][m]['arrived'] for r in common)
                                     for m in (*METHODS, 'full_first_offer')}),
        sampled=groups([s for r in results for s in r['samples']]),
        by_day={d: dict(roots=sum(r['day'] == d for r in results),
                       actual=groups([r['actual'] for r in results if r['day'] == d]),
                       sampled=groups([s for r in results if r['day'] == d for s in r['samples']]))
                for d in selection['by_day']},
        wall={k: sum(r['wall'][k] for r in results) for k in results[0]['wall']},
        shared_setup_wall=shared, day_setup_wall=setup, duration_support=duration_support,
        total_wall=perf_counter()-total)
    output = dict(results=results, skipped=skipped, summary=summary, complete=True)
    with gzip.open(output_dir/'results.json.gz', 'wt') as handle:
        json.dump(output, handle, separators=(',', ':'))
    (output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
