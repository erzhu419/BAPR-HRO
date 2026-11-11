"""Frozen whole-journey G2 ablations with a common committed publisher."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from math import pi
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v8.budget_run import SOURCE, write_json
from src.committed_transit import CommittedState, build_committed_model
from src.continuation_gate import check_continuation
from src.evolving_trip_belief import EvolvingTripBelief
from src.first_branch_belief import FirstBranchSampler
from src.indexed_nominal_tail import IndexedNominalTailRouter
from src.issued_continuation import ContinuationUpdate, replay_continuation
from src.recourse_itinerary_router import (RecourseItineraryRouter, _ConditionalNominal, build_trip_blocks,
                                           expand_trip_blocks, nominal_itinerary, trip_key)
from src.simulator import LazyTripScenario
from src.tail_rescue import TailRescueBounds, TailRescuePlan, compile_tail_rescue, freeze_tail_plan
from src.temporal_corridor import TemporalCorridor
from src.temporal_scan import TemporalScan, build_temporal_kernel

METHODS = ('off64', 'terminal64', 'middle64', 'off128', 'terminal128', 'middle128',
           'off512', 'compiled_nominal', 'analytic')
SCHEDULED = ('scheduled_off64', 'scheduled_terminal64', 'scheduled_middle64')
TOLERANCE, SAMPLES, ALPHA = 60, 128, .05


class Worker:
    def __init__(self, graph, fit, origin, goal, start, deadline, seed, method, scan):
        self.scheduled = method.startswith('scheduled_')
        method = method.removeprefix('scheduled_')
        self.graph, self.goal, self.deadline, self.seed, self.method = graph, goal, deadline, seed, method
        self.parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                           for c in graph.connections}
        self.belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
        kwargs = dict(event_belief=self.belief, max_time=deadline-start, seed=seed, samples=4)
        budget = int(''.join(c for c in method if c.isdigit())) if method[-1].isdigit() else 64
        self.router = IndexedNominalTailRouter(graph, self.parameters, algorithm='uct', work_budget=budget, **kwargs)
        self.router.online_mean = self.router.online_cancel = False
        self.router._initial_cancels = {r: p['cancel_rate'] for r, p in self.parameters.items()}
        self.router.route(origin, goal, start)
        self.analytic = None
        if method == 'analytic':
            self.analytic = RecourseItineraryRouter(graph, self.parameters, **kwargs)
            self.analytic.online_mean = self.analytic.online_cancel = False
            self.analytic._initial_cancels = self.router._initial_cancels
            self.analytic.route(origin, goal, start)
        self.trip_blocks = build_trip_blocks(graph)
        self.corridor = TemporalCorridor(scan) if scan is not None else None
        self.pending = []
        self.alpha_calls = 0

    def bounds(self):
        return TailRescueBounds(self.belief, self.parameters, self.graph, self.goal, self.deadline)

    def nominal_path(self, state, events):
        return nominal_itinerary(self.graph, state, self.goal, events, self.router.menu, self.deadline,
                                 1500, trip_blocks=self.trip_blocks) or ()

    def initial(self, state):
        path = self.nominal_path(state, self.belief.mean_events(self.parameters))
        if not path:
            return None
        if self.method == 'compiled_nominal':
            return freeze_tail_plan(self.bounds(), path, first_action_now=False)
        return compile_tail_rescue(self.bounds(), state, path, first_action_now=False,
                                   trip_blocks=self.trip_blocks)[0]

    def observe(self, before, after, terminal):
        self.pending.append((before, after, terminal))

    def viable(self, state, plan):
        if not plan.primary:
            return False
        c = self.graph.connections[plan.primary[0]]
        if c.dep_stop != state.stop or c.id in state.unavailable_ids:
            return False
        if not c.pickup_allowed and c.trip_id != state.current_trip:
            return False
        dep = self.belief.hard_support(c.id, 'departure', self.parameters)
        arr = self.belief.hard_support(c.id, 'arrival', self.parameters)
        transfer = (self.graph.stops[state.stop].min_transfer_time
                    if state.current_trip is not None and c.trip_id != state.current_trip else 0)
        ready = max(state.time, state.last_arrival + transfer)
        latest = min(self.deadline, ready + 720) if plan.first_action_now else self.deadline
        return (dep is not None and arr is not None and ready < self.deadline
                and max(ready, dep[0]) <= min(dep[1], arr[1], latest))

    def selected_plan(self, state, cid):
        bounds = self.bounds()
        first = self.graph.connections[cid]
        if first.arr_stop == self.goal:
            return freeze_tail_plan(bounds, (cid,))
        risk = bounds.commit(state, cid)
        if not risk.catch_probability:
            return freeze_tail_plan(bounds, (cid,))
        law = FirstBranchSampler(self.belief, self.parameters, cid, risk.ready, self.deadline)
        nominal = self.belief.mean_events(self.parameters)
        conditional = _ConditionalNominal(nominal, self.belief, trip_key(first), law.conditional_law(True))
        clock = min(self.deadline, round(conditional[cid].arrival))
        after = replace(state, stop=first.arr_stop, time=clock, last_arrival=clock, current_trip=trip_key(first),
                        unavailable_ids=state.unavailable_ids | {cid})
        tail = self.nominal_path(after, conditional)
        path = expand_trip_blocks(self.graph, (cid, *tail), self.goal, state.unavailable_ids,
                                  self.trip_blocks) if tail else None
        if path is None:
            return freeze_tail_plan(bounds, (cid,))
        return compile_tail_rescue(bounds, state, path, trip_blocks=self.trip_blocks)[0]

    def update(self, state, incumbent, ordinal):
        work = dict(gate=None, search_seconds=0., candidate_seconds=0., transitions=0)
        started = perf_counter()
        for record in self.pending:
            self.router._advance_belief(self.belief, *record)
        self.pending.clear()
        work['posterior_update_seconds'] = perf_counter() - started
        if self.method == 'compiled_nominal':
            return ContinuationUpdate(retain=incumbent is not None, defer=incumbent is None, work=work)
        if self.scheduled and incumbent is not None and incumbent.action(state, self.graph)[0] == 'select':
            # Its due commitment starts before any measured worker can publish.
            work['scheduled_skip'] = True
            return ContinuationUpdate(retain=True, work=work)
        if self.method.startswith(('terminal', 'middle')):
            started = perf_counter()
            next_alpha = ALPHA * 6. / (pi * (self.alpha_calls + 1))**2
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 10, ordinal]))
            gate = check_continuation(self.bounds(), state, incumbent, self.corridor, rng,
                                      tolerance=TOLERANCE, samples=SAMPLES, alpha=next_alpha,
                                      allow_middle=self.method.startswith('middle'))
            work.update(gate=gate, gate_seconds=perf_counter() - started)
            if gate is not None:
                self.alpha_calls += int(gate['alpha'] > 0)
                if gate['accepted']:
                    return ContinuationUpdate(retain=True, work=work)
        if self.analytic is not None:
            self.analytic.information_state = state
            self.analytic._path = ()
            started = perf_counter()
            selected = self.analytic.select_connection(state.stop, state.time, None)
            work['search_seconds'] = perf_counter() - started
            path, rescue = self.analytic._path, self.analytic._rescue
            plan = TailRescuePlan(path, 0 if rescue else None, rescue, selected is not None,
                                 tuple(self.analytic._fixed_departures.items())) if path else None
            return ContinuationUpdate(plan=plan, defer=selected is None, work=work)
        self.router.information_state, self.router.query_number = state, ordinal
        started = perf_counter()
        selected = self.router.select_connection(state.stop, state.time, None)
        work['search_seconds'] = perf_counter() - started
        work['transitions'] = self.router.predictions[-1]['work']
        if selected is None:
            return ContinuationUpdate(plan=incumbent, defer=True, work=work)
        cid = selected[0].connection_id
        work['proposed_first'] = cid
        if incumbent is not None and incumbent.action(state, self.graph) == ('select', cid):
            return ContinuationUpdate(retain=True, work=work)
        started = perf_counter()
        plan = self.selected_plan(state, cid)
        work['candidate_seconds'] = perf_counter() - started
        return ContinuationUpdate(plan=plan, work=work)


def arm(graph, actual, metadata, fit, origin, goal, start, deadline, seed, method, kernel, charged):
    started, cpu = perf_counter(), process_time()
    scan = TemporalScan(graph, kernel) if method.removeprefix('scheduled_').startswith('middle') else None
    worker = Worker(graph, fit, origin, goal, start, deadline, seed, method, scan)
    state = CommittedState(origin, start, None, start, initialized=True)
    scene = LazyTripScenario(actual.events, minute=60, publications={})
    model = build_committed_model(graph, [(1., scene)], origin, goal, start, deadline, initial_state=state)
    incumbent = worker.initial(state)
    setup = perf_counter() - started

    def known(public, cid):
        if cid not in actual.events:
            return False
        if public.arrivals and graph.connections[cid].trip_id == public.current_trip:
            previous = public.arrivals[-1][0]
            a, d = metadata[previous]['arrival'], metadata[cid]['departure']
            if d['status'] == 'REAL' and (a['official_journey'] != d['official_journey']
                    or actual.events[cid].departure < public.last_arrival):
                return False
        return True

    final, missing, decisions, actions = replay_continuation(model, model.scenarios[0], incumbent, worker.update,
        worker.observe, worker.viable, known, graph=graph, deadline=deadline, minute=60, charged=charged)
    lo = final.time - start if missing is not None or final.stop == goal else deadline-start
    hi = deadline-start if missing is not None else lo
    return dict(method=method, clock='charged' if charged else 'logical', loss_lower=lo, loss_upper=hi,
        status='censored' if missing is not None else 'arrived' if final.stop == goal else 'timeout',
        missing_connection=missing, setup_seconds=setup, initial_plan=incumbent.primary if incumbent else (),
        worker_seconds=sum(row['wall_seconds'] for row in decisions),
        worker_cpu_seconds=sum(row['cpu_seconds'] for row in decisions),
        actual_seconds=perf_counter()-started, actual_cpu_seconds=process_time()-cpu,
        decisions=decisions, actions=actions)


def run_case(spec, data, fit, library, methods, charged):
    day, qi, cpu = spec
    os.sched_setaffinity(0, {cpu})
    started = perf_counter()
    graph, actual, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
    graph_seconds = perf_counter() - started
    query = data['od_queries'][qi]
    origin, goal = stops[query['origin']], stops[query['destination']]
    start, deadline = query['ready_seconds'], query['deadline_seconds']
    seed = int(day.replace('-', '')) + 1000*qi
    # Recreate an independent posterior per arm while retaining it within a journey.
    started = perf_counter()
    kernel = build_temporal_kernel(library.with_name(f'temporal_{day}_{qi}.so')) if any(
        method.removeprefix('scheduled_').startswith('middle') for method in methods) else None
    kernel_seconds = perf_counter() - started if kernel is not None else 0.
    result = []
    order = methods if qi % 2 == 0 else methods[::-1]
    for method in order:
        row = arm(graph, actual, metadata, fit, origin, goal, start, deadline, seed, method, kernel, charged)
        result.append(dict(date=day, query=qi, cpu=cpu, graph_seconds=graph_seconds,
                           kernel_seconds=kernel_seconds if method.removeprefix('scheduled_').startswith('middle') else 0.,
                           **row))
        print(day, qi, method, row['status'], row['loss_lower'], row['loss_upper'],
              round(row['actual_seconds'], 2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logical', action='store_true')
    parser.add_argument('--methods', nargs='+', choices=(*METHODS, *SCHEDULED), default=METHODS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.logical and any(method in SCHEDULED for method in args.methods):
        parser.error('Commitment scheduling needs the charged clock')
    started = perf_counter()
    data, _ = load_city(SOURCE)
    fit = fit_training(data, retain_no_pickup=True)
    fit_seconds = perf_counter() - started
    cpus = sorted(os.sched_getaffinity(0))[:4]
    cases = [(day, qi) for day in data['holdout_dates'] for qi in range(len(data['od_queries']))]
    specs = [(day, qi, cpus[i % len(cpus)]) for i, (day, qi) in enumerate(cases)]
    protocol = dict(source=str(SOURCE.relative_to(ROOT)), methods=args.methods, samples=SAMPLES,
        tolerance=TOLERANCE, alpha_per_journey=ALPHA, mode='logical' if args.logical else 'charged',
        alpha_rule='6 alpha / (pi * k)^2, conditional fresh tape per probabilistic check',
        contract='Shared no-feed commitment with preissued policy; accepted gate retains incumbent. '
                 'Fallback computes on same snapshot and publishes after measured wall time, using public revalidation.',
        scheduling='The scheduled arms bypass a worker if the same public incumbent has an irrevocable attempt due now.',
        unresolved='If an actually attempted connection is unresolved, retain elapsed/deadline loss interval.',
        training_dates=fit['train_dates'])
    write_json(args.output.with_name(args.output.stem + '_protocol.json'), protocol)
    rows = []
    with TemporaryDirectory(prefix='bapr_v10_') as tmp, ProcessPoolExecutor(max_workers=len(cpus)) as pool:
        path = Path(tmp) / 'temporal.so'
        futures = [pool.submit(run_case, spec, data, fit, path, args.methods, not args.logical) for spec in specs]
        for future in futures:
            rows.extend(future.result())
            write_json(args.output, dict(protocol=protocol, fit_seconds=fit_seconds, rows=rows))


if __name__ == '__main__':
    main()
