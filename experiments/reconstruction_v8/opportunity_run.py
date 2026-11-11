"""Test first-exit survival bounds with zero physical travel-time floors."""

from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter, process_time
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
from src.evolving_trip_belief import EvolvingTripBelief
from src.information_planner import ExactPlanner
from src.opportunity_bounds import FirstExitOpportunities, OrderedTripOpportunities
from src.platform_transit import build_platform_model
from src.recourse_itinerary_router import expand_trip_blocks, nominal_itinerary
from src.risk_bounds import CommitmentBounds
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


def compressed_one_leg_marginal(bounds, cid, start, deadline):
    """The exit summary uses the success arrival or failure at the start.

    Successful departure values are interchangeable for this summary. Replacing
    them by start also makes the static skip-after-departure bound conservative.
    """
    trip, slot = bounds.belief._positions[cid, 'departure']
    law = bounds._law(trip)
    catches = ((law.times[slot] >= start) & (law.times[slot] <= deadline)
               & ~law.canceled[slot])
    successes = (law.marginal(slot) * catches) @ bounds._posterior_edge(trip, slot)
    successes *= law.times[slot + 1] < deadline
    result = [(float(p), {cid: VehicleEvent(0, False, start, int(arrival))})
              for p, arrival in zip(successes, law.times[slot + 1]) if p > 0.]
    failure = max(0., 1. - float(successes.sum()))
    if failure:
        result.append((failure, {cid: VehicleEvent(999, True, start, start)}))
    return result


def same_trip_repeated_opportunity():
    """A repeated vehicle is not a fresh independent boarding opportunity."""
    class TimetableClockBelief(EvolvingTripBelief):
        def _grid(self, slot, shifts, anchors, cancellation_possible):
            return super()._grid(slot, np.array([0]), anchors, cancellation_possible)

    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=0))
    for row in [('A', 'A', 0, 1, 0, 2), ('B', 'B', 1, 2, 3, 4),
                ('B', 'B', 2, 1, 4, 5), ('B', 'B', 1, 2, 6, 7),
                ('C', 'C', 1, 2, 3, 4), ('D', 'D', 1, 2, 9, 10)]:
        graph.add_connection(Connection(-1, *row))
    rates = {'A': 0., 'B': .6, 'C': .8, 'D': 0.}
    parameters = {route: dict(mean=0., std=1., cancel_rate=q) for route, q in rates.items()}
    belief = TimetableClockBelief(graph, minute=1, public_occurrences=False, cancel_persistence=1.)
    bounds = CommitmentBounds(belief, parameters, graph, 2, 12)
    envelope = FirstExitOpportunities(OrderedTripOpportunities(bounds), graph, 2)
    local = {0: VehicleEvent(0, False, 0, 2)}
    value = envelope.evaluate([(1., local)], (0,), 0, 12, range(1, 13))
    bank = []
    for b in (True, False):
        for c in (True, False):
            canceled = {'A': False, 'B': b, 'C': c, 'D': False}
            weight = (rates['B'] if b else 1. - rates['B']) * (rates['C'] if c else 1. - rates['C'])
            bank.append((weight, JourneyScenario({leg.id: VehicleEvent(
                999 if canceled[leg.route] else 0, canceled[leg.route], leg.dep_time, leg.arr_time)
                for leg in graph.connections}, minute=1, publications={})))
    exact = ExactPlanner(build_platform_model(graph, bank, 0, 2, 0, 12)).solve_compressed().value
    if abs(value.lower - exact) > 1e-10:
        raise AssertionError('The synthetic first-exit value must match the full causal reference')
    return dict(**asdict(value), full_optimal=exact, enumerated_worlds=len(bank))


def sampled_exit_relaxation(envelope, scene, path, state, deadline, incoming_cid):
    """A world-by-world first exit, with free travel after the first moved edge."""
    trip, inventory = envelope._inventory(path, state.unavailable_ids)
    local = {slot.cid: scene.events[slot.cid] for slot in envelope.opportunities.bounds.belief._trips[trip]}
    arrival, exits = envelope._summary(local, path, trip, inventory,
                                       state.time, deadline, incoming_cid)
    best = None
    for cid, ready in exits:
        event = scene.events[cid]
        if event.canceled or event.departure < ready:
            continue
        value = event.arrival + envelope.static.distance[envelope.graph.connections[cid].arr_stop]
        if value < arrival:
            arrival, best = value, cid
    return arrival, best


def main():
    from src.temporal_oracle import earliest_arrival

    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    archive_path = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    archive = json.loads(archive_path.read_text())
    journeys = [r for r in archive['rows'] if r['method'] == 'evolving_rollout']
    protocol = dict(scope='Diagnostic on previously exposed city histories; no new dates or independent confirmation.',
        source=str(source.relative_to(ROOT)), archive=str(archive_path.relative_to(ROOT)),
        training_dates=fit['train_dates'], current_clock='Original committed replay; lower relaxes boarding to the physical first-exit union.',
        selection='First two fresh-onboard, nonterminal consecutive next legs per frozen archive journey. Uses public arrival and timetable only.',
        static='Zero hard runtime and transfer floors, complete original graph, no footpaths in this archived contract.',
        threshold_seconds=60, tolerances_seconds=[1, 30, 60],
        witness='Force next same-vehicle leg, then route on root nominal values from its nominal arrival. Evaluate a fixed feasible V6 itinerary in the same frozen HMM, stop on any miss. If no tail, retain the deadline-cap witness.',
        approximations='Compress departure of each successful first leg to root time only for the first-exit calculation; static skip-after-departure baseline is consequently conservative.',
        paired_world_diagnostic='Two fixed private draws per history, compare exact first-exit relaxation with a full temporal perfect-information route on each identical world. Zero transfer floor and no footpaths; this is not used as the model bound.',
        sources={name:(ROOT / name).read_text() for name in ('src/opportunity_bounds.py',
            'src/risk_bounds.py', 'src/temporal_oracle.py', 'experiments/reconstruction_v8/opportunity_run.py')})
    (HERE / 'opportunity_city_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows = []
    for day in data['holdout_dates']:
        graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                      for c in graph.connections}
        menu = {c.id for c in graph.connections}
        for journey in [j for j in journeys if j['date'] == day]:
            destination, deadline = stops[journey['od']['destination']], journey['od']['deadline_seconds']
            cases = 0
            for number, decision in enumerate(journey['decisions'][1:], 1):
                previous = journey['decisions'][number - 1]
                if (previous['outcome'] != 'board' or decision['clock_seconds'] != previous['end_seconds']
                        or decision['clock_seconds'] >= deadline):
                    continue
                incoming = graph.connections[previous['action']]
                before, cpu_before = perf_counter(), process_time()
                belief, state = context(graph, metadata, journey, number, fit)
                trip, arrival_slot = belief._positions[incoming.id, 'arrival']
                slots = belief._trips[trip]
                if arrival_slot + 1 >= len(slots):
                    continue
                cid = slots[arrival_slot + 1].cid
                edge = graph.connections[cid]
                if (edge.dep_stop != state.stop or edge.arr_stop == destination
                        or edge.dep_time > state.time + 1500 or cid in state.unavailable_ids):
                    continue
                context_seconds = perf_counter() - before
                bounds = CommitmentBounds(belief, parameters, graph, destination, deadline)
                bounds_setup_seconds = perf_counter() - before - context_seconds
                before_opportunity = perf_counter()
                law = compressed_one_leg_marginal(bounds, cid, state.time, deadline)
                compute = OrderedTripOpportunities(bounds)
                envelope = FirstExitOpportunities(compute, graph, destination)
                grid = [*range(state.time + 60, deadline, 60), deadline]
                value = envelope.evaluate(law, (cid,), state.time, deadline, grid,
                                          incoming_cid=incoming.id, used=state.unavailable_ids)
                opportunity_seconds = perf_counter() - before_opportunity
                # Generate a feasible witness only from the frozen public model.
                before_upper = perf_counter()
                nominal = belief.mean_events(parameters)
                mean = nominal[cid].arrival
                path = None
                if mean is not None and round(mean) < deadline:
                    next_state = replace(state, stop=edge.arr_stop, time=max(state.time, round(mean)),
                        last_arrival=max(state.time, round(mean)),
                        unavailable_ids=state.unavailable_ids | {cid})
                    tail = nominal_itinerary(graph, next_state, destination, nominal,
                                             menu, deadline, 1500)
                    if tail:
                        path = expand_trip_blocks(graph, (cid, *tail), destination, state.unavailable_ids)
                upper = (bounds.itinerary(state, path).upper if path
                         else float(deadline - state.time))
                if value.lower > upper + 1e-8:
                    raise AssertionError('Incompatible lower and witness upper')
                row = dict(date=day, query=journey['query'], decision=number, clock=state.time,
                    first_connection=cid, stop=state.stop, boundary=edge.arr_stop, destination=destination,
                    upper=upper, gap=upper - value.lower, witness=list(path or ()), **asdict(value),
                    context_seconds=context_seconds, bounds_setup_seconds=bounds_setup_seconds,
                    opportunity_seconds=opportunity_seconds, witness_seconds=perf_counter() - before_upper,
                    full_context_cpu_seconds=process_time() - cpu_before,
                    work=compute.stats, analytic_work=bounds.stats,
                    lower_assumptions='Current-grid independent trips, no cancellation inference beyond the frozen law.')
                seed = [20261111, int(day.replace('-', '')), journey['query'], number]
                rng = np.random.default_rng(np.random.SeedSequence(seed))
                world_before = perf_counter()
                world_samples = belief.sample(parameters, rng, 2)
                sample_seconds = perf_counter() - world_before
                diagnostic = []
                for sample_id, scene in enumerate(world_samples):
                    world_before = perf_counter()
                    early, exit_cid = sampled_exit_relaxation(envelope, scene, (cid,), state,
                                                             deadline, incoming.id)
                    exit_seconds = perf_counter() - world_before
                    world_before = perf_counter()
                    arrival, route = earliest_arrival(graph, scene, state, destination, deadline)
                    full_seconds = perf_counter() - world_before
                    if early > arrival:
                        raise AssertionError('A sampled first-exit relaxation exceeds full temporal cost')
                    diagnostic.append(dict(sample=sample_id, first_exit_seconds=early-state.time,
                        external_exit_connection=exit_cid,
                        external_exit_arrival_stop=(graph.connections[exit_cid].arr_stop
                                                    if exit_cid is not None else None),
                        full_temporal_seconds=arrival-state.time, full_path=list(route),
                        extra_seconds_after_free_tail=arrival-early,
                        first_exit_lookup_seconds=exit_seconds, full_temporal_wall_seconds=full_seconds))
                row.update(paired_worlds=diagnostic, random_seed=seed, sample_setup_seconds=sample_seconds)
                rows.append(row)
                cases += 1
                print(day, journey['query'], number, 'L-static', value.static_lower, 'L-exit', value.lower,
                    'U', upper, 'seconds', round(opportunity_seconds, 3), flush=True)
                if cases == 2:
                    break
    paired = [w for r in rows for w in r['paired_worlds']]
    result = dict(protocol='opportunity_city_protocol.json', n=len(rows), rows=rows,
        synthetic=same_trip_repeated_opportunity(),
        summary=dict(mean_static_lower=float(np.mean([r['static_lower'] for r in rows])),
            mean_opportunity_lower=float(np.mean([r['lower'] for r in rows])),
            mean_trip_union_lower=float(np.mean([r['trip_union_lower'] for r in rows])),
            mean_edge_union_lower=float(np.mean([r['edge_union_lower'] for r in rows])),
            median_opportunity_seconds=float(np.median([r['opportunity_seconds'] for r in rows])),
            paired_world_count=len(paired),
            paired_mean_first_exit_seconds=float(np.mean([w['first_exit_seconds'] for w in paired])),
            paired_mean_full_temporal_seconds=float(np.mean([w['full_temporal_seconds'] for w in paired])),
            paired_mean_free_tail_relaxation_seconds=float(np.mean([w['extra_seconds_after_free_tail'] for w in paired])),
            gap_accept_counts={str(t):sum(r['gap'] <= t for r in rows) for t in (1, 30, 60)}))
    (HERE / 'opportunity_city_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['summary'], flush=True)


if __name__ == '__main__':
    main()
