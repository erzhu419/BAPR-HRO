"""Read archived histories; compare full and bounded inference on two city trips.

This evaluates an inference reference, not a new city policy. Complete kernels
are deliberately built for the reference comparison; boundary operation counts
are prospective, and do not claim a production end-to-end speedup.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import ROOT, dated_graph, fit_training
from experiments.reconstruction_v5.chain_boundary_reference import BoundaryChain, ordered_service_feasible
from src.evolving_trip_belief import EvolvingTripBelief


def apply_receipt(belief, graph, metadata, decision, last_arrival):
    cid = decision['action']
    if cid is None or cid == 'wait':
        return last_arrival
    c = graph.connections[cid]
    if decision['outcome'] == 'board':
        row = metadata[cid]
        departure, arrival = row['departure']['time_seconds'], row['arrival']['time_seconds']
        assert arrival == decision['end_seconds']
        belief.observe_connection_event(cid, 'departure', departure - c.dep_time)
        belief.observe_connection_event(cid, 'arrival', arrival - c.arr_time)
        return arrival
    if decision['outcome'] == 'wait_late':
        transfer = (graph.stops[decision['stop']].min_transfer_time
                    if decision['current_trip'] is not None and decision['current_trip'] != c.trip_id else 0)
        belief.observe_wait(cid, max(decision['clock_seconds'], last_arrival + transfer), decision['end_seconds'])
    return last_arrival


def inspect_case(name, belief, parameters, cid, identity):
    trip = belief._trip_by_cid[cid]
    before = perf_counter()
    law = belief._law(trip, parameters)
    build_ms = (perf_counter() - before) * 1000
    emissions = list(law.emissions)
    # Absorb prior and first likelihood into one initial factor. This preserves
    # all full/partial exact first-slot evidence without subtracting -inf.
    emissions[0] = law.forward[0]
    initial = np.zeros(len(emissions[0]))
    before = perf_counter()
    chain = BoundaryChain(initial, law.transitions, emissions)
    setup_ms = (perf_counter() - before) * 1000
    before = perf_counter()
    truth = law.marginals
    full_smooth_ms = (perf_counter() - before) * 1000
    comparison = []
    max_error = 0.
    for slot in range(len(law.times)):
        result = chain.marginal(slot)
        max_error = max(max_error, float(np.max(abs(result.probabilities - truth[slot]))))
    for kind in ('departure', 'arrival'):
        slot = belief._positions[cid, kind][1]
        before = perf_counter()
        result = chain.marginal(slot)
        query_ms = (perf_counter() - before) * 1000
        comparison.append({'kind': kind, 'slot': slot, 'left_separator': result.left_separator,
                           'right_separator': result.right_separator,
                           'forward_steps': result.forward_steps, 'backward_steps': result.backward_steps,
                           'query_ms': query_ms})
    slots = belief._trips[trip]
    q = parameters[belief.graph.connections[cid].route]['cancel_rate']
    rho = belief.cancel_persistence
    availability = np.array([[1 - (1-rho)*q, (1-rho)*q],
                             [(1-rho)*(1-q), rho + (1-rho)*q]]) > 0
    service_steps = [np.eye(2, dtype=bool) if slot.kind == 'arrival' else availability for slot in slots[1:]]
    before = perf_counter()
    feasible, inspected = ordered_service_feasible(law.times, law.canceled, law.emissions,
                                                  [q < 1, q > 0], service_steps)
    feasible_ms = (perf_counter() - before) * 1000
    return {'name': name, **identity, 'trip': trip, 'cid': cid, 'trip_slots': len(slots),
            'grid_widths': [len(t) for t in law.times], 'full_forward_steps': len(slots)-1,
            'full_backward_steps': len(slots)-1, 'kernel_entries_supplied': sum(k.size for k in law.transitions),
            'full_law_build_ms': build_ms, 'full_smoothing_ms': full_smooth_ms,
            'boundary_setup_including_kernel_row_checks_ms': setup_ms,
            'all_slot_max_absolute_probability_error': max_error, 'queries': comparison,
            'exact_separator_slots': sorted(chain.exact),
            'linear_feasibility': {'positive': feasible, 'grid_states_inspected': inspected, 'ms': feasible_ms}}


def main():
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    archive_path = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    archive = json.loads(archive_path.read_text())
    fit = fit_training(data, retain_no_pickup=True)
    profiles = {}
    for filename in ('profile_lazy_localcache_2023-07-26_q0.json', 'runtime_bank_parity_results.json'):
        record = json.loads(Path(__file__).with_name(filename).read_text())
        profiles[filename] = ({key: record[key] for key in
                              ['graph_connections', 'trips', 'samples', 'trip_laws_cached',
                               'nominal_events_cached', 'focused_functions', 'planning']}
                             if 'trip_laws_cached' in record else record)
    summaries, selected, monotonicity = [], {}, {}
    for day in data['holdout_dates']:
        graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
        parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
        layout = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
        bad = [(trip, index) for trip, slots in layout._trips.items()
               for index, (previous, following) in enumerate(zip(slots, slots[1:]))
               if previous.scheduled > following.scheduled]
        monotonicity[day] = {'trips': len(layout._trips), 'nonmonotonic_slot_transitions': bad}
        for qi, query in enumerate(data['od_queries']):
            archived = next(row for row in archive['rows'] if row['date'] == day and row['query'] == qi
                            and row['method'] == 'evolving_rollout')
            belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
            last_arrival = query['ready_seconds']
            for number, decision in enumerate(archived['decisions']):
                cid = decision['action']
                if isinstance(cid, int):
                    trip = belief._trip_by_cid[cid]
                    slot = belief._positions[cid, 'departure'][1]
                    exact = [i for i, event in enumerate(belief._trips[trip])
                             if (event.cid, event.kind) in belief._known_times]
                    left = max((i for i in exact if i <= slot), default=None)
                    summaries.append({'date': day, 'query': qi, 'decision': number, 'cid': cid,
                                      'trip_slots': len(belief._trips[trip]), 'departure_slot': slot,
                                      'left_exact_separator': left,
                                      'same_current_trip': decision['current_trip'] == trip})
                    identity = {'date': day, 'query': qi, 'decision': number, 'clock': decision['clock_seconds']}
                    if not selected:
                        selected['cold'] = (belief.copy(), parameters, cid, identity)
                    if left is not None and slot > left:
                        if ('observed' not in selected or left > selected['observed'][4]):
                            selected['observed'] = (belief.copy(), parameters, cid, identity, left)
                last_arrival = apply_receipt(belief, graph, metadata, decision, last_arrival)
    cases = [inspect_case(name, *record[:4]) for name, record in selected.items()]
    output = {'scope': 'All53 archived rollout states, executed-action slots only; two small within-trip posterior checks, no city policy rerun.',
              'selection': 'First cold action and the executed future action having the furthest already known exact same-trip prefix; selected to test boundary correctness, not estimate average savings.',
              'data': str(source.relative_to(ROOT)), 'archive': str(archive_path.relative_to(ROOT)),
              'train_dates': data['train_dates'], 'histories': summaries,
              'static_schedule_monotonicity': monotonicity,
              'actual_trip_checks': cases, 'archived_fullquery_evidence': profiles,
              'source_texts': {name: (ROOT / name).read_text() for name in
                               ['experiments/reconstruction_v5/chain_boundary_reference.py',
                                'experiments/reconstruction_v5/run_chain_boundaries.py',
                                'src/evolving_trip_belief.py']}}
    path = Path(__file__).with_name('chain_boundaries_results.json')
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'executed_actions': len(summaries), 'cases': cases}), flush=True)


if __name__ == '__main__':
    main()
