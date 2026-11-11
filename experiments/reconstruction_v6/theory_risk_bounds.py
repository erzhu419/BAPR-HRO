"""Fresh full-model commitment bounds and transfer-gate suffix certificates."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training, freeze_replay_inputs
from experiments.reconstruction_v5.run_sparse_certificates import context
from src.committed_transit import WAIT
from src.risk_bounds import CommitmentBounds


def actions(graph, state, deadline):
    if state.stop == -1 or state.time >= deadline:
        return ()
    candidates = []
    for c in graph.get_connections_from(state.stop):
        if (c.id in state.unavailable_ids or c.dep_time > state.time + 1500
                or (not c.pickup_allowed and c.trip_id != state.current_trip)
                or (c.dep_distribution is not None and c.dep_distribution.support_max < state.time)):
            continue
        candidates.append(('select', c.id))
    return (WAIT,) + tuple(candidates)


def onboard_suffix(belief, state, destination):
    if state.current_trip not in belief._trips or state.time != state.last_arrival:
        return ()
    slots = belief._trips[state.current_trip]
    reached = [index for index, slot in enumerate(slots) if slot.kind == 'arrival'
               and belief._known_times.get((slot.cid, slot.kind)) == state.time]
    if not reached:
        return ()
    first = max(reached) + 1
    stop, path = state.stop, []
    for index in range(first, len(slots), 2):
        c = belief.graph.connections[slots[index].cid]
        if c.dep_stop != stop or c.id in state.unavailable_ids:
            return ()
        path.append(c.id)
        stop = c.arr_stop
        if stop == destination:
            return tuple(path)
    return ()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'theory_risk_bounds_cold_results.json')
    args = parser.parse_args()
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    archive_path = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    archive = json.loads(archive_path.read_text())
    rows = [r for r in archive['rows'] if r['method'] == 'evolving_rollout']
    output = args.output.resolve()
    protocol = freeze_replay_inputs(output, data=data, forecasts={}, training=fit,
        settings={'archive_rows': rows, 'commitment_roots': [('2023-07-26', 0, 0),
                  ('2023-07-26', 0, 5), ('2023-07-27', 0, 0)], 'public_occurrences': False,
                  'clock': 'logical frozen current grids, no posterior worlds sampled',
                  'all_archived_states': True, 'lookahead_seconds': 1500,
                  'fresh_macro_context_after_commitment_measurement': True},
        sources=[str(source), str(archive_path), str(Path(__file__)),
                 'src/risk_bounds.py', 'experiments/reconstruction_v5/run_sparse_certificates.py',
                 'experiments/reconstruction_v5/run_chain_boundaries.py'],
        implementation='v6_analytic_commitment_and_transfer_gate_v1')
    prepared = {day: dated_graph(data, day, 'gtfs_station_max', True) for day in data['holdout_dates']}
    selected = {('2023-07-26', 0, 0), ('2023-07-26', 0, 5), ('2023-07-27', 0, 0)}
    commitments, macros = [], []
    for archived in rows:
        graph, _, stops, _, metadata, _ = prepared[archived['date']]
        destination = stops[archived['od']['destination']]
        parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
        for number, decision in enumerate(archived['decisions']):
            belief, state = context(graph, metadata, archived, number, fit)
            identity = {'date': archived['date'], 'query': archived['query'], 'decision': number,
                        'clock': state.time, 'stop': state.stop, 'current_trip': state.current_trip}
            if (archived['date'], archived['query'], number) in selected:
                before, cpu_before = perf_counter(), process_time()
                index = CommitmentBounds(belief, parameters, graph, destination, archived['od']['deadline_seconds'])
                values = []
                for action in actions(graph, state, archived['od']['deadline_seconds']):
                    if action == WAIT:
                        values.append({'action': list(action), 'lower': min(60, archived['od']['deadline_seconds'] - state.time)})
                    else:
                        risk = index.commit(state, action[1])
                        values.append({'action': list(action), 'lower': risk.expected_first_cost,
                                       **asdict(risk)})
                upper = min((v['direct_stop_upper'] for v in values
                             if v.get('direct_stop_upper') is not None), default=None)
                commitments.append({**identity, 'actions': values, 'n_actions': len(values),
                    'direct_one_attempt_stop_upper': upper,
                    'full_model_actions_strictly_removed': sum(v['lower'] > upper for v in values) if upper is not None else None,
                    'minimum_commitment_lower': min(v['lower'] for v in values),
                    'wall_seconds': perf_counter() - before, 'cpu_seconds': process_time() - cpu_before,
                    'work': index.stats})
                # The commitment phase can populate shared pure-law memos in
                # a copied belief. Rebuild this public context before timing
                # the gate so its cold cost does not inherit that work.
                belief, state = context(graph, metadata, archived, number, fit)
            path = onboard_suffix(belief, state, destination)
            if not path:
                macros.append({**identity, 'candidate_path': [], 'certificate': None})
                continue
            before, cpu_before = perf_counter(), process_time()
            index = CommitmentBounds(belief, parameters, graph, destination, archived['od']['deadline_seconds'])
            value = index.transfer_gate(state, path)
            macros.append({**identity, 'candidate_path': path,
                           'certificate': asdict(value) if value is not None else None,
                           'wall_seconds': perf_counter() - before, 'cpu_seconds': process_time() - cpu_before,
                           'work': index.stats})
    valid = [r for r in macros if r['certificate'] is not None]
    result = {'scope': 'Full current independent-trip finite-grid probability model, not finite4world bank; historical city public roots only.',
              'frozen_protocol': str(protocol.relative_to(ROOT)), 'commitment_roots': commitments,
              'transfer_gate_states': macros,
              'summary': {'states': len(macros), 'candidate_suffixes': sum(bool(r['candidate_path']) for r in macros),
                          'certified_suffixes': len(valid),
                          'regret_bound_at_most_1s': sum(r['certificate']['regret_bound'] <= 1 for r in valid),
                          'regret_bound_at_most_5s': sum(r['certificate']['regret_bound'] <= 5 for r in valid),
                          'regret_bound_at_most_20s': sum(r['certificate']['regret_bound'] <= 20 for r in valid),
                          'total_gate_wall_seconds': sum(r.get('wall_seconds', 0) for r in macros)}}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'summary': result['summary'], 'commitments': [
        {k: r[k] for k in ('date', 'decision', 'n_actions', 'minimum_commitment_lower',
                           'direct_one_attempt_stop_upper', 'full_model_actions_strictly_removed', 'wall_seconds')}
        for r in commitments]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
