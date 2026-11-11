"""Fresh matched nominal-first anytime search with and without sparse bounds."""
from dataclasses import asdict
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
from time import perf_counter, process_time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import ROOT, dated_graph, fit_training, freeze_replay_inputs
from experiments.reconstruction_v5.run_sparse_certificates import setup, bank_work
from src.certified_policy_search import CertifiedPolicySearch
from src.committed_transit import WAIT
from src.continuation_router import nominal_action
from src.sparse_certificate_cutoff import SparseCertificateIndex


def evaluate(method, graph, metadata, actual_world, archived, number, fit, destination):
    before, cpu_before = perf_counter(), process_time()
    current = setup(graph, metadata, archived, number, fit, destination)
    state, model = current['state'], current['model']
    actions = tuple(model.actions(state))
    bound_ms, relaxation = 0., None
    def tail(public_state, belief, available):
        board = {a[1] for a in available if a[0] == 'select'}
        return (nominal_action(graph, public_state, destination, belief.mean_events(current['parameters']),
                               current['router'].menu, current['deadline'], 1500,
                               root_boardings=board, minute=60) if board else WAIT)
    def bounds(upper):
        nonlocal bound_ms, relaxation
        start = perf_counter()
        if method == 'zero_bound_anytime':
            result = dict.fromkeys(actions, Fraction(0))
        else:
            relaxation = SparseCertificateIndex(graph, current['bank'], destination,
                                                state.time, current['deadline'])
            result = relaxation.action_lower(state, actions, cutoff=upper)
        bound_ms += (perf_counter() - start) * 1000
        return result
    rng = current['post_sample_rng']
    rng_before = rng.bit_generator.state
    planner = CertifiedPolicySearch(model, current['belief'], tail, current['router']._advance_belief,
                                    rng, 512, current['deadline'] - state.time)
    planning_started = perf_counter()
    result = planner.plan_certified(bounds, current['deadline'] - state.time)
    plan_ms = (perf_counter() - planning_started) * 1000
    total_ms, cpu_ms = (perf_counter() - before) * 1000, (process_time() - cpu_before) * 1000
    selected = result.planning.action
    actual = {'scope': 'After-query physical first-action diagnostic only; actual endpoints were not used by either planner.'}
    if selected[0] == 'select':
        event = actual_world.events.get(selected[1])
        if event is None:
            actual['outcome_known'] = False
        else:
            diagnostic = SparseCertificateIndex(graph, [(1, actual_world)], destination, state.time, current['deadline'])
            stop, time = diagnostic.first_projection(0, state, selected)
            actual.update(outcome_known=True, departure=event.departure, arrival=event.arrival,
                          canceled=event.canceled, physical_next_stop=stop, physical_next_clock=time,
                          caught=(stop == graph.connections[selected[1]].arr_stop and time == event.arrival
                                  and not event.canceled))
    row = {'method': method, 'date': archived['date'], 'query': archived['query'], 'archived_decision': number,
           'root_clock': state.time, 'root_stop': state.stop, 'n_actions': len(actions),
           'worlds': 4, 'work_budget': 512, 'selected_action': list(selected),
           'upper': float(result.upper), 'lower': float(result.lower), 'gap': float(result.gap),
           'removed_actions': len(result.removed), 'transitions': result.planning.transitions,
           'completed_root_policies': result.planning.scored_root_actions,
           'completed_world_rollouts': result.planning.rollouts,
           'planning_wall_ms': plan_ms, 'bound_wall_ms': bound_ms,
           'fresh_total_wall_ms': total_ms, 'fresh_total_cpu_ms': cpu_ms,
           'context_router_bank_setup_ms': current['setup_ms'],
           'bound_work': dict(relaxation.stats) if relaxation else {}, 'bank_work': bank_work(current),
           'action_values': [{'action': list(action), 'L': float(result.action_lower[action]),
                              'U': float(result.action_upper[action]),
                              'completed_worlds': result.evaluated_worlds[action]}
                             for action in actions], 'physical_first_action_diagnostic': actual,
           'rng_after_sampling_equals_after_planning': rng_before == rng.bit_generator.state}
    return row, current['sample_bank'].uniforms, rng_before


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('certificate_attribution_results.json'))
    args = parser.parse_args()
    started = perf_counter()
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    archive_path = ROOT / 'experiments/reconstruction_v4/city_replay_results.json'
    archive = json.loads(archive_path.read_text())
    specs = [('2023-07-26', 0, 0), ('2023-07-26', 0, 5), ('2023-07-27', 0, 0)]
    prepared = {}
    for day in {day for day, _, _ in specs}:
        prepared[day] = dated_graph(data, day, 'gtfs_station_max', True)
    preparation_ms = (perf_counter() - started) * 1000
    output = args.output
    protocol = freeze_replay_inputs(
        output, data=data, forecasts={'dates': {day: {'frames': []} for day, _, _ in specs}}, training=fit,
        settings={'contexts': [list(spec) for spec in specs], 'samples': 4, 'transition_budget': 512,
                  'methods': ['zero_bound_anytime', 'sparse_cutoff_anytime'],
                  'public_history_records': [r for r in archive['rows'] if r['method'] == 'evolving_rollout'
                                            and (r['date'], r['query']) in {(day, q) for day, q, _ in specs}],
                  'transfer_rule': 'gtfs_station_max', 'charge_computation': False},
        implementation='fresh_same_bank_nominal_first_certificate_attribution_v1',
        sources=[str(source), str(archive_path), 'src/certified_policy_search.py', 'src/sparse_certificate.py',
                 'src/sparse_certificate_cutoff.py', 'src/evolving_trip_belief.py',
                 'experiments/reconstruction_v5/run_sparse_certificates.py',
                 'experiments/reconstruction_v5/run_certificate_attribution.py'])
    rows, pairs = [], []
    for day, qi, number in specs:
        graph, actual, stops, _, metadata, _ = prepared[day]
        archived = next(r for r in archive['rows'] if r['date'] == day and r['query'] == qi
                        and r['method'] == 'evolving_rollout')
        previous_tapes = previous_rng = None
        for method in ('zero_bound_anytime', 'sparse_cutoff_anytime'):
            row, tapes, rng = evaluate(method, graph, metadata, actual, archived, number, fit,
                                        stops[archived['od']['destination']])
            rows.append(row)
            if previous_tapes is not None:
                pair = {'date': day, 'query': qi, 'archived_decision': number,
                        'same_trip_random_tapes': (tuple(tapes) == tuple(previous_tapes)
                                                  and all(np.array_equal(tapes[t], previous_tapes[t]) for t in tapes)),
                        'same_rng_state_after_sampling': rng == previous_rng}
                assert pair['same_trip_random_tapes'] and pair['same_rng_state_after_sampling']
                pairs.append(pair)
            previous_tapes, previous_rng = tapes, rng
            output.write_text(json.dumps({'scope': 'Three archived full-city public roots, fresh model/inference caches per method; common4world tapes, same causal nominal tail, same512 physical transition budget. No actual-day policy run.',
                                          'frozen_protocol': str(protocol.relative_to(ROOT)),
                                          'shared_input_fit_graph_preparation_ms': preparation_ms,
                                          'timing_scope': 'Fresh totals include public-context conditioning, router/bank setup and all planning/bounds. Input load, training fit, graph construction are shared and separately reported. CPU cost is not charged to the passenger event clock.',
                                          'pairs': pairs, 'rows': rows}, indent=2) + '\n')
            print(json.dumps({key: row[key] for key in ['date', 'archived_decision', 'method', 'selected_action',
                             'upper', 'lower', 'gap', 'transitions', 'completed_root_policies',
                             'fresh_total_wall_ms', 'fresh_total_cpu_ms', 'bound_wall_ms']}), flush=True)


if __name__ == '__main__':
    main()
