"""Equal-context timing of exactly the same frozen two-policy families."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v5.run_sparse_certificates import context
from experiments.reconstruction_v6.itinerary_mc_compare import thresholds, entry_ready
from experiments.reconstruction_v6.itinerary_mc_evaluator import FixedWitness, execute_fixed_witness
from src.committed_transit import build_committed_model
from src.first_branch_bank import FirstBranchBank
from src.risk_bounds import CommitmentBounds


def main():
    input_path = HERE / 'itinerary_mc_results.json'
    source = json.loads(input_path.read_text())
    protocol_path = ROOT / source['frozen_protocol']
    with gzip.open(protocol_path, 'rt') as handle:
        protocol = json.load(handle)
    data, fit = protocol['data'], protocol['training']
    prepared = {day: dated_graph(data, day, 'gtfs_station_max', True)
                for day in {case['date'] for case in source['contexts']}}
    rows = []
    for ci, case in enumerate(source['contexts']):
        graph, _, _, _, metadata, _ = prepared[case['date']]
        policies = [FixedWitness(tuple(item['policy']['primary']), tuple(item['policy']['rescue']),
                                 item['policy']['first_action_now']) for item in case['policies']]
        for method, n in [('analytic', 0), ('iid', 4), ('stratified', 4), ('iid', 8), ('stratified', 8)]:
            wall, cpu = perf_counter(), process_time()
            belief, state = context(graph, metadata, case['archived'], case['number'], fit)
            parameters = case['parameters']
            frozen = thresholds(belief, parameters, graph,
                                 [path for policy in policies for path in (policy.primary, policy.rescue)])
            for item in case['policies']:
                if any(frozen[int(cid)] != value for cid, value in item['departures'].items()):
                    raise ValueError('Benchmark thresholds differ from the original frozen policies')
            ready_wall, ready_cpu = perf_counter(), process_time()
            values, transitions = [], 0
            if method == 'analytic':
                index = CommitmentBounds(belief, parameters, graph, case['destination'], case['deadline'])
                values = [index.with_rescue(state, policy.primary, policy.rescue).upper for policy in policies]
                np.testing.assert_allclose(values, [item['value'] for item in case['analytic']], atol=1e-8, rtol=0.)
            else:
                rng = np.random.default_rng(np.random.SeedSequence([60089, ci]))
                bank = FirstBranchBank(belief, parameters, rng, n)
                for policy in policies:
                    ready = entry_ready(state, policy.primary[0], policy.first_action_now, frozen, graph, case['deadline'])
                    worlds = ([(1 / n, scene) for scene in bank.original] if method == 'iid' else
                              bank.for_action(policy.primary[0], ready, case['deadline']))
                    model = build_committed_model(graph, worlds, state.stop, case['destination'], state.time,
                                                  case['deadline'], lookahead=1500, initial_state=state)
                    evaluated = [execute_fixed_witness(model, world, graph, policy, frozen, case['deadline'])
                                 for world in model.scenarios]
                    transitions += sum(value['transitions'] for value in evaluated)
                    values.append(sum(p * value['cost'] for (p, _), value in zip(worlds, evaluated)))
            end_wall, end_cpu = perf_counter(), process_time()
            rows.append({'context': ci, 'date': case['date'], 'number': case['number'],
                         'method': method, 'n': n, 'values': values, 'transitions': transitions,
                         'public_context_threshold_wall_seconds': ready_wall - wall,
                         'public_context_threshold_cpu_seconds': ready_cpu - cpu,
                         'evaluation_wall_seconds': end_wall - ready_wall,
                         'evaluation_cpu_seconds': end_cpu - ready_cpu,
                         'total_wall_seconds': end_wall - wall, 'total_cpu_seconds': end_cpu - cpu})
    files = ['src/evolving_trip_belief.py', 'src/committed_transit.py', 'src/risk_bounds.py',
             'src/first_branch_bank.py', 'src/first_branch_belief.py',
             'experiments/reconstruction_v6/itinerary_mc_compare.py',
             'experiments/reconstruction_v6/itinerary_mc_evaluator.py', str(Path(__file__).relative_to(ROOT))]
    result = {'frozen_data_policy_protocol': str(protocol_path.relative_to(ROOT)),
              'source_text': {name: (ROOT / name).read_text() for name in files},
              'scope': 'One fresh public context per method, both fixed policies share thresholds and inference cache. '
                       'Path discovery and graph/input construction are common outside this evaluation-only comparison. '
                       'Fifteen one-shot timing observations are not a portable performance guarantee.',
              'rows': rows}
    (HERE / 'itinerary_mc_timing_results.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(row['context'], row['method'], row['n'], round(row['total_wall_seconds'], 5),
              round(row['evaluation_wall_seconds'], 5), row['transitions'])


if __name__ == '__main__':
    main()
