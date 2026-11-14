"""Single bounded, rotated V47 comparison; existing raw receipts are retained."""
import argparse
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import ordered, restore_node, save
from experiments.reconstruction_v41.continuation import science
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.run import checked_evaluation, engine, normalized, restored, score
from experiments.reconstruction_v44.matched import choose
from experiments.reconstruction_v45.prefix import evaluate
from experiments.reconstruction_v45.summarize import same_bank
from experiments.reconstruction_v46.shared_oracle import SharedObservationOracle

HERE = Path(__file__).resolve().parent


def run(hybrid=False, bulk=False):
    from experiments.reconstruction_v47.incremental import IncrementalOracle
    contract = read(HERE/'contract.json')
    if bulk:
        from experiments.reconstruction_v47.bulk import BulkIncrementalOracle
        extension = read(HERE/'bulk_amendment.json')
        indices, labels = extension['state_indices'], extension['configurations']
    elif hybrid:
        from experiments.reconstruction_v47.hybrid import HybridIncrementalOracle
        extension = read(HERE/'hybrid_amendment.json')
        indices, labels = extension['state_indices'], extension['configurations']
    else:
        indices, labels = contract['state_indices'], contract['configurations']
    tick, cpu = perf_counter(), process_time()
    manifest = read(HERE.parent/'reconstruction_v38/seed_manifest.json')
    data = read(HERE.parent/'reconstruction_v34/data.json.gz')
    fit = read(HERE.parent/'reconstruction_v34/model_fit.json')
    emission = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    preparation_name = 'bulk_preparation.json' if bulk else 'hybrid_preparation.json' if hybrid else 'preparation.json'
    save(HERE/preparation_name, dict(process_cpu=process_time()-cpu,
        wall=perf_counter()-tick, scope='Common resident archive load outside query cost'))
    for sequence, index in enumerate(indices):
        entry = manifest[index]
        state = entry['state']
        tick, cpu = perf_counter(), process_time()
        graph = context(data, state['day'])[0]
        preparation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        for label in ordered(labels, sequence):
            path = HERE/'natural'/f'{index:03}_{label}.json.gz'
            if path.exists():
                continue
            tick, cpu = perf_counter(), process_time()
            model, parameters = setup(graph, fit)
            node = restore_node(state['node'])
            history = restored('shallow', model, parameters, emission, state)
            planner = MatchedContinuationPlanner(graph, model, parameters, state['deadline'])
            base, base_receipt = score(engine(planner, 'shallow'), node, state['cid'], state['clock'], history)
            row = {key: state[key] for key in ('kind', 'cid', 'clock')}
            prefix_cost = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
            if label == 'legacy46':
                oracle = SharedObservationOracle(graph, model, parameters, state['deadline'], history,
                    max_compiles=contract['limits']['charged_compiles'],
                    max_process_cpu=contract['limits']['process_cpu_seconds'])
                result = oracle.forced(node, row)
            elif label.startswith(('incremental_', 'hybrid_', 'bulk_')):
                oracle_type = BulkIncrementalOracle if bulk else HybridIncrementalOracle if hybrid else IncrementalOracle
                oracle = oracle_type(graph, model, parameters, state['deadline'], history,
                    max_compiles=contract['limits']['charged_compiles'],
                    max_process_cpu=contract['limits']['process_cpu_seconds'])
                result = oracle.forced(node, row,
                    schedule='pooled' if label.endswith('_recycle') else 'fixed_split',
                    checkpoints=tuple(contract['checkpoints']))
            elif label == 'markov128':
                result = evaluate(graph, model, parameters, state['deadline'], node, row, history,
                    mode='markov128', seed=entry['seed'], count=128)
            else:
                result = checked_evaluation(graph, model, parameters, node, history, state,
                    'shallow', 128, entry['seed'])
            result.update(index=index, root_index=state['root_index'], label=label,
                clock=state['clock'], deadline=state['deadline'], source=entry['source'],
                public_history=history.public_state(), base_action=base,
                base_receipt=base_receipt, preparation=preparation, prefix_cost=prefix_cost)
            if label in ('markov128', 'full128'):
                result['selected_action'] = choose(result, base)
            # Saving/validating does not enter the decision query's CPU cost.
            result['complete_cost'] = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
            save(path, result)
            if label == 'markov128':
                old = read(HERE.parent/'reconstruction_v45/banks'/f'{index:03}_markov128.json.gz')
                same_bank(normalized(result), old)
                assert result['selected_action'] == old['selected_action']
            elif label == 'full128':
                old = read(HERE.parent/'reconstruction_v41/banks'/f'{index:03}_shallow.json.gz')
                assert normalized(science(result)) == normalized(science(old))
            elif label == 'legacy46':
                old = read(HERE.parent/'reconstruction_v46/natural'/f'{index:03}_full_policy_oracle.json.gz')
                assert normalized(result['actions']) == normalized(old['actions'])
            print(index, label, 'CPU', round(result['complete_cost']['process_cpu'], 5),
                'gap', result.get('difference_interval'),
                'certified', result.get('certified_action'),
                'count', result.get('counters', {}).get('charged_compiles'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('--hybrid', action='store_true')
    stages.add_argument('--bulk', action='store_true')
    args = parser.parse_args()
    run(hybrid=args.hybrid, bulk=args.bulk)
