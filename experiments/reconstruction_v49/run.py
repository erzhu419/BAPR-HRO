"""Frozen seven-state paired costs under unchanged V48 budgets and scheduler."""
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import ordered, restore_node, save
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.run import checked_evaluation, engine, restored, score
from experiments.reconstruction_v44.matched import choose
from experiments.reconstruction_v45.prefix import evaluate
from experiments.reconstruction_v48.lazy_frontier import LazyFrontierOracle

HERE = Path(__file__).resolve().parent


def run():
    from experiments.reconstruction_v49.bulk import IncrementalBulkOracle
    from experiments.reconstruction_v49.merging import OrdinaryMergedOracle
    contract = read(HERE/'contract.json')
    tick, cpu = perf_counter(), process_time()
    manifest = read(HERE.parent/'reconstruction_v38/seed_manifest.json')
    data = read(HERE.parent/'reconstruction_v34/data.json.gz')
    fit = read(HERE.parent/'reconstruction_v34/model_fit.json')
    emission = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    prep = HERE/'preparation.json'
    if not prep.exists():
        save(prep, dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick,
            scope='Common resident archive load outside query cost'))
    limits = contract['limits']
    for sequence, index in enumerate(contract['state_indices']):
        entry = manifest[index]
        state = entry['state']
        tick, cpu = perf_counter(), process_time()
        graph = context(data, state['day'])[0]
        preparation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        archived = read(HERE.parent/'reconstruction_v45/banks'/f'{index:03}_markov128.json.gz')
        for label in ordered(contract['configurations'], sequence):
            path = HERE/'natural'/f'{index:03}_{label}.json.gz'
            if path.exists():
                continue
            tick, cpu = perf_counter(), process_time()
            model, parameters = setup(graph, fit)
            node = restore_node(state['node'])
            history = restored('shallow', model, parameters, emission, state)
            planner = MatchedContinuationPlanner(graph, model, parameters, state['deadline'])
            base, base_receipt = score(engine(planner, 'shallow'), node, state['cid'], state['clock'], history)
            row = {key:state[key] for key in ('kind', 'cid', 'clock')}
            prefix_cost = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
            if label == 'markov128':
                result = evaluate(graph, model, parameters, state['deadline'], node, row, history,
                    mode='markov128', seed=entry['seed'], count=128)
            elif label == 'full128':
                result = checked_evaluation(graph, model, parameters, node, history, state,
                    'shallow', 128, entry['seed'])
            else:
                cls = {'v48':LazyFrontierOracle, 'shared':IncrementalBulkOracle,
                       'ordinary':OrdinaryMergedOracle}[label.split('_')[0]]
                oracle = cls(graph, model, parameters, state['deadline'], history,
                    max_compiles=limits['charged_compiles'],
                    max_process_cpu=limits['process_cpu_seconds'])
                result = oracle.forced(node, row, schedule='pooled',
                    priority_arms=('alight', 'continue'), stop_on_certificate=label.endswith('_stop'),
                    certificate_every=250, checkpoints=tuple(contract['checkpoint_charges']))
                if hasattr(oracle, 'relevance_profiles'):
                    result['relevance_profiles'] = list(oracle.relevance_profiles.values())
            result.update(index=index, root_index=state['root_index'], label=label,
                clock=state['clock'], deadline=state['deadline'], source=entry['source'],
                public_history=history.public_state(), base_action=base, base_receipt=base_receipt,
                preparation=preparation, prefix_cost=prefix_cost)
            if label in ('markov128', 'full128'):
                result['selected_action'] = choose(result, base)
            else:
                result['v45_fallback_action'] = archived['selected_action']
                result['selected_action'] = result['certified_action'] or result['v45_fallback_action']
                result['fallback_used'] = result['certified_action'] is None
                result['fallback_scope'] = 'Archived action attached after evaluation, never supplied to solver; no online deployment claim'
            result['complete_cost'] = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
            save(path, result)
            print(index, label, 'CPU', round(result['complete_cost']['process_cpu'], 6),
                'count', result.get('counters', {}).get('charged_compiles'),
                'gap', result.get('difference_interval'), 'cert', result.get('certified_action'),
                'stop', result.get('stop_reason'), flush=True)


if __name__ == '__main__':
    run()
