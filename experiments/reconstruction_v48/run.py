"""Frozen seven-state scheduling and actual early-stopping comparison."""
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
from experiments.reconstruction_v47.bulk import BulkIncrementalOracle
from experiments.reconstruction_v47.audit import matching_values

HERE = Path(__file__).resolve().parent


def run(lazy=False):
    from experiments.reconstruction_v48.frontier import FrontierOracle
    contract = read(HERE/'contract.json')
    if lazy:
        from experiments.reconstruction_v48.lazy_frontier import LazyFrontierOracle
        labels = read(HERE/'lazy_contract.json')['configurations']
    else:
        labels = contract['configurations']
    tick, cpu = perf_counter(), process_time()
    manifest = read(HERE.parent/'reconstruction_v38/seed_manifest.json')
    data = read(HERE.parent/'reconstruction_v34/data.json.gz')
    fit = read(HERE.parent/'reconstruction_v34/model_fit.json')
    emission = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    save(HERE/('lazy_preparation.json' if lazy else 'preparation.json'), dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick,
        scope='One common resident archive load outside query cost'))
    for sequence, index in enumerate(contract['state_indices']):
        entry = manifest[index]
        state = entry['state']
        tick, cpu = perf_counter(), process_time()
        graph = context(data, state['day'])[0]
        preparation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        archived_markov = read(HERE.parent/'reconstruction_v45/banks'/f'{index:03}_markov128.json.gz')
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
            if label == 'legacy47':
                oracle = BulkIncrementalOracle(graph, model, parameters, state['deadline'], history,
                    max_compiles=2000, max_process_cpu=120.)
                result = oracle.forced(node, row, schedule='pooled')
            elif label.startswith(('global_', 'lazy_')):
                oracle_type = LazyFrontierOracle if lazy else FrontierOracle
                oracle = oracle_type(graph, model, parameters, state['deadline'], history,
                    max_compiles=2000, max_process_cpu=120.)
                result = oracle.forced(node, row, schedule='pooled',
                    priority_arms=('alight',) if label == 'global_a_full' else ('alight', 'continue'),
                    stop_on_certificate=label.endswith('_stop'), certificate_every=250,
                    checkpoints=tuple(contract['checkpoint_charges']))
            elif label == 'markov128':
                result = evaluate(graph, model, parameters, state['deadline'], node, row, history,
                    mode='markov128', seed=entry['seed'], count=128)
            else:
                result = checked_evaluation(graph, model, parameters, node, history, state,
                    'shallow', 128, entry['seed'])
            result.update(index=index, root_index=state['root_index'], label=label,
                clock=state['clock'], deadline=state['deadline'], source=entry['source'],
                public_history=history.public_state(), base_action=base, base_receipt=base_receipt,
                preparation=preparation, prefix_cost=prefix_cost)
            if label in ('markov128', 'full128'):
                result['selected_action'] = choose(result, base)
            else:
                result['v45_fallback_action'] = archived_markov['selected_action']
                result['selected_action'] = result['certified_action'] or result['v45_fallback_action']
                result['fallback_used'] = result['certified_action'] is None
                result['fallback_scope'] = 'Frozen V45 action attached after evaluation, not fed to solver; no online deployment or uncharged fallback execution claim'
            result['complete_cost'] = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
            save(path, result)
            if label == 'markov128':
                same_bank(normalized(result), archived_markov)
                assert result['selected_action'] == archived_markov['selected_action']
            elif label == 'full128':
                old = read(HERE.parent/'reconstruction_v41/banks'/f'{index:03}_shallow.json.gz')
                assert normalized(science(result)) == normalized(science(old))
            elif label == 'legacy47':
                old = read(HERE.parent/'reconstruction_v47/natural'/f'{index:03}_bulk_recycle.json.gz')
                matching_values(result, old)
            print(index, label, 'CPU', round(result['complete_cost']['process_cpu'], 6),
                'count', result.get('counters', {}).get('charged_compiles'),
                'gap', result.get('difference_interval'), 'cert', result.get('certified_action'),
                'stop', result.get('stop_reason'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lazy', action='store_true')
    args = parser.parse_args()
    run(lazy=args.lazy)
