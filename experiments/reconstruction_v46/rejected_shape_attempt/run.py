"""Bounded full-pi reference on the four preregistered no-cross critical states."""
import argparse
import csv
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
from experiments.reconstruction_v46.oracle import FullPolicyOracle

HERE = Path(__file__).resolve().parent


def run():
    tick, cpu = perf_counter(), process_time()
    contract = read(HERE/'target_contract.json')
    manifest = read(HERE.parent/'reconstruction_v38/seed_manifest.json')
    data = read(HERE.parent/'reconstruction_v34/data.json.gz')
    fit = read(HERE.parent/'reconstruction_v34/model_fit.json')
    emission = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    save(HERE/'preparation.json', dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick,
        scope='One common resident load, outside each query cost'))
    for sequence, index in enumerate(contract['first_natural_state_indices']):
        entry = manifest[index]; state = entry['state']
        tick, cpu = perf_counter(), process_time()
        graph = context(data, state['day'])[0]
        preparation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
        for label in ordered(contract['natural_configurations'], sequence):
            path = HERE/'natural'/f'{index:03}_{label}.json.gz'
            if path.exists():
                continue
            tick, cpu = perf_counter(), process_time()
            model, parameters = setup(graph, fit)
            node = restore_node(state['node'])
            history = restored('shallow', model, parameters, emission, state)
            planner = MatchedContinuationPlanner(graph, model, parameters, state['deadline'])
            base, receipt = score(engine(planner, 'shallow'), node, state['cid'], state['clock'], history)
            row = {key: state[key] for key in ('kind', 'cid', 'clock')}
            if label == 'full_policy_oracle':
                oracle = FullPolicyOracle(graph, model, parameters, state['deadline'], history,
                    max_compiles=2000, max_process_cpu=120.)
                result = oracle.forced(node, row)
            elif label == 'markov128':
                result = evaluate(graph, model, parameters, state['deadline'], node, row, history,
                    mode='markov128', seed=entry['seed'], count=128)
            else:
                result = checked_evaluation(graph, model, parameters, node, history, state,
                    'shallow', 128, entry['seed'])
            result.update(index=index, root_index=state['root_index'], label=label,
                clock=state['clock'], deadline=state['deadline'], source=entry['source'],
                public_history=history.public_state(), base_action=base, base_receipt=receipt,
                preparation=preparation,
                complete_cost=dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick))
            if label != 'full_policy_oracle':
                result['selected_action'] = choose(result, base)
            save(path, result)
            if label == 'full128':
                old = read(HERE.parent/'reconstruction_v41/banks'/f'{index:03}_shallow.json.gz')
                if normalized(science(result)) != normalized(science(old)):
                    raise ValueError(f'Original full-policy sampled results changed: {index}')
            elif label == 'markov128':
                old = read(HERE.parent/'reconstruction_v45/banks'/f'{index:03}_markov128.json.gz')
                same_bank(normalized(result), old)
                if result['selected_action'] != old['selected_action']:
                    raise ValueError(f'Original Markov sample decision changed: {index}')
            print(index, label, 'CPU', round(result['complete_cost']['process_cpu'], 5),
                'actions', result.get('actions', result.get('selected_action')), flush=True)


def summarize():
    contract = read(HERE/'target_contract.json')
    rows = []
    for index in contract['first_natural_state_indices']:
        results = {label: read(HERE/'natural'/f'{index:03}_{label}.json.gz')
                   for label in contract['natural_configurations']}
        oracle = results['full_policy_oracle']
        actions = oracle['actions']
        low = actions['alight']['lower']-actions['continue']['upper']
        high = actions['alight']['upper']-actions['continue']['lower']
        exact = all(v['mean'] is not None and v['unresolved_probability'] == 0 for v in actions.values())
        certified = 'alight' if high < -1e-9 else 'continue' if low > 1e-9 else None
        rows.append(dict(index=index, root_index=oracle['root_index'], exact=exact,
            actions_absolute=actions, difference_interval=[low, high], certified_action=certified,
            oracle=oracle, costs={label: r['complete_cost'] for label, r in results.items()},
            sampled_references={label: dict(selected_action=r['selected_action'], paired=r['paired'],
                scope='Original128 finite-bank estimate, not full-policy population truth')
                for label, r in results.items() if label != 'full_policy_oracle'}))
    completed = sum(r['exact'] for r in rows)
    result = dict(stage='V46 first no-cross reference stage', first_states=len(rows),
        exact_states=completed, action_certifications=sum(r['certified_action'] is not None for r in rows),
        records=rows, sampling_comparators_reproduce_archived_science=True,
        later_shared_states=contract['later_shared_state_indices'],
        later_shared_values_evaluated=False, all_panel_structure_checked=True,
        all_panel_new_values_evaluated=False, online_controller_launched=False,
        independent_method_contribution_established=False, g2_passed=False,
        stage_gate='reference_complete' if completed == len(rows) else 'reference_incomplete_within_declared_cap',
        scope='Incomplete exact-policy reference retains all unresolved probability; it is not an impossibility theorem. '
              'Fresh Markov/full128 times concern finite estimators and are not matched-certification competitors.')
    save(HERE/'oracle_checks.json', result)
    flat = []
    for r in rows:
        item = dict(index=r['index'], root_index=r['root_index'], exact=r['exact'],
            certified_action=r['certified_action'], gap_lower=r['difference_interval'][0], gap_upper=r['difference_interval'][1])
        for arm, value in r['actions_absolute'].items():
            item.update({arm+'_'+key: value[key] for key in ('lower','upper','mean','unresolved_probability')})
        for label, cost in r['costs'].items():
            item[label+'_cpu'], item[label+'_wall'] = cost['process_cpu'], cost['wall']
        flat.append(item)
    with (HERE/'natural.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print('exact states', completed, '/', len(rows), 'certifications', result['action_certifications'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('run','summarize'))
    args = parser.parse_args()
    (run if args.stage == 'run' else summarize)()
