"""Cheap controls paired with V26's unchanged worlds and archived outcomes."""

from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import numpy as np

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from, ranks_from, selected_worlds
from experiments.reconstruction_v25.run import execute, public_offers
from experiments.reconstruction_v26.run import replay_world, LAWS, N_WORLDS
from experiments.reconstruction_v27.evaluate import integrate
from src.empirical_duration_belief import EmpiricalDurationBelief
from src.evolving_trip_belief import EvolvingTripBelief
from src.full_menu_opportunity import FullMenuOpportunity
from src.plan_waiting_controls import SchedulePriority, SingleTargetOpportunity, CombinedWaitingOpportunity
from src.recourse_itinerary_router import trip_key
from src.transfer_opportunity import TransferOpportunityScores


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'
PREVIOUS = HERE.parent/'reconstruction_v26'
NEW = ('schedule_priority', 'single_target_nominal', 'combined_wait_nominal')
ALL = ('full_first_offer', 'one_step_nominal', 'multistep_nominal', *NEW)
FIELDS = ('action', 'tail', 'loss', 'arrived', 'status')


def assert_reproduction(current, old):
    if any(current[field] != old[field] for field in FIELDS):
        raise AssertionError(('Reconstructed V26 first-offer outcome changed', current, old))


def paired_summary(rows):
    complete = [row for row in rows if all(row['outcomes'][m]['loss'] is not None for m in ALL)]
    comparisons = {}
    for method in ('full_first_offer', 'one_step_nominal', *NEW):
        differences = [r['outcomes'][method]['loss']-r['outcomes']['multistep_nominal']['loss'] for r in complete]
        comparisons['dp_vs_'+method] = dict(pairs=len(complete), mean_gain=mean(differences) if complete else None,
            difference_sd=stdev(differences) if len(complete)>1 else None,
            wins=sum(x > 0 for x in differences), harms=sum(x < 0 for x in differences),
            changed=sum(r['outcomes'][method]['action'] != r['outcomes']['multistep_nominal']['action'] for r in complete))
    return dict(rows=len(rows), common_complete=len(complete),
        mean_loss={m: mean(r['outcomes'][m]['loss'] for r in complete) if complete else None for m in ALL},
        arrivals={m: sum(r['outcomes'][m]['arrived'] for r in complete) for m in ALL},
        statuses={m: dict(Counter(r['outcomes'][m]['status'] for r in rows)) for m in ALL},
        comparisons=comparisons)


def run_root(graph, metadata, fit, root, law, law_index, index, old):
    tick = perf_counter()
    priority = SchedulePriority({o['first_cid']: o['scheduled_arrival'] for o in root['full_options']})
    priority_compile = perf_counter()-tick
    tick = perf_counter()
    options, ranks = options_from(root['full_options']), ranks_from(graph, root['full_options'])
    params = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    kwargs = dict(public_occurrences=False, grid_step_seconds=20, **fit['increments'])
    belief = (EvolvingTripBelief(graph, **kwargs) if law == 'original20' else
              EmpiricalDurationBelief(graph, duration_fit=fit['duration_fit'], **kwargs))
    compiler = TransferOpportunityScores(graph, belief, params, root['deadline'], options)
    setup = perf_counter()-tick
    tick = perf_counter()
    reference = FullMenuOpportunity(compiler, ranks, root['start'], root['start']+30, root['deadline'])
    nominal_compile = perf_counter()-tick
    tick = perf_counter()
    single = SingleTargetOpportunity(reference)
    single_compile = perf_counter()-tick
    tick = perf_counter()
    combined = CombinedWaitingOpportunity(single)
    combined_compile = perf_counter()-tick
    controls = dict(zip(NEW, (priority, single, combined)))
    offers = public_offers(metadata, options, root['start'], root['start']+30, root['deadline'])
    actual_new = {m: execute(graph, metadata, root, options, ranks, offers, control) for m, control in controls.items()}
    first = execute(graph, metadata, root, options, ranks, offers)
    assert_reproduction(first, old['actual']['outcomes']['full_first_offer'])
    actual = dict(outcomes={**old['actual']['outcomes'], **actual_new})
    tick = perf_counter()
    with np.load(PREVIOUS/f'{law}_{index}_values.npz') as archive:
        proxy, tables = integrate(reference, controls, archive)
    proxy['expected_loss'] = {m: t-root['start'] for m, t in proxy['expected_arrival'].items()}
    integration = perf_counter()-tick
    np.savez_compressed(HERE/f'{law}_{index}_values.npz', **tables)
    trips = {trip_key(graph.connections[cid]) for option in options.values()
             for path in (option.first_path, *option.tails.values()) for cid in path}
    tick = perf_counter()
    worlds = selected_worlds(belief, params, trips, N_WORLDS, [20261112, 26, law_index, index])
    rebuild = perf_counter()-tick
    tick = perf_counter()
    new_samples, combined_samples = [], []
    for events, prior in zip(worlds, old['samples'], strict=True):
        result = replay_world(graph, events, root, options, ranks, dict(full_first_offer=None, **controls))
        assert_reproduction(result['outcomes'].pop('full_first_offer'), prior['outcomes']['full_first_offer'])
        new_samples.append(result)
        combined_samples.append(dict(outcomes={**prior['outcomes'], **result['outcomes']}))
    replay = perf_counter()-tick
    return dict(root_id=root['root_id'], law=law, old_results=str(PREVIOUS/'results.json.gz'),
        actual_new=actual_new, actual_summary=paired_summary([actual]),
        proxy=proxy, new_samples=new_samples, sampled=paired_summary(combined_samples),
        reproduced_v26_worlds=len(worlds),
        wall=dict(priority_compile=priority_compile, setup=setup, nominal_compile=nominal_compile,
            single_compile=single_compile, combined_extra_compile=combined_compile,
            exact_evaluation=integration, old_world_reconstruction=rebuild, new_policy_replay=replay)), actual, combined_samples


def main():
    started = perf_counter()
    roots = json.loads((PREVIOUS/'selection.json').read_text())['roots']
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    graph, _, _, _, metadata, _ = dated_graph(data, '2023-07-16', 'gtfs_station_max', True)
    with gzip.open(PREVIOUS/'results.json.gz', 'rt') as handle:
        previous = json.load(handle)
    if not previous['complete'] or len(roots) != 6 or any(len(r['full_options']) != 5 for r in roots):
        raise AssertionError('V26 six-root panel must be complete and unchanged')
    old_rows = {(r['law'], r['root_id']): r for r in previous['results']}
    shared = perf_counter()-started
    results, report = [], dict(shared_preparation_wall=shared, by_law={})
    for law_index, law in enumerate(LAWS):
        actuals, samples = [], []
        for index, root in enumerate(roots):
            old = old_rows[law, root['root_id']]
            if len(old['samples']) != N_WORLDS:
                raise AssertionError('Original V26 world budget changed')
            row, actual, paired = run_root(graph, metadata, fit, root, law, law_index, index, old)
            results.append(row)
            actuals.append(actual)
            samples.extend(paired)
            print(law, root['root_id'], row['proxy']['expected_loss'], row['sampled']['mean_loss'], flush=True)
            with gzip.open(HERE/'results.json.gz', 'wt') as handle:
                json.dump(dict(complete=False, results=results), handle, separators=(',', ':'))
        selected = [r for r in results if r['law'] == law]
        sampled = paired_summary(samples)
        for name, comparison in sampled['comparisons'].items():
            comparison['stratified_mc_standard_error'] = sum(
                r['sampled']['comparisons'][name]['difference_sd']**2/N_WORLDS for r in selected)**.5/len(selected)
        report['by_law'][law] = dict(actual=paired_summary(actuals), sampled=sampled,
            proxy_expected_loss={**previous['summary']['by_law'][law]['proxy_expected_loss'],
                **{m: mean(r['proxy']['expected_loss'][m] for r in selected) for m in NEW}},
            wall={k: sum(r['wall'][k] for r in selected) for k in selected[0]['wall']},
            actual_online_wall={m: sum(r['actual_new'][m]['online_wall'] for r in selected) for m in NEW},
            archived_v26_wall=previous['summary']['by_law'][law]['wall'],
            archived_v26_actual_online_wall=previous['summary']['by_law'][law]['actual_online_wall'])
    report.update(total_wall=perf_counter()-started, reused_worlds=sum(r['reproduced_v26_worlds'] for r in results), new_worlds=0)
    with gzip.open(HERE/'results.json.gz', 'wt') as handle:
        json.dump(dict(complete=True, results=results, summary=report), handle, separators=(',', ':'))
    (HERE/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
