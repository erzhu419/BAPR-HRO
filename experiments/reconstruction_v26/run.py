"""Six fixed five-offer roots: complete waiting DP, proxy and physical pairs."""

from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import numpy as np

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from, ranks_from, route_replay, selected_worlds
from experiments.reconstruction_v25.run import execute, public_offers
from experiments.reconstruction_v26.evaluate import METHODS, integrate
from src.empirical_duration_belief import EmpiricalDurationBelief
from src.evolving_trip_belief import EvolvingTripBelief
from src.full_menu_opportunity import FullMenuOpportunity
from src.full_menu_waiting_dp import FullMenuWaitingDP
from src.recourse_itinerary_router import trip_key
from src.transfer_opportunity import TransferOpportunityScores


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'
LAWS = ('original20', 'route_empirical20')
N_WORLDS = 128


def replay_world(graph, events, root, options, ranks, controls):
    entry, ready, deadline = root['start'], root['start']+30, root['deadline']
    clocks = sorted({events[cid].departure for cid in options
                     if not events[cid].canceled and ready <= events[cid].departure < deadline})
    outcomes = {}
    for method, controller in controls.items():
        selected, arrival, status, tail, trace = None, deadline, 'no_offer', None, []
        for clock in clocks:
            visible = tuple(cid for cid in options if not events[cid].canceled and events[cid].departure == clock)
            seen = tuple(cid for cid in options if not events[cid].canceled and entry <= events[cid].departure <= clock)
            if controller is None:
                chosen, receipt = min(visible, key=ranks.__getitem__), {}
            else:
                chosen, receipt = controller.choose(visible, seen, clock)
            trace.append(dict(clock=clock, visible=visible, seen=seen, choice=chosen, receipt=receipt))
            if chosen is not None:
                selected = chosen
                arrival, status, tail = route_replay(graph, events, options[chosen], clock, deadline)
                break
        if trace and selected is None:
            status = 'declined_all_offers'
        outcomes[method] = dict(action=selected, tail=tail, loss=arrival-root['start'],
            arrived=status == 'arrived', status=status, trace=trace)
    return dict(outcomes=outcomes)


def summary(rows):
    paired = [r for r in rows if all(r['outcomes'][m]['loss'] is not None for m in METHODS)]
    comparisons = {}
    for baseline in METHODS[:-1]:
        differences = [r['outcomes'][baseline]['loss']-r['outcomes'][METHODS[-1]]['loss'] for r in paired]
        comparisons['multistep_vs_'+baseline] = dict(pairs=len(paired),
            mean_gain=mean(differences) if paired else None,
            paired_difference_sd=stdev(differences) if len(paired)>1 else None,
            wins=sum(x > 0 for x in differences), harms=sum(x < 0 for x in differences),
            changed=sum(r['outcomes'][baseline]['action'] != r['outcomes'][METHODS[-1]]['action'] for r in paired))
    return dict(rows=len(rows), complete_pairs=len(paired),
        mean_loss={m: mean(r['outcomes'][m]['loss'] for r in paired) if paired else None for m in METHODS},
        arrivals={m: sum(r['outcomes'][m]['arrived'] for r in paired) for m in METHODS},
        statuses={m: dict(Counter(r['outcomes'][m]['status'] for r in rows)) for m in METHODS},
        comparisons=comparisons)


def run_root(graph, metadata, fit, root, law, law_index, index, original_result):
    tick = perf_counter()
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    kwargs = dict(public_occurrences=False, grid_step_seconds=20, **fit['increments'])
    belief = (EvolvingTripBelief(graph, **kwargs) if law == 'original20' else
              EmpiricalDurationBelief(graph, duration_fit=fit['duration_fit'], **kwargs))
    options, ranks = options_from(root['full_options']), ranks_from(graph, root['full_options'])
    compiler = TransferOpportunityScores(graph, belief, parameters, root['deadline'], options)
    setup = perf_counter()-tick
    tick = perf_counter()
    one = FullMenuOpportunity(compiler, ranks, root['start'], root['start']+30, root['deadline'])
    nominal_compile = perf_counter()-tick
    tick = perf_counter()
    dp = FullMenuWaitingDP(one)
    dp_compile = perf_counter()-tick
    controls = dict(zip(METHODS, (None, one, dp)))
    offers = public_offers(metadata, options, root['start'], root['start']+30, root['deadline'])
    actual = dict(outcomes={m: execute(graph, metadata, root, options, ranks, offers, control)
                           for m, control in controls.items()})
    previous = 'original_full_nominal' if law == 'original20' else 'empirical_full_nominal'
    for current, old in (('full_first_offer', 'full_first_offer'), ('one_step_nominal', previous)):
        for field in ('action', 'tail', 'loss', 'arrived', 'status'):
            # A failure detects an unintended change in V25's inputs or replay.
            if actual['outcomes'][current][field] != original_result['outcomes'][old][field]:
                raise AssertionError((root['root_id'], law, current, field, 'V25 control changed'))
    public_states = []
    for public in offers or ():
        old_choice, old_receipt = one.choose(public['visible'], public['seen'], public['clock'])
        new_choice, new_receipt = dp.choose(public['visible'], public['seen'], public['clock'])
        public_states.append(dict(**public, one_step_choice=old_choice, multistep_choice=new_choice,
            one_step_receipt=old_receipt, multistep_receipt=new_receipt))
    tick = perf_counter()
    exact, tables = integrate(one, dp)
    exact['expected_loss'] = {m: v-root['start'] for m, v in exact['expected_arrival'].items()}
    integration = perf_counter()-tick
    np.savez_compressed(HERE/f'{law}_{index}_values.npz', dp=dp.values, **tables)
    trips = {trip_key(graph.connections[cid]) for option in options.values()
             for path in (option.first_path, *option.tails.values()) for cid in path}
    tick = perf_counter()
    worlds = selected_worlds(belief, parameters, trips, N_WORLDS, [20261112, 26, law_index, index])
    sampling = perf_counter()-tick
    tick = perf_counter()
    samples = [replay_world(graph, world, root, options, ranks, controls) for world in worlds]
    sample_replay = perf_counter()-tick
    return dict(root_id=root['root_id'], law=law, menu_size=len(options), model=exact,
        actual=actual, public_states=public_states, sampled=summary(samples), samples=samples,
        dp_counters=dict(dp.counters), wall=dict(setup=setup, nominal_compile=nominal_compile,
            dp_compile_increment=dp_compile, exact_policy_integration=integration,
            sampling=sampling, sample_replay=sample_replay))


def main():
    started = perf_counter()
    selection = json.loads((HERE/'selection.json').read_text())
    original_selection = json.loads((SOURCE/'selection.json').read_text())
    frozen = {r['root_id']: r for g in original_selection['by_day'].values() for r in g['roots']}
    roots = selection['roots']
    # Detect lost positive cases or a silently changed menu before evaluating.
    expected = [f'2023-07-16_0_{start}_origin' for start in range(27600, 32101, 900)]
    if [r['root_id'] for r in roots] != expected or any(r != frozen[r['root_id']] or len(r['full_options']) != 5 for r in roots):
        raise AssertionError('Six frozen V25 roots and menus must be preserved')
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    graph, _, _, _, metadata, _ = dated_graph(data, '2023-07-16', 'gtfs_station_max', True)
    with gzip.open(SOURCE/'results.json.gz', 'rt') as handle:
        originals = {r['root_id']: r for r in json.load(handle)['results']}
    shared = perf_counter()-started
    results = []
    for law_index, law in enumerate(LAWS):
        for index, root in enumerate(roots):
            row = run_root(graph, metadata, fit, root, law, law_index, index, originals[root['root_id']])
            results.append(row)
            print(law, root['root_id'], row['model']['expected_loss'],
                  {m: o['loss'] for m, o in row['actual']['outcomes'].items()}, flush=True)
            with gzip.open(HERE/'results.json.gz', 'wt') as handle:
                json.dump(dict(complete=False, results=results), handle, separators=(',', ':'))
    report = dict(shared_preparation_wall=shared, by_law={})
    for law in LAWS:
        selected = [r for r in results if r['law'] == law]
        sampled = summary([world for r in selected for world in r['samples']])
        for name, comparison in sampled['comparisons'].items():
            comparison['stratified_mc_standard_error'] = (
                sum(r['sampled']['comparisons'][name]['paired_difference_sd']**2/N_WORLDS
                    for r in selected)**.5/len(selected))
        report['by_law'][law] = dict(actual=summary([r['actual'] for r in selected]),
            sampled=sampled,
            proxy_expected_loss={m: mean(r['model']['expected_loss'][m] for r in selected) for m in METHODS},
            wall={key: sum(r['wall'][key] for r in selected) for key in selected[0]['wall']},
            actual_online_wall={m: sum(r['actual']['outcomes'][m]['online_wall'] for r in selected) for m in METHODS},
            v25_controls_reproduced=True)
    report['total_wall'] = perf_counter()-started
    with gzip.open(HERE/'results.json.gz', 'wt') as handle:
        json.dump(dict(complete=True, results=results, summary=report), handle, separators=(',', ':'))
    (HERE/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
