"""Complete the archived 45-second alighting witness on its original menu.

The original origin receipts, not later vehicle endpoints, condition each
trip. These receipts freeze when cid 10241 is boarded. All three original
alighting options retain their first-legal-tail policy. Their tail values and
the observed-arrival stopping chain are integrated in the frozen V31 model;
fresh shared-trip worlds are used only for independent physical evaluation.
"""

from copy import copy
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from md.BAPR_HRO_GPT_diagnosis_v5.board_gap.probe_board import alternative_options
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v25.run import physical_path, route_outcome
from experiments.reconstruction_v31.model import SegmentDurationBelief
from experiments.reconstruction_v32.conditional import departure_evidence, UnsupportedHistory
from experiments.reconstruction_v32.journey import TripWorlds, paired_statistics
from experiments.reconstruction_v32.run import killed_kernel, nominal_policy
from src.evolving_trip_belief import _ChainLaw
from src.onboard_stopping import OnboardStopping
from src.recourse_itinerary_router import build_trip_blocks
from src.risk_bounds import CommitmentBounds
from src.transfer_opportunity import TransferOpportunityScores


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT/'experiments/reconstruction_v25'
WITNESS = ROOT/'md/BAPR_HRO_GPT_diagnosis_v5/board_gap/headroom_45.json'
ROOT_ID = '2023-07-17_0_29400_origin'
FIRST_CID = 10241
PAIR_COUNT = 4096
PAIR_SEED = 3304501


def observed_origin_history(trace, cids, entry):
    """Recover only the information represented by the saved public receipts.

    A pre-readiness vehicle appearing only in ``seen`` is interval evidence;
    its exact departure is not recovered by peeking at archived metadata.
    """
    first_seen, exact = {}, {}
    for event in trace:
        for cid in event['seen']:
            first_seen.setdefault(cid, event['clock'])
        exact.update({cid: event['clock'] for cid in event['visible']})
    leave = trace[-1]['clock']
    return [dict(cid=cid, kind='observed', observed=exact[cid], entry=entry, through=leave)
            if cid in exact else
            dict(cid=cid, kind='seen_interval', entry=entry, through=first_seen[cid])
            if cid in first_seen else
            dict(cid=cid, kind='nonoccurrence', entry=entry, through=leave)
            for cid in cids]


def seen_interval_evidence(law, slot, entry, end):
    prior = law.marginal(slot)
    keep = ~law.canceled[slot] & (law.times[slot] >= entry) & (law.times[slot] <= end)
    mass = float(prior[keep].sum())
    if mass <= 0:
        raise UnsupportedHistory('The archived seen interval has no frozen operating mass')
    emissions = list(law.emissions)
    emissions[slot] = np.where(keep, emissions[slot], -np.inf)
    conditioned = _ChainLaw(law.times, law.initial, law._transition,
                            tuple(emissions), law.canceled, law.stochastic)
    return conditioned, mass


def freeze_root_history(belief, parameters, history):
    """Condition the existing trip chains, retaining the origin cut-off clock."""
    laws, receipts = {}, []
    for item in history:
        trip, slot = belief._positions[item['cid'], 'departure']
        law = laws.get(trip) or belief._law(trip, parameters)
        if item['kind'] == 'seen_interval':
            law, mass = seen_interval_evidence(law, slot, item['entry'], item['through'])
            evidence = dict(kind='seen_interval', prior_mass=mass)
        else:
            law, evidence = departure_evidence(law, slot, item['entry'], item['through'],
                                               item.get('observed'))
        laws[trip] = law
        receipts.append(dict(**item, trip=trip, evidence=evidence))
    conditioned = copy(belief)
    conditioned._law = lambda trip, params: laws[trip] if trip in laws else belief._law(trip, params)
    return conditioned, receipts


def compile_witness(graph, belief, parameters, root, options):
    """Integrate first-offer tail costs and the sufficient own-arrival chain."""
    deadline = root['deadline']
    bounds = CommitmentBounds(belief, parameters, graph, None, deadline)
    trip, departure_slot = belief._positions[FIRST_CID, 'departure']
    law = bounds._law(trip)
    slots = [belief._positions[op.first_path[-1], 'arrival'][1] for op in options]
    times = [law.times[s][~law.canceled[s]] for s in slots]
    costs, compilers = [], []
    for op, clocks in zip(options, times):
        compiler = TransferOpportunityScores(graph, belief, parameters, deadline, {FIRST_CID: op})
        stop = graph.connections[op.first_path[-1]].arr_stop
        ready = clocks+graph.stops[stop].min_transfer_time
        costs.append(np.asarray(compiler.tail_value(FIRST_CID, ready, 'exact')))
        compilers.append(compiler)
    kernels = [killed_kernel(bounds, trip, left, right) for left, right in zip(slots, slots[1:])]
    initial = law.marginal(departure_slot)[~law.canceled[departure_slot]]
    initial = initial @ killed_kernel(bounds, trip, departure_slot, slots[0])
    chain = OnboardStopping(times, costs, kernels, deadline)
    result = chain.solve(initial)
    policies = dict(result.policies)
    policies['rolling_mean'] = nominal_policy(chain, initial)
    for target in range(len(options)):
        actions = [np.full(len(t), stage == target, bool) for stage, t in enumerate(times)]
        policies[f'fixed_{target}'] = chain.evaluate_policy(actions, initial)
    return dict(chain=chain, result=result, policies=policies, initial=initial,
                compilers=compilers, times=times, tail_costs=costs)


def actual_policy(graph, metadata, root, options, compiled, board, policy):
    trace = []
    for stage, (option, clocks) in enumerate(zip(options, compiled['times'])):
        arrival, status = physical_path(metadata, option.first_path, board, root['deadline'])
        if status != 'arrived':
            return dict(loss=None if arrival is None else arrival-root['start'], status=status, trace=trace)
        if not clocks[0] <= arrival <= clocks[-1]:
            return dict(loss=None, status='unsupported_arrival', trace=trace)
        comparisons = policy.decision_costs[stage]
        alight, continuation = [float(np.interp(arrival, clocks, comparisons[:, column])) for column in (0, 1)]
        # Fixed commitments are intentionally executed as their fixed actions.
        if np.all(policy.actions[stage]) or not np.any(policy.actions[stage]):
            choose = bool(policy.actions[stage][0])
        else:
            choose = alight < root['deadline'] and alight <= continuation+1e-9
        trace.append(dict(stage=stage, arrival_cid=option.first_path[-1],
            observed_arrival=arrival, origin_evidence_frozen_at=board,
            alight_cost=alight, continue_cost=continuation, action='alight' if choose else 'continue'))
        if choose:
            final, state, tail = route_outcome(graph, metadata, option, board, root['deadline'])
            return dict(loss=None if final is None else final-root['start'], status=state,
                        alighting_stage=stage, tail=tail, trace=trace)
    return dict(loss=root['deadline']-root['start'], status='no_alight', trace=trace)


def evaluate_worlds(graph, root, options, compiled, worlds):
    costs = []
    for option in options:
        stop = graph.connections[option.first_path[-1]].arr_stop
        row = dict(first_path=option.first_path, transfer_stop=stop, tail_paths=list(option.tails.values()))
        costs.append(worlds.option(graph, row, root['deadline']))
    losses, stages = {}, {}
    for method, policy in compiled['policies'].items():
        value = np.full(worlds.count, root['deadline'], float)
        live = np.ones(worlds.count, bool)
        chosen = np.full(worlds.count, -1, int)
        for stage, option in enumerate(options):
            arrival = worlds.path(option.first_path, root['deadline'])
            live &= arrival < root['deadline']
            ids = np.flatnonzero(live)
            index = np.searchsorted(compiled['times'][stage], arrival[ids])
            if not np.array_equal(compiled['times'][stage][index], arrival[ids]):
                raise AssertionError('Drawn arrival is outside the frozen conditional grid')
            selected = ids[policy.actions[stage][index]]
            value[selected], chosen[selected], live[selected] = costs[stage][selected], stage, False
        losses[method], stages[method] = value-root['start'], chosen
    return losses, stages


def main():
    started = perf_counter()
    archived = json.loads(WITNESS.read_text())
    selection = json.loads((SOURCE/'selection.json').read_text())
    root = next(r for r in selection['by_day']['2023-07-17']['roots'] if r['root_id'] == ROOT_ID)
    board = archived['base']['trace'][-1]['clock']
    history = observed_origin_history(archived['base']['trace'],
        [o['first_cid'] for o in root['catalog_options']], root['scheduled_entry'])
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    segment = json.loads((ROOT/'experiments/reconstruction_v31/segment_fit.json').read_text())
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    graph, _, ids, _, metadata, _ = dated_graph(data, root['day'], 'gtfs_station_max', True)
    belief = SegmentDurationBelief(graph, duration_fit=fit['duration_fit'], segment_fit=segment,
        stop_identities={v:k for k,v in ids.items()}, public_occurrences=False,
        grid_step_seconds=20, **fit['increments'])
    parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    options = alternative_options(graph, build_trip_blocks(graph), root)[FIRST_CID]
    if [list(op.first_path) for op in options] != [o['first_path'] for o in archived['oracle_options']]:
        raise AssertionError('The archived three-alighting menu changed')
    conditioned, receipts = freeze_root_history(belief, parameters, history)
    compile_started = perf_counter()
    compiled = compile_witness(graph, conditioned, parameters, root, options)
    compile_wall = perf_counter()-compile_started
    actual = {method: actual_policy(graph, metadata, root, options, compiled, board, policy)
              for method, policy in compiled['policies'].items()}
    forced = [actual[f'fixed_{i}']['loss'] for i in range(len(options))]
    if forced != [r['loss'] for r in archived['oracle_options']]:
        raise AssertionError(('Archived physical alternatives changed', forced))
    paths = [p for op in options for p in [op.first_path, *op.tails.values()]]
    sample_started = perf_counter()
    worlds = TripWorlds(conditioned, parameters, paths, PAIR_COUNT, PAIR_SEED)
    losses, stages = evaluate_worlds(graph, root, options, compiled, worlds)
    paired = {method: paired_statistics(losses['rolling_fixed'], loss) for method, loss in losses.items()}
    sample_wall = perf_counter()-sample_started
    history_at_arrivals = []
    for stage, op in enumerate(options):
        now, status = physical_path(metadata, op.first_path, board, root['deadline'])
        row = dict(stage=stage, prefix=list(op.first_path), arrival=now, status=status,
                   stop=graph.stops[graph.connections[op.first_path[-1]].arr_stop].name)
        if status == 'arrived':
            row['costs'] = {method: dict(alight=float(np.interp(now, compiled['times'][stage], policy.decision_costs[stage][:, 0])),
                continuation=float(np.interp(now, compiled['times'][stage], policy.decision_costs[stage][:, 1])))
                for method, policy in compiled['policies'].items() if not method.startswith('fixed_')}
            row['commitments'] = {str(target): float(np.interp(now, compiled['times'][stage], value))
                for target, value in compiled['result'].fixed_commitment_values[stage].items()}
        history_at_arrivals.append(row)
    scores = {str(i): compiled['compilers'][i].board_score(FIRST_CID, board).exact-root['start']
              for i in range(len(options))}
    for index in range(len(options)):
        if abs(scores[str(index)]-(compiled['policies'][f'fixed_{index}'].root_value-root['start'])) > 1e-7:
            raise AssertionError('Fixed-path integral differs from the chain commitment value')
    sample_means = {method: dict(mean_loss=float(value.mean()),
        se=float(value.std(ddof=1)/np.sqrt(len(value)))) for method, value in losses.items()}
    report = dict(root_id=ROOT_ID, original_strong_actual_loss=1008,
        retrospective_best_actual_loss=963, retrospective_headroom=45,
        scope='Original three alighting points; exact frozen-model first-legal-tail values, causal own-arrival chain. Not full-network or arbitrary tail-policy optimality.',
        information='Original published origin receipts freeze at 29535; pre-readiness seen-only vehicle uses interval evidence, no offsite nonoccurrence updates.',
        history=receipts, fixed_board_scores=scores,
        model_expected_losses={m: p.root_value-root['start'] for m,p in compiled['policies'].items()},
        first_stop_survival=float(compiled['initial'].sum()),
        actual=actual, actual_arrival_diagnostics=history_at_arrivals,
        paired_model=paired, independent_model_means=sample_means,
        paired_world_count=PAIR_COUNT, paired_seed=PAIR_SEED,
        clock_projection='Fixed 20-second support; off-grid origin departures and actual decision queries use the inherited adjacent-state convention.',
        menu=[dict(first_path=op.first_path, tail_paths=list(op.tails.values())) for op in options],
        wall=dict(compile=compile_wall, independent_paired_evaluation=sample_wall, total=perf_counter()-started))
    HERE.mkdir(exist_ok=True)
    (HERE/'witness.json').write_text(json.dumps(report, indent=2)+'\n')
    with gzip.open(HERE/'witness_worlds.json.gz', 'wt') as handle:
        json.dump(dict(losses={m:v.tolist() for m,v in losses.items()},
                       alighting_stages={m:v.tolist() for m,v in stages.items()}), handle)
    print(json.dumps(dict(model=report['model_expected_losses'], actual={m:r['loss'] for m,r in actual.items()},
        history_at_arrivals=history_at_arrivals, paired=paired, wall=report['wall']), indent=2), flush=True)


if __name__ == '__main__':
    main()
