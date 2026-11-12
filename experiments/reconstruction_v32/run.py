"""Bounded onboard mechanism discovery, with shared-trip paired execution."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v25.run import departure, physical_path, public_offers, route_outcome
from experiments.reconstruction_v31.model import SegmentDurationBelief
from experiments.reconstruction_v32.journey import TailMenu, TripWorlds, TailOutcomes, paired_statistics
from experiments.reconstruction_v32.tail_values import integrate
from src.full_menu_opportunity import FullMenuOpportunity
from src.full_menu_waiting_dp import FullMenuWaitingDP
from src.onboard_stopping import OnboardStopping
from src.plan_waiting_controls import (
    SchedulePriority, SingleTargetOpportunity, CombinedWaitingOpportunity, TwoTargetOpportunity)
from src.risk_bounds import CommitmentBounds


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'
TAILS = ('first', 'nominal', 'single', 'combined', 'pair', 'exact')
PRIMARY = 'pair_one_step_rollout'


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def reference_at(menu, entry):
    ready = entry + menu.graph.stops[menu.hub['stop']].min_transfer_time
    reference = FullMenuOpportunity(menu, menu.ranks, entry, ready, menu.deadline)
    return reference, ready


def killed_kernel(bounds, trip, left, right):
    law = bounds._law(trip)
    result = np.eye(int((~law.canceled[left]).sum()))
    for slot in range(left, right):
        kernel = bounds._posterior_edge(trip, slot)
        result = result @ kernel[np.ix_(~law.canceled[slot], ~law.canceled[slot+1])]
    return result


def first_chain(graph, model, params, root):
    bounds = CommitmentBounds(model, params, graph, None, root['deadline'])
    trip, first_slot = model._positions[root['first_cid'], 'departure']
    law = bounds._law(trip)
    slots = [model._positions[h['arrival_cid'], 'arrival'][1] for h in root['hubs']]
    times = [law.times[slot][~law.canceled[slot]] for slot in slots]
    kernels = [killed_kernel(bounds, trip, a, b) for a, b in zip(slots, slots[1:])]
    first_times = law.times[first_slot][~law.canceled[first_slot]]
    initial = law.marginal(first_slot)[~law.canceled[first_slot]].copy()
    initial *= (first_times >= root['start']) & (first_times < root['deadline'])
    initial = initial @ killed_kernel(bounds, trip, first_slot, slots[0])
    return times, kernels, initial


def nominal_policy(chain, initial):
    """Replan a best fixed hub using conditional mean future arrival clocks."""
    actions, comparisons = [], []
    for stage, times in enumerate(chain.times):
        continuing = np.full(len(times), chain.deadline)
        transition = np.eye(len(times))
        for target in range(stage+1, len(chain.times)):
            transition = transition @ chain.transitions[target-1]
            survival = transition.sum(axis=1)
            means = np.divide(transition @ chain.times[target], survival,
                out=np.full(len(times), chain.deadline), where=survival > 0)
            tail = np.interp(means, chain.times[target], chain.alight_costs[target],
                             left=chain.deadline, right=chain.deadline)
            value = chain.deadline-survival*(chain.deadline-tail)
            continuing = np.minimum(continuing, value)
        alight = chain.alight_costs[stage]
        actions.append((alight < chain.deadline) & (alight <= continuing+1e-9))
        comparisons.append(np.column_stack((alight, continuing)))
    return chain.evaluate_policy(actions, initial, comparisons)


def board_first(graph, metadata, root):
    event = departure(metadata[root['first_cid']])
    if event is None:
        return None, 'unresolved_first_departure'
    clock, canceled = event
    if canceled or not root['start'] <= clock < root['deadline']:
        return root['deadline'], 'unavailable_first_vehicle'
    return clock, 'boarded'


def actual_tail(menu, metadata, entry, method):
    reference, ready = reference_at(menu, entry)
    single = SingleTargetOpportunity(reference)
    combined = CombinedWaitingOpportunity(single)
    if method == 'pair':
        control = TwoTargetOpportunity(combined)
    elif method == 'exact':
        control = FullMenuWaitingDP(reference)
    elif method == 'nominal':
        control = SchedulePriority({cid: rank[0] for cid, rank in menu.ranks.items()})
    else:
        control = dict(single=single, combined=combined, first=reference)[method]
    offers = public_offers(metadata, menu.cids, entry, ready, menu.deadline)
    trace = []
    if offers is None:
        return None, 'unresolved_tail_departure', trace
    for public in offers:
        if method == 'first':
            choice, receipt = reference.choose(public['visible'], public['seen'], public['clock'], 'first_offer')
        else:
            choice, receipt = control.choose(public['visible'], public['seen'], public['clock'])
        trace.append(dict(**public, choice=choice, receipt=receipt))
        if choice is None:
            continue
        score, index = menu.best(choice, public['clock'])
        if index is None:
            return None, 'unsupported_tail_board_clock', trace
        option = menu.compilers[index].options[choice]
        trace[-1].update(variant=index, first_path=option.first_path, board_score=score)
        arrival, status, tail = route_outcome(menu.graph, metadata, option, public['clock'], menu.deadline)
        trace[-1]['final_tail'] = tail
        return arrival, status, trace
    return menu.deadline, 'no_tail_board', trace


def actual_policy(graph, metadata, root, menus, times, policy, tail_method):
    began = perf_counter()
    board, status = board_first(graph, metadata, root)
    trace, tail_trace, arrival = [], [], board
    if status == 'boarded':
        arrival, status = root['deadline'], 'no_alighting'
        for stage, hub in enumerate(root['hubs']):
            now, state = physical_path(metadata, hub['prefix'], board, root['deadline'])
            if state != 'arrived':
                arrival, status = now, state
                break
            if now < times[stage][0] or now > times[stage][-1]:
                arrival, status = None, 'unsupported_onboard_arrival'
                break
            costs = policy.decision_costs[stage]
            alight, continuation = [float(np.interp(now, times[stage], costs[:, j])) for j in (0, 1)]
            chosen = alight < root['deadline'] and alight <= continuation+1e-9
            trace.append(dict(stage=stage, stop=hub['stop'], arrival_cid=hub['arrival_cid'],
                observed_arrival=now, visible_information='own arrival only',
                alight_cost=alight, continue_cost=continuation, action='alight' if chosen else 'continue'))
            if chosen:
                arrival, status, tail_trace = actual_tail(menus[stage], metadata, now, tail_method)
                break
    return dict(loss=None if arrival is None else arrival-root['start'], status=status,
                trace=trace, tail_trace=tail_trace, online_and_replay_wall=perf_counter()-began)


def sampled_policy(root, worlds, times, policy, tail_values):
    departure_times, canceled = worlds.events[root['first_cid'], 'departure']
    live = ~canceled & (departure_times >= root['start']) & (departure_times < root['deadline'])
    values = np.full(worlds.count, root['deadline'], dtype=float)
    chosen = np.full(worlds.count, -1, dtype=int)
    for stage, hub in enumerate(root['hubs']):
        arrival = worlds.path(hub['prefix'], root['deadline'])
        live &= arrival < root['deadline']
        ids = np.flatnonzero(live)
        if not len(ids):
            continue
        index = np.searchsorted(times[stage], arrival[ids])
        if not np.array_equal(times[stage][index], arrival[ids]):
            raise AssertionError('Operating arrival absent from the declared chain states')
        take = policy.actions[stage][index]
        rows, cols = ids[take], index[take]
        values[rows] = tail_values[stage][cols, rows]
        chosen[rows] = stage
        live[rows] = False
    return values-root['start'], chosen


def run_root(graph, metadata, model, params, root, protocol, root_index):
    started = perf_counter()
    if any(h['incoming_trip_reused'] for h in root['hubs']):
        return dict(root_id=root['root_id'], status='requires_joint_first_vehicle_tail_belief')
    times, transitions, initial = first_chain(graph, model, params, root)
    menus = [TailMenu(graph, model, params, h, root['deadline']) for h in root['hubs']]
    paths = [root['first_path'], *(p for m in menus for p in m.paths)]
    draw_start = perf_counter()
    q_seed = protocol['quadrature_seed']+1000*root_index
    e_seed = protocol['evaluation_seed']+1000*root_index
    q = TripWorlds(model, params, paths, protocol['quadrature_worlds'], q_seed)
    evaluation = TripWorlds(model, params, paths, protocol['evaluation_worlds'], e_seed)
    draw_wall = perf_counter()-draw_start
    estimates = {method: [] for method in TAILS}
    realized = {method: [] for method in TAILS}
    hub_reports = []
    tail_start = perf_counter()
    for stage, (menu, states) in enumerate(zip(menus, times)):
        tick = perf_counter()
        quadrature = TailOutcomes(menu, q)
        validation = TailOutcomes(menu, evaluation)
        means = {m: np.full(len(states), root['deadline'], float) for m in TAILS}
        se = {m: np.zeros(len(states), float) for m in TAILS}
        outcomes = {m: np.full((len(states), evaluation.count), root['deadline'], float) for m in TAILS}
        proxy = {m: np.full(len(states), root['deadline'], float) for m in TAILS}
        for index, entry in enumerate(states):
            if entry >= root['deadline']:
                continue
            reference, ready = reference_at(menu, int(entry))
            summary, tables = integrate(reference, scheduled_arrivals={cid: menu.ranks[cid][0] for cid in menu.cids})
            fit_values = quadrature.evaluate(tables, entry, ready, TAILS)
            test_values = validation.evaluate(tables, entry, ready, TAILS)
            for method in TAILS:
                means[method][index] = fit_values[method].mean()
                se[method][index] = fit_values[method].std(ddof=1)/np.sqrt(q.count)
                outcomes[method][index] = test_values[method]
                proxy[method][index] = summary['expected_arrival'][method]
        for method in TAILS:
            estimates[method].append(means[method])
            realized[method].append(outcomes[method])
        hub_reports.append(dict(stop=menu.hub['stop'], states=states, estimated_costs=means,
            estimate_se=se, fixed_mark_proxy_costs=proxy,
            root_tail_overlap=menu.hub['root_tail_overlap'], entry_count=len(states),
            wall=perf_counter()-tick))
        print(root['root_id'], 'hub', stage, 'entries', len(states),
              'seconds', round(hub_reports[-1]['wall'], 2), flush=True)
    tail_wall = perf_counter()-tail_start
    solve_start = perf_counter()
    policies = {}
    model_values, tail_methods = {}, {}
    for tail_method in TAILS:
        chain = OnboardStopping(times, estimates[tail_method], transitions, root['deadline'])
        result = chain.solve(initial)
        nominal = nominal_policy(chain, initial)
        name = tail_method+'_rolling_nominal'
        policies[name], tail_methods[name] = nominal, tail_method
        model_values[name] = nominal.root_value-root['start']
        if tail_method == 'pair':
            for method, policy in result.policies.items():
                name = 'pair_'+method
                policies[name], tail_methods[name] = policy, 'pair'
                model_values[name] = policy.root_value-root['start']
    solve_wall = perf_counter()-solve_start
    outcomes, choices = {}, {}
    for method, policy in policies.items():
        outcomes[method], choices[method] = sampled_policy(root, evaluation, times, policy, realized[tail_methods[method]])
    actual = {method: actual_policy(graph, metadata, root, menus, times, policy, tail_methods[method])
              for method, policy in policies.items()}
    paired = {method: paired_statistics(outcomes[PRIMARY], values) for method, values in outcomes.items()}
    # Reach probabilities and Bellman residuals are for the supplied G, not
    # for clairvoyant completion in the sampled worlds.
    probability = initial.copy()
    attribution = []
    optimum = policies['pair_optimal']
    primary = policies[PRIMARY]
    for stage, clock_states in enumerate(times):
        continuation = (root['deadline']-transitions[stage] @ (root['deadline']-optimum.values[stage+1])
                        if stage+1 < len(times) else np.full(len(clock_states), root['deadline']))
        selected_value = np.where(primary.actions[stage], estimates['pair'][stage], continuation)
        residual = selected_value-optimum.values[stage]
        for index in np.flatnonzero((residual > 1e-7) & (probability > 0)):
            attribution.append(dict(stage=stage, clock=int(clock_states[index]),
                probability=float(probability[index]), conditional_gap=float(residual[index]),
                contribution=float(probability[index]*residual[index])))
        if stage+1 < len(times):
            probability = (probability*~primary.actions[stage]) @ transitions[stage]
    return dict(root_id=root['root_id'], day=root['day'], layer=root['selected_layer'], status='completed',
        initial_survival=float(initial.sum()), quadrature_seed=q_seed, evaluation_seed=e_seed,
        model_values=model_values, paired=paired,
        sampled_losses=outcomes, sampled_alighting_stages=choices, actual=actual,
        attribution=attribution, tail_estimates=hub_reports,
        wall=dict(total=perf_counter()-started, model_draw=draw_wall, tail_compile_and_integrate=tail_wall,
                  chain_solve_all_controls=solve_wall,
                  actual_all_controls=sum(x['online_and_replay_wall'] for x in actual.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-index', type=int)
    parser.add_argument('--selection', type=Path, default=HERE/'selection.json')
    parser.add_argument('--output', type=Path, default=HERE/'roots')
    args = parser.parse_args()
    protocol = json.loads((HERE/'protocol.json').read_text())
    selection = json.loads(args.selection.read_text())
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    segment = json.loads((HERE.parent/'reconstruction_v31/segment_fit.json').read_text())
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    days = {}
    args.output.mkdir(exist_ok=True, parents=True)
    for index, root in enumerate(selection['selected']):
        if args.root_index is not None and index != args.root_index:
            continue
        day = root['day']
        if day not in days:
            graph, _, ids, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
            model = SegmentDurationBelief(graph, duration_fit=fit['duration_fit'], segment_fit=segment,
                stop_identities={v:k for k,v in ids.items()}, public_occurrences=False,
                grid_step_seconds=20, **fit['increments'])
            params = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
            days[day] = graph, metadata, model, params
        output = args.output/f'{index:02}.json.gz'
        if output.exists():
            print('reuse completed', index, root['root_id'], flush=True)
            continue
        result = run_root(*days[day], root, protocol, index)
        with gzip.open(output, 'wt') as handle:
            json.dump(result, handle, default=jsonable)
        print('completed', index, root['root_id'], result['wall'], flush=True)


if __name__ == '__main__':
    main()
