"""Value the two adverse decisions on their fixed archived public histories."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.continuation_v1.analyze_probability import generating_parameters, replay_row
from experiments.continuation_v1.historical_router import HistoricalContinuationRouter
from experiments.reconstruction_v2.run_matched_panel import historical_priors
from src.committed_transit import CommittedState, build_committed_model
from src.trip_belief import OUTCOMES

CASES = (
    ('primary_002', 'matched_results_methods/case_002_conditioned_rescue.json',
     67, (1927, 68, 1304), ((1927, 68), (1304, 68))),
    ('independent_001', 'independent_results_methods/case_001_conditioned_rescue.json',
     48, (68, 1304), ((1304, 68),)),
)
SEED = 99108


def monte_carlo_summary(values):
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return {'mean': mean, 'mc_se': se, 'approximate_mc_95_interval': [mean - 1.96 * se, mean + 1.96 * se]}


def action_outcomes(router, bank, cid):
    router._model = build_committed_model(
        router.graph, [(1.0 / len(bank), scene) for scene in bank],
        router.origin, router.destination, router.departure, router.journey_deadline,
        candidate_ids=router._fresh_menu, initial_state=router.information_state,
    )
    values, first_failures, _ = zip(*(router.rollout_one(world, cid) for world in router._model.scenarios))
    return np.asarray(values, dtype=float), np.asarray(first_failures, dtype=bool)


def direct_mass(router, parameters, cid):
    c = router.graph.connections[cid]
    state = router.information_state
    same_trip = bool(state.current_trip and c.trip_id == state.current_trip)
    ready = max(state.time, state.last_arrival + (
        0 if state.current_trip is None or same_trip else router.graph.stops[c.dep_stop].min_transfer_time))
    wait_end = min(router.journey_deadline, ready + 12)
    delays, weights = router._initial_belief.posterior(cid, parameters)
    catch = ((delays != 999) & (ready < router.journey_deadline)
             & (ready <= c.dep_time + delays) & (c.dep_time + delays <= wait_end))
    p = float(weights[catch].sum())
    if p and c.arr_stop != router.destination:
        raise ValueError('A successful nonterminal first leg requires a continuation valuation')
    A = float(np.dot(weights[catch], np.clip(c.arr_time + delays[catch] - router.departure,
                                            0, router.max_time)) / p) if p else None
    # Same outcome support as TripBelief for conditioning one selected trip.
    failure_grid = ~((OUTCOMES != 999) & (ready < router.journey_deadline)
                     & (ready <= c.dep_time + OUTCOMES) & (c.dep_time + OUTCOMES <= wait_end))
    return p, A, float(weights[~catch].sum()), failure_grid, ready


def evaluate_law(router, parameters, cid, draws):
    p, A, failure_mass, failure_grid, ready = direct_mass(router, parameters, cid)
    if failure_mass == 0.0:
        F = None
        costs = np.full(draws, A)
        n = 0
    else:
        # Only the sampling distribution receives this branch condition. The
        # actual tail policy starts at H, then learns the failed attempt.
        sampling_belief = router._initial_belief.copy()
        key = sampling_belief._trip_by_cid[cid]
        sampling_belief._mask(key)[:] &= failure_grid
        rng = np.random.default_rng(SEED)
        values = []
        for start in range(0, draws, 32):
            bank = sampling_belief.sample(parameters, rng, min(32, draws - start))
            loss, failed = action_outcomes(router, bank, cid)
            if not failed.all():
                raise AssertionError('The conditional bank includes a successful first attempt')
            values.extend(loss.tolist())
        F = monte_carlo_summary(values)
        costs = p * (A or 0.0) + failure_mass * np.array(values)
        n = draws
    return {'p_board_exact': p, 'A_capped_loss_given_board': A,
            'F_capped_loss_given_first_failure': F, 'Q': monte_carlo_summary(costs),
            'failure_rollouts': n, 'ready_time': ready}, costs


def replay_eight(router, row, q, targets):
    history = router._initial_belief
    bank = history.sample(router._parameters_now, np.random.default_rng(
        np.random.SeedSequence([row['private_seed'], q])), row['samples'])
    result = {}
    for cid in targets:
        loss, failed = action_outcomes(router, bank, cid)
        result[cid] = {
            'p_board': float((~failed).mean()),
            'A': float(loss[~failed].mean()) if (~failed).any() else None,
            'F': float(loss[failed].mean()) if failed.any() else None,
            'Q': float(loss.mean()),
        }
    return result


def audit_case(specification, draws):
    name, filename, query, targets, pairs = specification
    started = perf_counter()
    source = HERE / filename
    row = json.loads(source.read_text())
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as handle:
        graph = pickle.load(handle)
    with (ROOT / 'data/per_day_distributions.pkl').open('rb') as handle:
        days = pickle.load(handle)
    date, stress, source_name, destination_name, _, cap, limit = row['spec']
    routes = {c.route for c in graph.connections}
    priors = historical_priors(days, date, routes)
    true_parameters = generating_parameters(days[date], routes, stress)
    snapshot = replay_row(row, graph, priors, true_parameters, capture_query=query)
    prediction = snapshot['prediction']
    by_name = {stop.name: sid for sid, stop in graph.stops.items()}
    departure = next(event['time'] for event in row['events'] if event['event_type'] == 'depart')
    # Setting the historical offline values to the captured fitted parameters
    # reproduces the actual actor's marginal PMFs at this query.
    fitted = snapshot['fitted_parameters']
    router = HistoricalContinuationRouter(graph, fitted, max_time=cap, candidate_limit=limit)
    router.online_mean = router.online_cancel = False
    router._initial_cancels = {route: params['cancel_rate'] for route, params in fitted.items()}
    router.route(by_name[source_name], by_name[destination_name], departure)
    router.information_state = CommittedState(
        prediction['stop'], prediction['time'], snapshot['current_trip'], snapshot['last_arrival'],
        frozenset(snapshot['unavailable']), frozenset(snapshot['seen']),
        frozenset(snapshot['learned']), initialized=True,
    )
    router._initial_belief = snapshot['history']
    router._parameters_now = fitted
    router._initial_belief.mean_events(fitted)
    state = router.information_state
    archived_menu = list(map(int, prediction['candidates']))
    if router._snapshot_candidates(state) != archived_menu:
        raise AssertionError(f'{name}: historical candidate menu was not reproduced')
    router.candidate_limit = None
    ordered_pool = router._snapshot_candidates(state)
    router.candidate_limit = limit
    eight = replay_eight(router, row, query, targets)
    for cid in targets:
        original = prediction['candidates'].get(str(cid))
        if original and (abs(eight[cid]['Q'] - original['score']) > 1e-12 or
                         abs(1.0 - eight[cid]['p_board'] - original['first_attempt_failure_probability']) > 1e-12):
            raise AssertionError(f'{name}: historical eight-world estimate differs for {cid}')
    print(f'{name}: historical menu and logged values reproduced', flush=True)
    actions = {}
    samples_by_law = {'fitted': {}, 'generator': {}}
    for cid in targets:
        c = graph.connections[cid]
        actions[cid] = {
            'old_menu_member': cid in archived_menu, 'historical_pool_rank': ordered_pool.index(cid) + 1,
            'route': c.route, 'trip_id': c.trip_id, 'scheduled_departure': c.dep_time,
            'scheduled_arrival': c.arr_time, 'same_trip': c.trip_id == state.current_trip,
            'destination_of_leg': graph.stops[c.arr_stop].name,
            'historical_eight_worlds': eight[cid],
        }
        for law, parameters in (('fitted', fitted), ('generator', true_parameters)):
            result, samples = evaluate_law(router, parameters, cid, draws)
            actions[cid][law] = result
            samples_by_law[law][cid] = samples
        actions[cid]['signed_score_errors'] = {
            'historical_eight_minus_fitted': eight[cid]['Q'] - actions[cid]['fitted']['Q']['mean'],
            'fitted_minus_generator': monte_carlo_summary(
                samples_by_law['fitted'][cid] - samples_by_law['generator'][cid]),
            'historical_eight_minus_generator': eight[cid]['Q'] - actions[cid]['generator']['Q']['mean'],
        }
        print(f'{name}: {cid} fitted={actions[cid]["fitted"]["Q"]["mean"]:.3f} '
              f'generator={actions[cid]["generator"]["Q"]["mean"]:.3f}', flush=True)
    comparisons = [{
        'left_minus_right': [left, right],
        'historical_eight': eight[left]['Q'] - eight[right]['Q'],
        **{law: monte_carlo_summary(bank[left] - bank[right]) for law, bank in samples_by_law.items()},
    } for left, right in pairs]
    return {
        'name': name, 'source': str(source.relative_to(ROOT)), 'spec': row['spec'], 'query': query,
        'state': {'stop': graph.stops[state.stop].name, 'time': state.time,
                  'onboard_trip': state.current_trip, 'last_arrival': state.last_arrival},
        'archived_selected': prediction['connection'], 'archived_menu': archived_menu,
        'last_poll': row['polls'][query], 'actions': actions, 'ranking_differences': comparisons,
        'wall_seconds': perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=256)
    args = parser.parse_args()
    if args.draws < 2:
        parser.error('--draws must be at least 2 for Monte Carlo error estimates')
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(audit_case, case, args.draws) for case in CASES]
        cases = [future.result() for future in futures]
    output = {
        'scope': 'Two fixed public histories on the full archived 2000-connection graph.',
        'tail': 'Historical nominal earliest-arrival policy, frozen fitted route parameters and historical five-by-conditional-arrival-mean menus after the forced root action.',
        'p_A_F': 'p is analytic first-attempt boarding mass; A is exact capped journey loss conditional on boarding a direct leg; F averages the deployable frozen tail conditional on first-attempt failure. Q=p*A+(1-p)*F.',
        'laws': 'Fitted and original generated trip laws are conditioned on the identical received history. Both use the fitted public tail policy. The generated law is used in retrospective evaluation only.',
        'uncertainty': 'Intervals describe Monte Carlo numerical error for a fixed policy and law. Paired differences use matched conditional-bank indices and seeds. Zero standard error on a sampled failure tail records identical observed draws. The deterministic direct branch is marked by zero failure rollouts.',
        'failure_draws_per_action_and_law': args.draws, 'conditional_sampling_seed': SEED,
        'cases': cases,
    }
    destination = HERE / 'adverse_states.json'
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
