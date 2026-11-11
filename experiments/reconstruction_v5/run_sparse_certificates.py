"""Full-city finite-bank lower/upper certificates at two archived public roots."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import ROOT, dated_graph, fit_training
from experiments.reconstruction_v5.run_chain_boundaries import apply_receipt
from src.causal_planning_router import CausalPlanningRouter
from src.committed_transit import CommittedState, WAIT, build_committed_model
from src.continuation_router import nominal_action
from src.evolving_trip_belief import EvolvingTripBelief, _SampleBank
from src.sparse_certificate import SparseCertificateIndex


def context(graph, metadata, archived, upto, fit):
    belief = EvolvingTripBelief(graph, minute=60, public_occurrences=False, **fit['increments'])
    last_arrival = archived['od']['ready_seconds']
    unavailable, learned = set(), set()
    for decision in archived['decisions'][:upto]:
        last_arrival = apply_receipt(belief, graph, metadata, decision, last_arrival)
        if isinstance(decision['action'], int) and decision['outcome'] == 'board':
            unavailable.add(decision['action'])
            learned.add(graph.connections[decision['action']].trip_id)
    decision = archived['decisions'][upto]
    state = CommittedState(decision['stop'], decision['clock_seconds'], decision['current_trip'],
                           last_arrival, frozenset(unavailable), learned_trips=frozenset(learned),
                           initialized=True)
    return belief, state


def setup(graph, metadata, archived, upto, fit, destination):
    started = perf_counter()
    belief, state = context(graph, metadata, archived, upto, fit)
    priors = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    deadline = archived['od']['deadline_seconds']
    router = CausalPlanningRouter(graph, priors, event_belief=belief, menu='full', algorithm='rollout',
                                 samples=4, work_budget=512, max_time=deadline - archived['od']['ready_seconds'])
    router.online_mean = router.online_cancel = False
    router._initial_cancels = {route: row['cancel_rate'] for route, row in priors.items()}
    router.route(state.stop, destination, archived['od']['ready_seconds'])
    parameters = router._parameters()
    seed = int(archived['date'].replace('-', '')) + archived['query'] * 1000
    rng = np.random.default_rng(np.random.SeedSequence([seed, upto]))
    # Capture the bank once at construction; event reads retain their original
    # mapping and cost. LazyTripScenario intentionally hides it behind a proxy.
    captured = []
    constructor = _SampleBank.__init__
    def capture(bank, *args):
        constructor(bank, *args)
        captured.append(bank)
    _SampleBank.__init__ = capture
    try:
        scenes = belief.sample(parameters, rng, 4)
    finally:
        _SampleBank.__init__ = constructor
    bank = [(Fraction(1, 4), scene) for scene in scenes]
    model = build_committed_model(graph, bank, state.stop, destination, state.time, deadline,
                                  lookahead=1500, initial_state=state)
    return {'belief': belief, 'state': state, 'router': router, 'parameters': parameters,
            'bank': bank, 'model': model, 'destination': destination, 'deadline': deadline,
            'setup_ms': (perf_counter() - started) * 1000, 'seed': seed, 'query_number': upto,
            'sample_bank': captured[0], 'post_sample_rng': rng}


def witness(setup, transition_budget=512):
    model, initial = setup['model'], setup['state']
    cap = setup['deadline'] - initial.time
    nominal_cache = {}
    losses, histories, world_ms, completed = [], [], [], []
    transitions = 0
    started = perf_counter()
    for world in model.scenarios:
        before_world = perf_counter()
        state, elapsed = initial, 0
        belief = setup['belief'].copy()
        trace, terminal = [], False
        for _ in range(model.horizon):
            actions = model.actions(state)
            if not actions:
                terminal = True
                break
            if transitions >= transition_budget:
                break
            if state not in nominal_cache:
                board = {a[1] for a in actions if a[0] == 'select'}
                action = (nominal_action(setup['router'].graph, state, setup['destination'],
                                        belief.mean_events(setup['parameters']), setup['router'].menu,
                                        setup['deadline'], 1500, root_boardings=board, minute=60)
                          if board else WAIT)
                nominal_cache[state] = action if action in actions else WAIT
            action = nominal_cache[state]
            outcome = model.transition(world, state, action)
            transitions += 1
            setup['router']._advance_belief(belief, state, outcome.state, outcome.terminal)
            trace.append([state.time, state.stop, list(action), outcome.state.time, outcome.state.stop])
            state = outcome.state
            elapsed += outcome.cost
            if outcome.terminal:
                terminal = True
                break
        losses.append(elapsed + model.terminal_cost(world, state) if terminal else cap)
        histories.append(trace)
        completed.append(terminal)
        world_ms.append((perf_counter() - before_world) * 1000)
    upper = sum(Fraction(world.probability) * Fraction(loss) for world, loss in zip(model.scenarios, losses))
    return {'upper': float(upper), 'world_loss_or_cap': losses, 'world_complete': completed,
            'transitions': transitions, 'wall_ms': (perf_counter() - started) * 1000,
            'world_ms_in_order': world_ms, 'root_action': histories[0][0][2] if histories[0] else None,
            'public_state_policy_cache_entries': len(nominal_cache), 'histories': histories}


def bank_work(setup):
    bank = setup['sample_bank']
    return {'sampled_trips': len(bank._sampled),
            'trip_laws_cached': sum(memo.law is not None for memo in bank.belief._cache.values()),
            'materialized_vehicle_events_all_worlds': sum(len(events) for worlds in bank._sampled.values()
                                                        for events in worlds)}


def run_case(graph, metadata, archived, upto, fit, destination):
    current = setup(graph, metadata, archived, upto, fit, destination)
    state, model = current['state'], current['model']
    before = perf_counter()
    index = SparseCertificateIndex(graph, current['bank'], destination, state.time, current['deadline'])
    index_ms = (perf_counter() - before) * 1000
    before = perf_counter()
    lower = index.action_lower(state, model.actions(state))
    lower_ms = (perf_counter() - before) * 1000
    after_lower_bank = bank_work(current)
    upper = witness(current)
    # Independent fresh belief/caches, same model tape: cold one-policy rollout
    # is not allowed to inherit the certificate's warmed trip inference.
    fresh = setup(graph, metadata, archived, upto, fit, destination)
    baseline = witness(fresh)
    before_ordered_lower_bank = bank_work(fresh)
    ordered_index = SparseCertificateIndex(graph, fresh['bank'], destination, state.time, fresh['deadline'])
    before = perf_counter()
    ordered_lower = ordered_index.action_lower(state, fresh['model'].actions(state))
    ordered_lower_ms = (perf_counter() - before) * 1000
    if ordered_lower != lower:
        raise AssertionError('Certificate values changed when the causal witness was evaluated first')
    same = upper['world_loss_or_cap'] == baseline['world_loss_or_cap'] and upper['histories'] == baseline['histories']
    bound = min(lower.values())
    if float(bound) > upper['upper'] + 1e-9:
        raise AssertionError('Causal witness falls below the alleged lower bound')
    removed = [action for action, value in lower.items() if value > Fraction(upper['upper'])]
    return {'date': archived['date'], 'query': archived['query'], 'archived_decision': upto,
            'root': {'stop': state.stop, 'clock': state.time, 'current_trip': state.current_trip,
                     'last_arrival': state.last_arrival, 'unavailable': sorted(state.unavailable_ids)},
            'samples': 4, 'seed': current['seed'], 'model_query_number': current['query_number'],
            'all_physical_root_actions': len(lower), 'lower': float(bound), 'upper': upper['upper'],
            'finite_bank_optimality_gap_upper': upper['upper'] - float(bound),
            'actions_safely_removed': [list(a) for a in removed], 'n_safely_removed': len(removed),
            'action_lower_remaining_seconds': [[list(action), float(value)] for action, value in lower.items()],
            'index_setup_ms': index_ms, 'context_and_lazy_bank_setup_ms': current['setup_ms'],
            'cold_action_lower_ms': lower_ms, 'lower_work': dict(index.stats),
            'bank_after_cold_lower': after_lower_bank,
            'witness_after_lower': upper, 'fresh_cold_witness': baseline,
            'witness_first_then_lower_ms': ordered_lower_ms,
            'witness_first_then_lower_work': dict(ordered_index.stats),
            'bank_after_witness_before_lower': before_ordered_lower_bank,
            'bank_after_witness_then_lower': bank_work(fresh),
            'fresh_setup_ms': fresh['setup_ms'], 'matched_policy_histories_and_losses': same,
            'clock_scope': 'All logical planning at the fixed public root; CPU latency is measured separately, not charged to passenger clock.'}


def main():
    before = perf_counter()
    source = ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz'
    data, _ = load_city(source)
    fit = fit_training(data, retain_no_pickup=True)
    day = data['holdout_dates'][0]
    graph, _, stops, _, metadata, _ = dated_graph(data, day, 'gtfs_station_max', True)
    preparation_ms = (perf_counter() - before) * 1000
    archive = json.loads((ROOT / 'experiments/reconstruction_v4/city_replay_results.json').read_text())
    archived = next(r for r in archive['rows'] if r['date'] == day and r['query'] == 0 and r['method'] == 'evolving_rollout')
    destination = stops[archived['od']['destination']]
    source_texts = {name: (ROOT / name).read_text() for name in
                    ['src/sparse_certificate.py', 'src/evolving_trip_belief.py', 'src/committed_transit.py',
                     'src/continuation_router.py', 'experiments/reconstruction_v5/run_sparse_certificates.py']}
    rows = []
    for upto in (0, 5):
        row = run_case(graph, metadata, archived, upto, fit, destination)
        rows.append(row)
        print(json.dumps({key: row[key] for key in ['archived_decision', 'all_physical_root_actions',
                         'lower', 'upper', 'n_safely_removed', 'cold_action_lower_ms',
                         'matched_policy_histories_and_losses']}), flush=True)
        output = {'scope': 'Two full-city public contexts, complete root actions, four synthetic posterior worlds; finite-bank certificate, not full-city exact causal optimum.',
                  'data': str(source.relative_to(ROOT)), 'training_dates': data['train_dates'],
                  'graph_connections': len(graph.connections), 'preparation_ms': preparation_ms,
                  'source_texts': source_texts, 'rows': rows}
        Path(__file__).with_name('sparse_certificate_results.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
