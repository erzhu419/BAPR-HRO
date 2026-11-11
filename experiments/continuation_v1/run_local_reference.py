"""Exact local commitment values from two archived public decision histories.

All kept connections lead straight to the destination. The only continuing
state is the current stop, so independent trip posteriors can be conditioned
factor by factor after the selected attempt and the ensuing public poll.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import cache
from itertools import product
import json
from math import isclose, prod
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.continuation_v1.analyze_probability import generating_parameters, replay_row
from experiments.reconstruction_v2.run_matched_panel import historical_priors
from src.committed_transit import CommittedState, WAIT, build_committed_model
from src.continuation_router import ContinuationRouter, nominal_action
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import TransitGraph
from src.trip_belief import NominalEvent


@dataclass(frozen=True)
class State:
    time: int
    supports: tuple[int, ...]
    unavailable: int
    seen: int


@dataclass(frozen=True)
class Service:
    cid: int
    departure: int
    arrival: int
    ready: int
    delays: tuple[int, ...]
    fitted: tuple[float, ...]
    generating: tuple[float, ...]
    original_support: int


@dataclass(frozen=True)
class Branch:
    mass: float
    state: State | None
    loss: float | None
    boarded: bool


def posterior(history, cid, parameters):
    delays, weights = history.posterior(cid, parameters)
    return {int(d): float(p) for d, p in zip(delays, weights) if p > 0}


def build_services(graph, context, ids, silent, generating):
    services = []
    stop, t = context['prediction']['stop'], context['prediction']['time']
    for cid in ids:
        c = graph.connections[cid]
        fitted = posterior(context['history'], cid, context['fitted_parameters'])
        true = posterior(context['history'], cid, generating)
        delays = tuple(sorted(fitted.keys() | true.keys()))
        original_support = len(delays)
        if cid in silent:
            # No future poll or attempt can distinguish these past outcomes.
            if any(d == 999 or c.dep_time + d >= t - 10 for d in delays):
                raise AssertionError('A collapsed old connection can still publish or board')
            mode = max(fitted, key=fitted.get)
            delays, fitted, true = (mode,), {mode: 1.0}, {mode: 1.0}
        ready = context['last_arrival'] + (0 if context['current_trip'] in (None, c.trip_id)
                                          else graph.stops[stop].min_transfer_time)
        services.append(Service(cid, c.dep_time, c.arr_time, ready, delays,
                                tuple(fitted.get(d, 0) for d in delays),
                                tuple(true.get(d, 0) for d in delays), original_support))
    return tuple(services)


class LocalReference:
    def __init__(self, graph, context, services, destination, departure, deadline, law):
        self.context, self.services = context, services
        self.stop, self.destination = context['prediction']['stop'], destination
        self.departure, self.deadline, self.cap = departure, deadline, deadline - departure
        self.law = law
        self.by_id = {s.cid: i for i, s in enumerate(services)}
        self.all_flags = (1 << len(services)) - 1
        active = [s for s in services if any(d != 999 and s.departure + d >= max(s.ready, context['prediction']['time'])
                                             for d in s.delays)]
        if (len(active) > 5 or len({s.arrival - s.departure for s in active}) != 1
                or any(not -5 <= d <= 64 for s in active for d in s.delays if d != 999)):
            raise AssertionError('Earliest public departure must attain the local earliest arrival')
        self.local_graph = TransitGraph()
        for sid in (self.stop, destination):
            self.local_graph.add_stop(graph.stops[sid])
        for service in services:
            self.local_graph.add_connection(replace(graph.connections[service.cid],
                                                    dep_distribution=None, arr_distribution=None))
        self.local_menu = frozenset(range(len(services)))
        self.root = State(context['prediction']['time'], tuple((1 << len(s.delays)) - 1 for s in services),
                          self.flags(context['unavailable']),
                          self.flags(int(event.split(':')[0]) for event in context['seen']
                                     if event.endswith((':departure', ':cancel'))))

    def flags(self, ids):
        return sum(1 << self.by_id[cid] for cid in set(ids) if cid in self.by_id)

    @cache
    def atoms(self, index, mask):
        return tuple(i for i in range(len(self.services[index].delays)) if mask & (1 << i))

    @cache
    def mass(self, index, mask, law):
        weights = getattr(self.services[index], law)
        return sum(weights[i] for i in self.atoms(index, mask))

    def actions(self, state):
        if state.time >= self.deadline:
            return ()
        return (None,) + tuple(s.cid for i, s in enumerate(self.services)
                               if not state.unavailable & (1 << i)
                               and state.time - 64 <= s.departure <= state.time + 25)

    @cache
    def poll_groups(self, index, mask, time, seen):
        if not mask or seen:
            return ((mask, False, False, 1.0),)
        service = self.services[index]
        groups = {}
        for atom in self.atoms(index, mask):
            d = service.delays[atom]
            event_time = service.departure if d == 999 else service.departure + d
            key = atom if time - 10 <= event_time <= time else None
            groups[key] = groups.get(key, 0) | (1 << atom)
        total = self.mass(index, mask, self.law)
        rows = []
        for atom, support in groups.items():
            weight = self.mass(index, support, self.law)
            if not weight:
                continue
            blocked = False
            if atom is not None:
                d = service.delays[atom]
                blocked = d == 999 or service.departure + d < time
            rows.append((support, atom is not None, blocked, weight / total))
        return tuple(rows)

    def publish(self, state, probability):
        options = [self.poll_groups(i, mask, state.time, bool(state.seen & (1 << i)))
                   for i, mask in enumerate(state.supports)]
        children = {}
        for combo in product(*options):
            supports, seen, blocked, masses = zip(*combo)
            unavailable = state.unavailable | sum(1 << i for i, flag in enumerate(blocked) if flag)
            observed = state.seen | sum(1 << i for i, flag in enumerate(seen) if flag)
            # The fate of a permanently unavailable trip cannot affect another
            # independent service. At most two kept trips can publish, so it
            # cannot affect the message limit either. Marginalize these factors.
            child = State(state.time, tuple(mask if not unavailable & (1 << i) else 0
                                           for i, mask in enumerate(supports)),
                          unavailable, observed & ~unavailable)
            children[child] = children.get(child, 0.) + probability * prod(masses)
        for child, weight in children.items():
            yield Branch(weight, child, None, False)

    @cache
    def branches(self, state, action):
        if action is None:
            time = state.time + 1
            if time >= self.deadline:
                return (Branch(1., None, self.cap, False),)
            return tuple(self.publish(replace(state, time=time), 1.))
        index = self.by_id[action]
        service = self.services[index]
        ready = max(state.time, service.ready)
        if ready >= self.deadline:
            return (Branch(1., None, self.cap, False),)
        end = min(self.deadline, ready + 12)
        mask = state.supports[index]
        groups = {}
        for atom in self.atoms(index, mask):
            d = service.delays[atom]
            if d == 999:
                kind, time = ('cancel', max(ready, service.departure)) if service.departure <= end else ('late', end)
            elif service.departure + d < ready:
                kind, time = 'miss', ready
            elif service.departure + d > end:
                kind, time = 'late', end
            else:
                kind, time = 'board', service.arrival + d
            groups[kind, time] = groups.get((kind, time), 0) | (1 << atom)
        total = self.mass(index, mask, self.law)
        results = []
        for (kind, time), support in groups.items():
            probability = self.mass(index, support, self.law) / total
            if not probability:
                continue
            if kind == 'board' or time >= self.deadline:
                results.append(Branch(probability, None, min(self.cap, time - self.departure), kind == 'board'))
            else:
                blocked = state.unavailable | ((1 << index) if kind in ('miss', 'cancel') else 0)
                next_supports = state.supports[:index] + (support,) + state.supports[index + 1:]
                results.extend(self.publish(State(time, next_supports, blocked, state.seen), probability))
        return tuple(results)

    @cache
    def nominal(self, state):
        means = {}
        for i, (service, mask) in enumerate(zip(self.services, state.supports)):
            ids = self.atoms(i, mask)
            operating = sum(service.fitted[j] for j in ids if service.delays[j] != 999)
            d = (sum(service.delays[j] * service.fitted[j] for j in ids if service.delays[j] != 999)
                 / operating if operating else None)
            total = self.mass(i, mask, 'fitted')
            means[i] = NominalEvent(d, service.departure + d if d is not None else None,
                                    service.arrival + d if d is not None else None,
                                    1 - operating / total if total else 1.)
        public = CommittedState(self.stop, state.time, self.context['current_trip'], self.context['last_arrival'],
                                frozenset(i for i in range(len(self.services)) if state.unavailable & (1 << i)),
                                initialized=True)
        action = nominal_action(self.local_graph, public, self.destination, means, self.local_menu, self.deadline, 25)
        cid = None if action == WAIT else self.services[action[1]].cid
        if cid not in self.actions(state):
            raise AssertionError('Nominal policy chose outside its shared local action menu')
        return cid

    @cache
    def arrival_distribution(self, index, mask, time):
        if not mask:
            return ((self.cap, 1.),)
        service = self.services[index]
        total = self.mass(index, mask, self.law)
        weights = getattr(service, self.law)
        by_loss = {}
        for atom in self.atoms(index, mask):
            if not weights[atom]:
                continue
            delay = service.delays[atom]
            loss = self.cap
            if delay != 999 and service.departure + delay >= max(time, service.ready):
                loss = min(self.cap, service.arrival + delay - self.departure)
            by_loss[loss] = by_loss.get(loss, 0.) + weights[atom] / total
        return tuple(by_loss.items())

    @cache
    def arrivals(self, state):
        return tuple(self.arrival_distribution(i, mask, state.time)
                     for i, mask in enumerate(state.supports) if not state.unavailable & (1 << i))

    @cache
    def value(self, state, optimal):
        if state.time >= self.deadline or state.unavailable == self.all_flags:
            return float(self.cap)
        arrivals = tuple(dist for dist in self.arrivals(state)
                         if any(cost < self.cap for cost, _ in dist))
        if not arrivals:
            return float(self.cap)
        if optimal:
            # Waiting minute by minute and boarding the first public eligible
            # departure attains the physical minimum in each world. All live
            # legs have equal duration, observed dep==time remains boardable,
            # and each trip's fixed readiness threshold is known at the root.
            return sum(min(cost for cost, _ in combo) * prod(p for _, p in combo)
                       for combo in product(*arrivals))
        return self.q(state, self.nominal(state), False)

    def q(self, state, action, optimal):
        return sum(b.mass * (b.loss if b.state is None else self.value(b.state, optimal))
                   for b in self.branches(state, action))

    def root_summary(self, action, optimal):
        if action is None:
            return {'Q': self.q(self.root, action, optimal), 'p_board': None,
                    'A_if_board': None, 'F_if_not_board': None}
        branches = self.branches(self.root, action)
        catch_mass = sum(b.mass for b in branches if b.boarded)
        failed_mass = sum(b.mass for b in branches if not b.boarded)
        if not isclose(catch_mass + failed_mass, 1., abs_tol=1e-10):
            raise AssertionError('Root outcome probabilities do not sum to one')
        catch = sum(b.mass * b.loss for b in branches if b.boarded)
        failure = sum(b.mass * (b.loss if b.state is None else self.value(b.state, optimal))
                      for b in branches if not b.boarded)
        return {'Q': catch + failure, 'p_board': catch_mass,
                'A_if_board': catch / catch_mass if catch_mass else None,
                'F_if_not_board': failure / failed_mass if failed_mass else None}


def adapter_preflight(reference):
    """Detect timing/announcement mismatches against the shared execution adapter."""
    context, local = reference.context, reference.local_graph
    variants = []
    for index, service in enumerate(reference.services):
        if len(service.delays) > 1:
            variants.extend((index, d) for d in (service.delays[0], service.delays[-1]))
    checks = waiting_checks = 0
    for changed_index, shift in variants:
        selected = [max(range(len(s.delays)), key=lambda j: s.fitted[j]) for s in reference.services]
        selected[changed_index] = reference.services[changed_index].delays.index(shift)
        events = {}
        for i, (s, atom) in enumerate(zip(reference.services, selected)):
            d = s.delays[atom]
            events[i] = VehicleEvent(d, d == 999, s.departure if d == 999 else s.departure + d,
                                     s.arrival if d == 999 else s.arrival + d)
        initial = CommittedState(reference.stop, reference.root.time, context['current_trip'], context['last_arrival'],
                                 frozenset(i for i in range(len(selected)) if reference.root.unavailable & (1 << i)),
                                 frozenset(f'{i}:{"cancel" if events[i].canceled else "departure"}'
                                           for i in range(len(selected)) if reference.root.seen & (1 << i)),
                                 initialized=True)
        model = build_committed_model(local, [(1., JourneyScenario(events))], reference.stop,
                                      reference.destination, reference.departure, reference.deadline,
                                      initial_state=initial)
        for action in reference.actions(reference.root):
            transformed = WAIT if action is None else ('select', reference.by_id[action])
            outcome = model.transition(model.scenarios[0], initial, transformed)
            post = outcome.state
            unavailable = sum(1 << i for i in post.unavailable_ids)
            observed = sum(1 << i for i in {int(e.split(':')[0]) for e in post.seen_events})
            matching = [b for b in reference.branches(reference.root, action)
                        if ((b.state is None and outcome.terminal and b.loss == post.time - reference.departure)
                            or (b.state is not None and not outcome.terminal and b.state.time == post.time
                                and b.state.unavailable == unavailable
                                and b.state.seen == observed & ~unavailable
                                and all(not mask or mask & (1 << atom)
                                        for mask, atom in zip(b.state.supports, selected))))]
            if not matching:
                raise AssertionError(f'Local timing/publication differs from committed adapter on {action}')
            checks += 1
        known = {i: s.departure + s.delays[0] for i, s in enumerate(reference.services)
                 if len(s.delays) == 1 and s.delays[0] != 999}
        public = initial
        while model.actions(public):
            departures = {int(m.event_id.split(':')[0]) for m in public.messages
                          if m.event_id.endswith(':departure') and m.actual_time == public.time}
            departures.update(i for i, departure in known.items() if departure == public.time)
            ready = [i for i in departures if i not in public.unavailable_ids
                     and public.time >= reference.services[i].ready]
            action = ('select', min(ready)) if ready else WAIT
            outcome = model.transition(model.scenarios[0], public, action)
            public = outcome.state
            if outcome.terminal:
                break
        realized = public.time - reference.departure + model.terminal_cost(model.scenarios[0], public)
        lower_bound = min([reference.cap] + [e.arrival - reference.departure for i, e in events.items()
                          if not e.canceled and i not in initial.unavailable_ids
                          and e.departure >= max(initial.time, reference.services[i].ready)])
        if realized != lower_bound:
            raise AssertionError('Causal wait-to-departure policy failed to attain its physical bound')
        waiting_checks += 1
    return checks, waiting_checks


def analyze(reference, archived_ids, chosen):
    all_actions = reference.actions(reference.root)
    rows = {a: {'optimal_tail': reference.root_summary(a, True),
                'nominal_tail': reference.root_summary(a, False)} for a in all_actions}
    full = min(all_actions, key=lambda a: rows[a]['optimal_tail']['Q'])
    restricted = min((a for a in all_actions if a is None or a in archived_ids),
                     key=lambda a: rows[a]['optimal_tail']['Q'])
    vf, vh = rows[full]['optimal_tail']['Q'], rows[restricted]['optimal_tail']['Q']
    qo, qn = rows[chosen]['optimal_tail']['Q'], rows[chosen]['nominal_tail']['Q']
    h_select = min(rows[a]['optimal_tail']['Q'] for a in all_actions if a in archived_ids)
    # A fixed observed action and a feasible public tail can never beat the
    # exact reference under the same law and future local choices.
    if min(vh - vf, qo - vh, qn - qo) < -1e-9:
        raise AssertionError('Local regret decomposition violates action inclusion')
    return {'law': reference.law, 'J_full': vf, 'J_H': vh,
            'full_action': full, 'H_action': restricted, 'archived_action': chosen,
            'archived_action_optimal_tail': qo, 'archived_action_nominal_tail': qn,
            'root_candidate_loss': vh - vf, 'first_action_loss_in_H': qo - vh,
            'J_H_commit_now': h_select, 'root_wait_option_value_in_H': h_select - vh,
            'archived_ranking_loss_among_H_commitments': qo - h_select,
            'nominal_tail_loss_after_archived_action': qn - qo,
            'root_actions': {'wait' if a is None else str(a): r for a, r in rows.items()}}


def score_repaired_full_graph(graph, context, row, destination, query):
    """Requery the repaired router at the archived history with frozen fits."""
    fitted = context['fitted_parameters']
    cap, limit = row['spec'][5:7]
    router = ContinuationRouter(deepcopy(graph), fitted, seed=row['private_seed'], samples=8,
                                max_time=cap, candidate_limit=limit)
    router.online_mean = router.online_cancel = False
    router._initial_cancels = {route: pars['cancel_rate'] for route, pars in fitted.items()}
    origin = next(s.id for s in graph.stops.values() if s.name == row['spec'][2])
    router.route(origin, destination, 490)
    router.trip_belief = context['history'].copy()
    stop, time = context['prediction']['stop'], context['prediction']['time']
    router.set_execution_context(context['current_trip'], context['last_arrival'], context['unavailable'])
    router.set_information_context(stop, time, context['current_trip'], context['last_arrival'],
                                   context['unavailable'], context['seen'], context['learned'])
    router.query_number = query
    started = perf_counter()
    selection = router.select_connection(stop, time, np.random.default_rng(0))
    raw_board = {}
    for cid in dict.fromkeys((*map(int, context['prediction']['candidates']),
                              *(cid for cid in router._scores if cid != 'wait'))):
        _, weights, caught = router._boarding_support(cid, router.information_state, router.trip_belief, fitted)
        raw_board[cid] = float(weights[caught].sum())
    return {'action': selection[0].connection_id if selection is not None else 'wait',
            'root_menu': list(router._scores), 'scores': router._scores, 'samples': router.samples,
            'raw_board_probabilities_for_archived_or_offered_connections': raw_board,
            'private_seed': row['private_seed'], 'query': query, 'rollout_steps': router.rollout_steps,
            'query_elapsed_seconds': perf_counter() - started}


def main():
    with (ROOT / 'data/zurich_wide.pkl').open('rb') as handle:
        graph = pickle.load(handle)
    with (ROOT / 'data/per_day_distributions.pkl').open('rb') as handle:
        days = pickle.load(handle)
    routes = {c.route for c in graph.connections}
    destination = next(s.id for s in graph.stops.values() if s.name == 'Zürich, Triemli')
    result = {
        'scope': 'Single-stop direct-service subproblems after the actual archived public receipt and feedback history. Future local polls publish only kept vehicles. No archived realized future is read.',
        'full_and_H': 'Full offers all kept direct connections and WAIT. H restricts only the captured decision to its archived direct menu and WAIT. Both use identical full local options thereafter. These are local admission losses.',
        'law': 'Independent per-trip posteriors conditioned on archived history, with complete positive supports on 68 and 1304. Old primary offers are collapsed only when every supported outcome departed before the future poll window and every attempt misses.',
        'exact_reduction': 'Every live direct leg takes one minute. Fixed public transfer thresholds may differ by trip. Delays are in [-5,64], hence a departure is in the scheduled window and PMF support when it occurs. At most two kept live departures can be published, and an occurred departure at the current clock is still boardable. Waiting for the first eligible departure attains the per-world earliest feasible arrival without foreknowledge. Terminal cost is averaged over independent conditioned factors. Public past facts of unavailable trips are marginalized because they cannot affect any kept live service or the message limit.',
        'tail': 'Nominal policy uses the repaired project earliest-mean rule and the same local future menu as the exact tail. Its forecasts use fitted weights even under generating-law evaluation. Both tails can wait. The separate repaired full-graph query re-creates current labels from frozen fitted parameters and uses eight private draws. No journey is replayed or assigned its local reference value.',
        'decomposition': 'J(nominal after archived action) - J_full = (J_H - J_full) + (Q*_archived - J_H) + (Q_nominal_archived - Q*_archived). All costs include the common elapsed prefix.',
        'completed_cases': 0, 'expected_cases': 2, 'cases': [],
    }
    for panel, index, query in (('matched', 2, 67), ('independent', 1, 48)):
        start = perf_counter()
        path = HERE / f'{panel}_results_methods/case_{index:03d}_conditioned_rescue.json'
        row = json.loads(path.read_text())
        date, stress = row['spec'][:2]
        true = generating_parameters(days[date], routes, stress)
        context = replay_row(row, graph, historical_priors(days, date, routes), true, capture_query=query)
        archived = tuple(map(int, context['prediction']['candidates']))
        # Keep archived ordering for the nominal tail and add the two named
        # direct choices. Detours are outside this single-stop subproblem.
        ids = tuple(cid for cid in dict.fromkeys((*archived, 68, 1304))
                    if graph.connections[cid].arr_stop == destination)
        omitted = [cid for cid in archived if cid not in ids]
        if len({graph.connections[cid].trip_id for cid in ids}) != len(ids):
            raise AssertionError('Local factorization requires separate trips')
        services = build_services(graph, context, ids, set(ids) - {68, 1304}, true)
        models = [LocalReference(graph, context, services, destination, 490, 490 + row['spec'][5], law)
                  for law in ('fitted', 'generating')]
        checks, waiting_checks = adapter_preflight(models[0])
        chosen = context['prediction']['connection']
        analyses = [analyze(model, archived, chosen) for model in models]
        repaired = score_repaired_full_graph(graph, context, row, destination, query)
        result['cases'].append({'source': str(path.relative_to(ROOT)), 'panel': panel, 'index': index, 'query': query,
                        'stop': context['prediction']['stop'], 'time': context['prediction']['time'],
                        'current_trip': context['current_trip'], 'last_arrival': context['last_arrival'],
                        'archived_journey_loss': row['loss'], 'archived_private_scores': context['prediction']['candidates'],
                        'H_archived_ids': [cid for cid in archived if cid in ids], 'omitted_archived_detours': omitted,
                        'full_local_ids': ids, 'services': [vars(s) for s in services],
                        'adapter_differential_checks': checks, 'analyses': analyses,
                        'waiting_policy_equivalence_checks': waiting_checks,
                        'repaired_full_graph_query': repaired,
                        'elapsed_seconds': perf_counter() - start})
        result['completed_cases'] = len(result['cases'])
        destination_file = HERE / 'local_reference_results.json'
        temp = destination_file.with_suffix('.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        temp.replace(destination_file)
        print(json.dumps({'panel': panel, 'query': query, 'adapter_checks': checks,
                          'values': [{k: a[k] for k in ('law', 'J_full', 'J_H', 'archived_action_optimal_tail',
                                                       'archived_action_nominal_tail')} for a in analyses],
                          'repaired_full_graph_action': repaired['action'],
                          'elapsed_seconds': result['cases'][-1]['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
