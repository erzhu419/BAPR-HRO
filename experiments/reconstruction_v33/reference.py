"""Original-law, two-root shared-vehicle public-history reference slices.

Selection is the first two V32 hubs with a direct root vehicle B and a different
root A having singleton-B continuations. This is a declared diagnostic slice:
keep B's direct option and every qualifying sequential A alighting variant.
Other roots/variants are not silently described as solved. No realized labels,
sampled worlds, modified probabilities, or compressed time grid select slices.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v31.model import SegmentDurationBelief
from experiments.reconstruction_v32.run import killed_kernel
from src.public_history_reference import FinitePublicModel, PublicHistoryReference, PublicTransition
from src.risk_bounds import CommitmentBounds


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'reconstruction_v25'


def select_slices(selection, graph_by_day):
    selected = []
    for root_index, root in enumerate(selection['selected']):
        graph = graph_by_day[root['day']]
        for hub_index, hub in enumerate(root['hubs']):
            if not hub['root_tail_overlap']:
                continue
            for direct in hub['options']:
                if direct['tail_paths']:
                    continue
                b = direct['first_cid']
                trip_b = graph.connections[b].trip_id
                by_a = {}
                for option in hub['options']:
                    if (option['first_cid'] != b and len(option['tail_paths']) == 1
                            and graph.connections[option['tail_paths'][0][0]].trip_id == trip_b):
                        by_a.setdefault(option['first_cid'], []).append(option)
                if not by_a:
                    continue
                a = min(by_a)
                variants = sorted(by_a[a], key=lambda o: len(o['first_path']))
                longest = variants[-1]['first_path']
                if not all(o['first_path'] == longest[:len(o['first_path'])] for o in variants):
                    continue
                selected.append(dict(root_index=root_index, hub_index=hub_index,
                    root_id=root['root_id'], day=root['day'], deadline=root['deadline'],
                    entry=hub['scheduled_ready']-graph.stops[hub['stop']].min_transfer_time,
                    ready=hub['scheduled_ready'], stop=hub['stop'], a=a, b=b,
                    variants=variants, direct=direct,
                    original_root_count=len(hub['variants']), original_variant_count=len(hub['options']),
                    scope='Two-root diagnostic action slice with all qualifying singleton-B A variants'))
                break
            if len(selected) == 2:
                return selected
    return selected


class SharedSlice:
    """Exact sparse event marginals from the unchanged original Markov laws."""
    def __init__(self, graph, belief, parameters, spec):
        self.graph, self.belief, self.spec = graph, belief, spec
        self.H, self.entry, self.ready = spec['deadline'], spec['entry'], spec['ready']
        self.bounds = CommitmentBounds(belief, parameters, graph, None, self.H)
        self.trip_a, self.slot_a = belief._positions[spec['a'], 'departure']
        self.trip_b, self.slot_b = belief._positions[spec['b'], 'departure']
        self.law_a, self.law_b = self.bounds._law(self.trip_a), self.bounds._law(self.trip_b)
        self.at = self.law_a.times[self.slot_a]
        self.bt = self.law_b.times[self.slot_b]
        self.ac = self.law_a.canceled[self.slot_a]
        self.bc = self.law_b.canceled[self.slot_b]
        self.ap = self.law_a.marginal(self.slot_a)
        self.bp = self.law_b.marginal(self.slot_b)
        self.a_operating = np.flatnonzero(~self.ac)
        self.a_position = {int(index): i for i, index in enumerate(self.a_operating)}
        self.stages = [belief._positions[o['first_path'][-1], 'arrival'][1] for o in spec['variants']]
        self.times = [self.law_a.times[slot][~self.law_a.canceled[slot]] for slot in self.stages]
        self.a_start = killed_kernel(self.bounds, self.trip_a, self.slot_a, self.stages[0])
        self.a_edges = [killed_kernel(self.bounds, self.trip_a, x, y)
                        for x, y in zip(self.stages, self.stages[1:])]
        end_b = belief._positions[spec['direct']['first_path'][-1], 'arrival'][1]
        goal_gain = np.maximum(0., self.H-self.law_b.times[end_b][~self.law_b.canceled[end_b]])
        self.direct_cost = np.full(len(self.bt), float(self.H))
        self.direct_cost[~self.bc] -= killed_kernel(self.bounds, self.trip_b, self.slot_b, end_b) @ goal_gain
        self.tail_costs = []
        for option, arrival_times in zip(spec['variants'], self.times):
            tail = option['tail_paths'][0]
            tail_slot = belief._positions[tail[0], 'departure'][1]
            tail_end = belief._positions[tail[-1], 'arrival'][1]
            if tail_slot < self.slot_b:
                raise ValueError('This forward reference slice requires B downstream of its root')
            message = np.eye(len(self.bt))
            for slot in range(self.slot_b, tail_slot):
                message = message @ self.bounds._posterior_edge(self.trip_b, slot)
            operating = ~self.law_b.canceled[tail_slot]
            tail_times = self.law_b.times[tail_slot][operating]
            tail_gain = killed_kernel(self.bounds, self.trip_b, tail_slot, tail_end) @ np.maximum(
                0., self.H-self.law_b.times[tail_end][~self.law_b.canceled[tail_end]])
            ready = arrival_times + graph.stops[option['transfer_stop']].min_transfer_time
            legal = (tail_times[None, :] >= ready[:, None]) & (tail_times[None, :] < self.H)
            # Service before boarding B may resume. After boarding B every
            # canceled path is killed. Root-B state is shared, never redrawn.
            self.tail_costs.append(self.H-(legal*tail_gain) @ message[:, operating].T)

    def actions(self, history):
        event = history[-1][1]
        if event[0] == 'platform':
            _, clock, visible = event
            if clock < self.ready:
                return ('wait',)
            return tuple((['board_b'] if visible & 2 else [])
                         + (['board_a'] if visible & 1 else []) + ['wait'])
        return ('alight', 'continue') if event[1]+1 < len(self.stages) else ('alight',)

    def _arrival(self, probabilities, stage, b_index):
        live = 0.
        for index in np.flatnonzero(probabilities > 0):
            p, clock = float(probabilities[index]), int(self.times[stage][index])
            live += p
            if clock >= self.H:
                yield PublicTransition(p, None, ('deadline', self.H), float(self.H))
            else:
                yield PublicTransition(p, ('onboard', stage, int(index), b_index),
                                       ('onboard', stage, clock))
        if live < 1.-1e-14:
            yield PublicTransition(max(0., 1.-live), None, ('failure', self.H), float(self.H))

    def transition(self, state, action):
        if state[0] == 'platform':
            _, a_index, b_index, seen = state
            if action == 'board_b':
                yield PublicTransition(1., None, 'complete_b', float(self.direct_cost[b_index]))
            elif action == 'board_a':
                yield from self._arrival(self.a_start[self.a_position[a_index]], 0, b_index)
            else:
                offers = [(int(t), bit) for t, canceled, bit in
                          ((self.at[a_index], self.ac[a_index], 1),
                           (self.bt[b_index], self.bc[b_index], 2))
                          if not canceled and not (seen & bit) and self.entry <= t < self.H]
                if not offers:
                    yield PublicTransition(1., None, ('deadline', self.H), float(self.H))
                else:
                    clock = min(t for t, _ in offers)
                    visible = sum(bit for t, bit in offers if t == clock)
                    yield PublicTransition(1., ('platform', a_index, b_index, seen | visible),
                                           ('platform', clock, visible))
        else:
            _, stage, index, b_index = state
            if action == 'alight':
                yield PublicTransition(1., None, 'complete_tail', float(self.tail_costs[stage][index, b_index]))
            else:
                yield from self._arrival(self.a_edges[stage][index], stage+1, b_index)

    def public_bounds(self, history):
        # These bounds cover every action, including a deliberately bad WAIT.
        # A feasible boarding value bounds the optimal node, but cannot serve
        # as an upper bound for another, already committed action branch.
        event = history[-1][1]
        clock = float(event[1] if event[0] == 'platform' else event[2])
        return min(clock, self.H), float(self.H)

    def model(self):
        initial = {('platform', int(a), int(b), 0): float(self.ap[a]*self.bp[b])
                   for a in np.flatnonzero(self.ap > 0) for b in np.flatnonzero(self.bp > 0)}
        return FinitePublicModel(initial, ('platform', self.entry, 0), self.actions,
            self.transition, self.public_bounds, description='Original V31 two-root shared-trip slice')


def structured_reference(adapter):
    """Exact two-offer decomposition plus ordinary conditional chain stopping.

    B's posterior stops changing after departure from the root. For each
    possible PUBLIC B posterior, integrate it into G before taking any minimum.
    Averaging policies optimized separately for hidden B states would be wrong.
    """
    x, H = adapter, float(adapter.H)
    a_operating = np.flatnonzero(~x.ac)
    b_operating = np.flatnonzero(~x.bc)

    # A public B departure reveals its operating finite state. Canceled B
    # states are never presented as observed departures.
    known_posteriors = np.eye(len(x.bt))[:, b_operating]
    a_times = x.at[a_operating]
    b_times = x.bt[b_operating]
    a_mass, b_mass = x.ap[a_operating], x.bp[b_operating]
    a_event = (a_times >= x.entry) & (a_times < H)
    b_event = (b_times >= x.entry) & (b_times < H)
    a_survival = x.ap[x.ac | (x.at < x.entry)].sum() + np.array([
        x.ap[(~x.ac) & (x.at > t)].sum() for t in b_times])
    b_weights = x.bp[:, None] * (x.bc[:, None] | (x.bt[:, None] < x.entry)
                                | (x.bt[:, None] > a_times[None, :]))
    b_survival = b_weights.sum(axis=0)
    # Zero survival means this A-first branch has zero probability. Fill a
    # harmless prior so matrix columns remain finite; those columns are unused.
    b_weights = np.divide(b_weights, b_survival[None, :],
        out=np.broadcast_to(x.bp[:, None], b_weights.shape).copy(), where=b_survival[None, :] > 0)
    posteriors = np.column_stack((known_posteriors, b_weights))
    G = [costs @ posteriors for costs in x.tail_costs]
    fixed = [dict() for _ in G]
    opt_values, opt_actions, rolling_actions = [None]*len(G), [None]*len(G), [None]*len(G)
    for r in range(len(G)-1, -1, -1):
        fixed[r][r] = G[r]
        for target in range(r+1, len(G)):
            fixed[r][target] = H-x.a_edges[r] @ (H-fixed[r+1][target])
        continuation = H-x.a_edges[r] @ (H-opt_values[r+1]) if r+1 < len(G) else np.full_like(G[r], H)
        opt_actions[r] = (G[r] < H) & (G[r] <= continuation)
        opt_values[r] = np.where(opt_actions[r], G[r], continuation)
        fixed_wait = (np.minimum.reduce([fixed[r][j] for j in range(r+1, len(G))])
                      if r+1 < len(G) else np.full_like(G[r], H))
        rolling_actions[r] = (G[r] < H) & (G[r] <= fixed_wait)

    def evaluate_onboard(actions):
        values, visits = [None]*len(G), [None]*len(G)
        for r in range(len(G)-1, -1, -1):
            continuation = H-x.a_edges[r] @ (H-values[r+1]) if r+1 < len(G) else np.full_like(G[r], H)
            values[r] = np.where(actions[r], G[r], continuation)
            next_visits = x.a_edges[r] @ visits[r+1] if r+1 < len(G) else np.zeros_like(G[r])
            visits[r] = (actions[r] != opt_actions[r]).astype(float) + (~actions[r])*next_visits
        return dict(values=values, board=H-x.a_start @ (H-values[0]),
                    disagreement_visits=x.a_start @ visits[0])

    optimum = evaluate_onboard(opt_actions)
    rolling = evaluate_onboard(rolling_actions)
    rollout_actions = []
    for r in range(len(G)):
        continuation = H-x.a_edges[r] @ (H-rolling['values'][r+1]) if r+1 < len(G) else np.full_like(G[r], H)
        rollout_actions.append((G[r] < H) & (G[r] <= continuation))
    onboard = dict(chain_stop=optimum, rolling_fixed=rolling,
                   one_step_rollout=evaluate_onboard(rollout_actions))
    control_modes = dict(conditional_single_chain_stop='chain_stop',
                        conditional_two_target_chain_stop='chain_stop',
                        conditional_single_rolling_fixed='rolling_fixed',
                        conditional_single_one_step_rollout='one_step_rollout')
    totals = {name: dict(gain=0., root_action_disagreements=0,
        root_disagreement_probability=0., root_bellman_contribution=0.,
        onboard_continuation_contribution=0., onboard_disagreement_visits=0.)
        for name in control_modes}
    probability, records = [], []

    def event(first, clock, weight, values, aboard):
        """Integrate actual cheap actions, then attribute against optimal Q."""
        probability.append(weight)
        legal = (('wait',) if clock < x.ready else
                 ('board_a', 'wait') if first == 'A' else
                 ('board_b', 'wait') if first == 'B' else ('board_a', 'board_b', 'wait'))
        optimal_q = values['chain_stop']
        optimal_action = min(legal, key=optimal_q.__getitem__)
        row = dict(first=first, clock=int(clock), probability=weight,
                   optimal_action=optimal_action, optimal_cost=optimal_q[optimal_action], controls={})
        for name, mode in control_modes.items():
            q = values[mode]
            action = min(legal, key=q.__getitem__)
            total = totals[name]
            total['gain'] += weight*(H-q[action])
            different = action != optimal_action
            total['root_action_disagreements'] += int(different)
            total['root_disagreement_probability'] += weight*different
            root_regret = max(0., optimal_q[action]-optimal_q[optimal_action])
            total['root_bellman_contribution'] += weight*root_regret
            # With the current action fixed, any value difference is entirely
            # the subsequent A stopping policy, under the same public posterior.
            following_regret = max(0., q[action]-optimal_q[action])
            total['onboard_continuation_contribution'] += weight*following_regret
            visit = aboard[mode].get(action, 0.)
            total['onboard_disagreement_visits'] += weight*visit
            row['controls'][name] = dict(action=action, cost=q[action],
                root_bellman_contribution=weight*root_regret,
                onward_contribution=weight*following_regret)
        records.append(row)

    for ai, (clock, mass) in enumerate(zip(a_times, a_mass)):
        if not a_event[ai] or not mass or b_survival[ai] <= 0:
            continue
        legal_b = b_event & (b_times > clock) & (b_times >= x.ready)
        wait = H-float(b_mass[legal_b] @ (H-x.direct_cost[b_operating][legal_b]))/b_survival[ai]
        weight = float(mass*b_survival[ai])
        column = len(b_times)+ai
        values = {mode: dict(board_a=float(data['board'][ai, column]), wait=wait)
                  for mode, data in onboard.items()}
        visits = {mode: dict(board_a=float(data['disagreement_visits'][ai, column]))
                  for mode, data in onboard.items()}
        event('A', clock, weight, values, visits)
    for bi, (clock, mass) in enumerate(zip(b_times, b_mass)):
        if not b_event[bi] or not mass or a_survival[bi] <= 0:
            continue
        legal_a = a_event & (a_times > clock) & (a_times >= x.ready)
        weight = float(mass*a_survival[bi])
        values, visits = {}, {}
        for mode, data in onboard.items():
            wait = H-float(a_mass[legal_a] @ (H-data['board'][legal_a, bi]))/a_survival[bi]
            values[mode] = dict(board_b=float(x.direct_cost[b_operating[bi]]), wait=wait)
            visits[mode] = dict(wait=float(a_mass[legal_a] @ data['disagreement_visits'][legal_a, bi])/a_survival[bi])
        event('B', clock, weight, values, visits)
    for ai in np.flatnonzero(a_event):
        for bi in np.flatnonzero(b_event & (b_times == a_times[ai])):
            clock = a_times[ai]
            weight = float(a_mass[ai]*b_mass[bi])
            values = {mode: dict(board_a=float(data['board'][ai, bi]),
                                 board_b=float(x.direct_cost[b_operating[bi]]), wait=H)
                      for mode, data in onboard.items()}
            visits = {mode: dict(board_a=float(data['disagreement_visits'][ai, bi]))
                      for mode, data in onboard.items()}
            event('both', clock, weight, values, visits)
    no_event = float((1.-a_mass[a_event].sum())*(1.-b_mass[b_event].sum()))
    mass_error = abs(sum(probability)+no_event-1.)
    if mass_error > 1e-8:
        raise AssertionError(('Public first-event partition lost mass', mass_error))
    exact_value = H-totals['conditional_single_chain_stop']['gain']
    for row in totals.values():
        row['expected_arrival'] = H-row.pop('gain')
        row['expected_loss_from_entry'] = row['expected_arrival']-x.entry
        row['gap_to_exact'] = row['expected_arrival']-exact_value
        row['attribution_error'] = abs(row['gap_to_exact']-row['root_bellman_contribution']
                                       -row['onboard_continuation_contribution'])
        if row['attribution_error'] > 1e-7 or row['gap_to_exact'] < -1e-7:
            raise AssertionError(('Cheap-policy Bellman attribution failed', row))
    return dict(value=exact_value, exact=True,
        scope='Original finite-law two-root slice, adaptive A alighting, shared B posterior',
        method='Public first-offer partition plus standard conditional chain stopping',
        first_event_mass_error=mass_error, public_first_event_cases=len(probability),
        posterior_columns=len(a_times)+len(b_times), no_root_offer_probability=no_event,
        controls=totals,
        equivalence='After the first root offer at most one unseen root remains; conditional single plus standard conditional chain stopping is exact. Two-target reduces to the same action.',
        decisions=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-histories', type=int, default=500)
    parser.add_argument('--max-transition-terms', type=int, default=1000000)
    args = parser.parse_args()
    started = perf_counter()
    fit = json.loads((SOURCE/'model_fit.json').read_text())
    seg = json.loads((HERE.parent/'reconstruction_v31/segment_fit.json').read_text())
    selection = json.loads((HERE.parent/'reconstruction_v32/selection.json').read_text())
    data, _ = load_city(SOURCE/'city_transfer_endpoints.json.gz')
    environments = {}
    for day in sorted({r['day'] for r in selection['selected']}):
        graph, _, stop_ids, _, _, _ = dated_graph(data, day, 'gtfs_station_max', True)
        model = SegmentDurationBelief(graph, duration_fit=fit['duration_fit'], segment_fit=seg,
            stop_identities={v:k for k, v in stop_ids.items()}, public_occurrences=False,
            grid_step_seconds=20, **fit['increments'])
        parameters = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled'])
                      for c in graph.connections}
        environments[day] = graph, model, parameters
    specs = select_slices(selection, {day: env[0] for day, env in environments.items()})
    (HERE/'reference_selection.json').write_text(json.dumps(dict(selection_uses_actual=False,
        rule=__doc__, selected=specs), indent=2)+'\n')
    rows = []
    for spec in specs:
        tick = perf_counter()
        adapter = SharedSlice(*environments[spec['day']], spec)
        model = adapter.model()
        compiled = perf_counter()-tick
        reference = PublicHistoryReference(model, max_histories=args.max_histories,
            max_transition_terms=args.max_transition_terms, max_depth=16)
        result = reference.solve()
        reference_wall = perf_counter()-tick-compiled
        structured_start = perf_counter()
        structured = structured_reference(adapter)
        structured_wall = perf_counter()-structured_start
        if not result.lower-1e-6 <= structured['value'] <= result.upper+1e-6:
            raise AssertionError('Structured value is outside the general public-history interval')
        row = dict(root_id=spec['root_id'], hub=spec['hub_index'], root_a=spec['a'], root_b=spec['b'],
            alighting_stages=len(adapter.stages), initial_support=len(model.initial_mass),
            law='Unchanged V31 finite law, 20-second grid, exact conditional messages',
            scope=spec['scope'], lower=result.lower, upper=result.upper, exact=result.exact,
            width=result.upper-result.lower, unresolved_mass=result.unresolved_mass,
            budget_exhausted=result.budget_exhausted, histories=result.histories,
            transition_terms=result.transition_terms, compile_wall=compiled,
            generic_tree_wall=reference_wall, structured_wall=structured_wall,
            structured_reference=structured,
            total_wall=perf_counter()-tick,
            root_action=result.action, decisions=list(result.decisions))
        rows.append(row)
        print({k:v for k,v in row.items() if k != 'decisions'}, flush=True)
    (HERE/'reference_results.json').write_text(json.dumps(dict(protocol=vars(args), rows=rows,
        wall=perf_counter()-started, conclusion='Finite-law intervals; no sampled-bank exactness or full-menu optimality claim'),
        indent=2)+'\n')


if __name__ == '__main__':
    main()
