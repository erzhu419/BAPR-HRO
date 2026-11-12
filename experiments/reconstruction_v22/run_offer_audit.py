"""Enumerate reachable public offers for the frozen direct-menu policies."""

from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
from itertools import product
import json
from math import fsum, prod
from pathlib import Path
from time import perf_counter

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v15.run_attribution_banks import prepare
from experiments.reconstruction_v18.run_policy_pair import root_menu, support_additions
from experiments.reconstruction_v19.run_policy_pair import RELEASE_OFFSET
from src.information_planner import ExactPlanner
from src.offgrid_conditional import ActualClockRescue, ActualClockScores
from src.platform_rollout_search import PlatformRolloutSearch
from src.platform_transit import OBSERVE, WAIT, build_platform_model
from src.public_wait_recourse import compile_through_scores, through_itineraries
from src.simulator import JourneyScenario, VehicleEvent
from src.through_wait_value import ThroughWaitValue
from src.transit_graph import Connection, Stop, TransitGraph


HERE = Path(__file__).resolve().parent
GRIDS = (None, 60, 20, 10)
POLICIES = ('first_offer', 'nominal', 'exact')


class FirstDeparture:
    def __init__(self, clocks, weights, canceled, start, ready, deadline):
        operating = [(int(t), float(p)*(1-canceled)) for t, p in zip(clocks, weights, strict=True)
                     if p > 0]
        self.none = fsum([canceled, *(p for t, p in operating if not start <= t < deadline)])
        self.mass = {t: p for t, p in operating if start <= t < deadline}
        self.early = tuple((t, p) for t, p in self.mass.items() if t < ready)
        self.ticks = sorted(self.mass)
        self.tails = [fsum([self.none, *(self.mass[t] for t in self.ticks[i:])])
                      for i in range(len(self.ticks)+1)]

    def after(self, t):
        return self.tails[bisect_right(self.ticks, t)]


class OfferAudit:
    """Independent first-departure factors and the actual root decision code.

    The frozen controllers ignore other root arrivals and unrelated departures.
    A/B pre-readiness receipts retain their clocks because they affect expiry.
    Other pre-readiness receipts enter only through their observed identities.
    Conditional boarded values come from the common fixed same-trip suffix.
    """

    def __init__(self, belief, parameters, plan, root, goal, deadline, original, extras, *,
                 original_openers=False, archived_nominal=False):
        graph = belief.graph
        scores, direct = compile_through_scores(belief, parameters, graph, root, goal, deadline,
                                                original, extras)
        self.scores, self.root, self.deadline = scores, root, deadline
        self.ready = scores.ready
        self.release = root.time+RELEASE_OFFSET
        assert self.release < self.ready
        self.paths = {cid: route.primary for cid, route in scores.paths.items()}
        self.cids = sorted(self.paths)
        self.ab = set(original)
        assert len(self.cids) == len({graph.connections[cid].trip_id for cid in self.cids})
        self.rank = {cid: (graph.connections[path[-1]].arr_time-graph.connections[cid].dep_time, cid)
                     for cid, path in self.paths.items()}
        wait = ThroughWaitValue(scores, direct.keys() & scores.rows.keys(), margin=0)
        self.exact = ActualClockRescue(graph, plan, root, deadline, scores, direct, 0,
                                      ActualClockScores(belief, parameters, scores, goal),
                                      mode='projection', censor_unknown_extras=False, wait_value=wait)
        if not original_openers:
            # V22 compared only extras against the timed original controller.
            for cid in original:
                self.exact.suffixes.pop(cid, None)

        @lru_cache(maxsize=50_000)
        def wait_at(tick, seen, cursor):
            log = tuple((tick, cid, 'departure') for cid in seen)
            return wait.waiting_score(replace(root, time=tick, public_log=log, initialized=True), cursor)

        self.exact.waiting_score = lambda state, cursor: wait_at(
            state.time, tuple(sorted(cid for _, cid, _ in state.public_log)), cursor)
        self.nominal = PlatformRolloutSearch(graph, belief, parameters, root, goal, deadline,
                                             self.paths, None)
        if archived_nominal:
            # The V22 denominator treated departures before arrival as observed.
            self.nominal.root = replace(root, local_since=float('-inf'))
            _, self.nominal.nominal_wait = self.nominal._compile_marks(belief, parameters)
            self.nominal.root = root
        self.factors = {cid: FirstDeparture(*belief.event_distribution(cid, 'departure', parameters),
                                             root.time, self.ready, deadline) for cid in self.cids}
        self.ticks = sorted({t for law in self.factors.values() for t in law.ticks if t >= self.ready})
        assert all(t in scores.rows[cid] for cid, law in self.factors.items() for t in law.ticks
                   if t >= self.ready)

    def state(self, tick, history):
        log = tuple((t, cid, 'departure') for t, cid in history if t <= tick)
        visible = tuple(cid for t, cid, _ in log if t == tick and tick >= self.ready)
        return replace(self.root, time=tick, visible_connections=visible, public_log=log, initialized=True)

    def nominal_action(self, state):
        choices = tuple(self.nominal.option_index[cid] for cid in state.visible_connections)
        selected, _ = self.nominal.choose(state, None, 0, None, choices)
        return WAIT if selected is None else ('board', self.nominal.options[selected])

    def exact_before_first(self, history):
        tick = history[-1][0]
        binding = self.exact.bind('reachable_support_audit')
        earlier = sorted({t for t, cid in history if cid in self.ab and self.release < t < tick})
        cursor = self.release
        while cursor < tick:
            assert binding.action(self.state(cursor, history)) == WAIT
            expiry = binding.original.cursor.expiry
            j = bisect_right(earlier, cursor)
            candidates = [tick, *earlier[j:j+1]]
            if expiry is not None and expiry > cursor:
                candidates.append(expiry)
            cursor = min(candidates)
        return binding

    def exact_first(self, history):
        binding = self.exact_before_first(history)
        action = binding.action(self.state(history[-1][0], history))
        return action, binding.original.cursor

    def first_offers(self):
        for tick in self.ticks:
            if not any(tick in self.factors[cid].mass for cid in self.cids):
                continue
            alternatives = []
            for cid in self.cids:
                law = self.factors[cid]
                events = [(None, law.after(tick)), ((tick, cid), law.mass.get(tick, 0.))]
                if law.early:
                    if cid in self.ab:
                        events.extend(((t, cid), p) for t, p in law.early)
                    else:
                        events.append(((law.early[0][0], cid), fsum(p for _, p in law.early)))
                alternatives.append([(event, p) for event, p in events if p > 0])
            for combination in product(*alternatives):
                receipts = tuple(event for event, _ in combination if event is not None)
                if any(t == tick for t, _ in receipts):
                    yield tuple(sorted(receipts)), prod(p for _, p in combination)

    @lru_cache(maxsize=4096)
    def later_offers(self, since, active):
        offers = []
        for tick in self.ticks[bisect_right(self.ticks, since):]:
            if not any(tick in self.factors[cid].mass for cid in active):
                continue
            alternatives = []
            for cid in active:
                law = self.factors[cid]
                normalizer = law.after(since)
                alternatives.append([(seen, p/normalizer) for seen, p in (
                    (False, law.after(tick)), (True, law.mass.get(tick, 0.))) if p > 0])
            for combination in product(*alternatives):
                visible = tuple(cid for cid, (seen, _) in zip(active, combination, strict=True) if seen)
                if visible:
                    offers.append((tick, visible, prod(p for _, p in combination)))
        return offers

    def value(self, policy, action, tick, active, cursor=None):
        if action != WAIT:
            assert action[0] == 'board'
            return self.scores.rows[action[1]][tick].expected_arrival_or_deadline
        return self.continue_waiting(policy, tick, active, cursor)

    @lru_cache(maxsize=100_000)
    def continue_waiting(self, policy, since, active, cursor):
        # While still alighted, survival and the A/B cursor completely
        # describe all remembered first departures. Quiet expiries still run.
        already_seen = tuple((since, cid) for cid in self.cids if cid not in active)
        savings = []
        for tick, visible, probability in self.later_offers(since, active):
            history = (*already_seen, *((tick, cid) for cid in visible))
            state = self.state(tick, history)
            if policy == 'nominal':
                action, next_cursor = self.nominal_action(state), None
            else:
                binding = self.exact.bind('reachable_support_audit')
                binding.original.cursor = cursor
                while binding.original.cursor.expiry is not None and binding.original.cursor.expiry < tick:
                    expiry = binding.original.cursor.expiry
                    assert expiry > since and binding.action(self.state(expiry, already_seen)) == WAIT
                action = binding.action(state)
                next_cursor = binding.original.cursor
            remaining = tuple(cid for cid in active if cid not in visible)
            value = self.value(policy, action, tick, remaining, next_cursor)
            savings.append(probability*(self.deadline-value))
        return self.deadline-fsum(savings)

    def summarize(self):
        all_mass, tie_mass, loss = [], [], {p: [] for p in POLICIES}
        disagreement = {p: [] for p in POLICIES[1:]}
        nominal_exact = []
        for history, mass in self.first_offers():
            all_mass.append(mass)
            tick = history[-1][0]
            state = self.state(tick, history)
            if len(state.visible_connections) > 1:
                tie_mass.append(mass)
            exact, cursor = self.exact_first(history)
            actions = dict(first_offer=('board', min(state.visible_connections, key=self.rank.__getitem__)),
                           nominal=self.nominal_action(state), exact=exact)
            seen = {cid for _, cid in history}
            active = tuple(cid for cid in self.cids if cid not in seen)
            values = {p: self.value(p, action, tick, active, cursor if p == 'exact' else None)
                      for p, action in actions.items()}
            for p in POLICIES:
                loss[p].append(mass*(self.deadline-values[p]))
            if actions['nominal'] != actions['exact']:
                nominal_exact.append(mass)
            for p in POLICIES[1:]:
                if actions[p] != actions['first_offer']:
                    disagreement[p].append(dict(clock=tick, visible=state.visible_connections,
                                                observed_departures=history, mass=mass,
                                                first_offer_action=actions['first_offer'], action=actions[p],
                                                conditional_gain=values['first_offer']-values[p]))
        no_offer = prod(fsum([law.none, *(p for _, p in law.early)]) for law in self.factors.values())
        assert abs(fsum(all_mass)+no_offer-1) < 1e-9
        return dict(first_histories=len(all_mass), no_offer_mass=no_offer, simultaneous_mass=fsum(tie_mass),
                    mean_loss={p: self.deadline-self.root.time-fsum(loss[p]) for p in POLICIES},
                    nominal_exact_first_disagreement_mass=fsum(nominal_exact),
                    disagreement={p: dict(histories=len(rows), mass=fsum(r['mass'] for r in rows),
                                          wait_mass=fsum(r['mass'] for r in rows if r['action']==WAIT),
                                          positive_gain_mass=fsum(r['mass'] for r in rows if r['conditional_gain']>1e-8),
                                          net_gain=fsum(r['mass']*r['conditional_gain'] for r in rows),
                                          witnesses=sorted(rows, key=lambda r: -r['mass'])[:8])
                                  for p, rows in disagreement.items()})


def run_root(case):
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    original, extras = root_menu(graph, root, goal, deadline, plan)
    extras.update(support_additions(graph, belief, parameters, root, goal, deadline, original, extras))
    trips = {belief._trip_by_cid[cid] for cid in original.keys() | through_itineraries(graph, extras).keys()}
    results = []
    for step in GRIDS:
        before = perf_counter()
        law = belief if step is None else belief.with_grid_step(step, trips=trips)
        audit = OfferAudit(law, parameters, plan, root, goal, deadline, original, extras,
                           archived_nominal=True)
        result = audit.summarize()
        audit.later_offers.cache_clear()
        audit.continue_waiting.cache_clear()
        results.append(dict(grid_step=step, seconds=perf_counter()-before, **result))
        print(case, step, result['mean_loss'], {p: d['mass'] for p,d in result['disagreement'].items()}, flush=True)
    return dict(case=case, paths=audit.paths, refined_trips=sorted(trips), grids=results)


def resolution_example():
    graph = TransitGraph()
    for stop in (0,1):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    graph.add_connection(Connection(0, 'a', 'a', 0, 1, 0, 100))
    graph.add_connection(Connection(0, 'b', 'b', 0, 1, 0, 60))
    result = {}
    for name, departures in (('fine', (10, 50)), ('coarse', (0, 60))):
        # Condition on the public non-occurrence of B at clock zero.
        compatible = [d for d in departures if d > 0]
        worlds = [(1/len(compatible), JourneyScenario({0: VehicleEvent(0, False, 0, 100),
                    1: VehicleEvent(d, False, d, d+60)}, publications={})) for d in compatible]
        model = build_platform_model(graph, worlds, 0, 1, 0, 150, poll_interval=1)
        state = model.transition(model.scenarios[0], model.initial_state, OBSERVE).state
        sol = ExactPlanner(replace(model, initial_state=state)).solve()
        result[name] = dict(unconditional_mean_departure=fsum(departures)/len(departures),
                            unconditional_mean_arrival=60+fsum(departures)/len(departures),
                            action=sol.action, values={str(a): v for a,v in sol.action_values.items()})
    assert result['fine']['action']==WAIT and result['coarse']['action']==('board',0)
    return result


def main():
    example = resolution_example()
    with ProcessPoolExecutor(max_workers=len(CASES)) as workers:
        roots = list(workers.map(run_root, CASES))
    out = dict(status='complete', scope='Exact independent first-offer process for the frozen through menu, '
                                        'with common fitted fixed tails. Legacy point law is separate. '
                                        'Computed probabilities use floating point, not certified enclosures.',
               example=example, roots=roots)
    (HERE/'offer_audit.json').write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
