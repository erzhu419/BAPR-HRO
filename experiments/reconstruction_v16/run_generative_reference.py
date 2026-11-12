"""Exact public-belief branching for small independent ordered trip chains.

The causal solve stores masks on event clocks of each trip, not indices of
pre-drawn complete worlds. A departure, its absence, or a ridden arrival
filters the corresponding clock. The factorized posterior follows because
each public observation is an intersection of per-trip evidence events once
the action and preceding public state are fixed. Same-trip future clocks use
the original conditional chain probabilities. A remote factor gets no new
restriction merely because a local factor has been observed.

These fixtures use the V13 local occurrence, WAIT, ALIGHT and same-trip
inertia rules, no footpaths or messages, and explicit finite clock laws.
Deliberate hindsight comparators and independent small-model checks enumerate
worlds. The full causal generative solve does not.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from functools import lru_cache
from itertools import combinations_with_replacement, product
import json
from math import factorial, isclose, prod
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v14.run_exact_diagnostics import graph
from src.causal_prefix import causal_prefix_lower
from src.information_planner import ExactPlanner
from src.platform_transit import ALIGHT, OBSERVE, WAIT, PlatformState, build_platform_model
from src.simulator import JourneyScenario, VehicleEvent


def pmf(weights):
    total = sum(weights)
    assert total > 0
    return tuple(Q(w, total) for w in weights)


@dataclass(frozen=True)
class TripLaw:
    legs: tuple
    clocks: tuple
    prior: tuple
    transitions: tuple

    @classmethod
    def create(cls, legs, clocks, prior, transitions):
        assert len(clocks) == 2 * len(legs) == len(transitions) + 1
        assert len(prior) == len(clocks[0])
        assert all(tuple(sorted(set(slot))) == slot for slot in clocks)
        for i, matrix in enumerate(transitions):
            assert len(matrix) == len(clocks[i])
            for tick, row in zip(clocks[i], matrix, strict=True):
                assert len(row) == len(clocks[i + 1])
                assert all(w >= 0 and (w == 0 or clock >= tick)
                           for w, clock in zip(row, clocks[i + 1], strict=True))
        return cls(tuple(legs), tuple(clocks), pmf(prior), tuple(tuple(pmf(row) for row in matrix)
                                                                for matrix in transitions))

    @property
    def unrestricted(self):
        return tuple((1 << len(slot)) - 1 for slot in self.clocks)

    def forward_mass(self, masks):
        forward = tuple(p if masks[0] >> i & 1 else 0 for i, p in enumerate(self.prior))
        for mask, transition in zip(masks[1:], self.transitions, strict=True):
            forward = tuple(sum(p * row[j] for p, row in zip(forward, transition, strict=True))
                            if mask >> j & 1 else 0 for j in range(len(transition[0])))
        return sum(forward)

    def deterministic_masks(self, masks):
        for path in product(*(tuple(1 << i for i in range(len(slot)) if mask >> i & 1)
                              for slot, mask in zip(self.clocks, masks, strict=True))):
            mass = self.forward_mass(path)
            if mass:
                yield mass, path


class PublicBelief:
    def __init__(self, timetable, laws, goal=1, cap=30, mct=0):
        self.graph = graph(timetable)
        self.laws = tuple(laws)
        self.initial = PlatformState(0, 0)
        self.goal, self.cap, self.mct = goal, cap, mct
        self.prior = tuple(law.unrestricted for law in self.laws)
        self.locations = defaultdict(list)
        self.slots = {}
        self.factor_work = 0
        for factor, law in enumerate(laws):
            for position, cid in enumerate(law.legs):
                c = self.graph.connections[cid]
                dep, arr = (factor, position * 2), (factor, position * 2 + 1)
                self.locations[c.dep_stop].append((dep, cid))
                self.locations[c.arr_stop].append((arr, None))
                self.slots[cid] = (dep, arr)
        assert sorted(self.slots) == list(range(len(timetable)))
        self.mass = lru_cache(None)(self._mass)
        self.step = lru_cache(None)(self._step)
        self.value = lru_cache(None)(self._value)

    def _mass(self, factor, masks):
        self.factor_work += 1
        return self.laws[factor].forward_mass(masks)

    def condition(self, belief, slot, mask):
        i, j = slot
        before = belief[i]
        after = before[:j] + (before[j] & mask,) + before[j + 1:]
        if after == before:
            return Q(1), belief
        probability = self.mass(i, after) / self.mass(i, before)
        return probability, belief[:i] + (after,) + belief[i + 1:]

    def marginal(self, belief, slot):
        i, j = slot
        return {t: p for index, t in enumerate(self.laws[i].clocks[j])
                if (p := self.condition(belief, slot, 1 << index)[0])}

    def can_board(self, state, cid):
        c = self.graph.connections[cid]
        if cid in state.used_connections:
            return False
        if state.current_trip is not None:
            return state.current_trip == c.trip_id
        if not c.pickup_allowed:
            return False
        return state.last_connection is None or state.time >= state.last_arrival + self.mct

    def publish(self, state, belief):
        branches = [(Q(1), belief, ())]
        for slot, cid in self.locations[state.stop]:
            i, j = slot
            clocks = self.laws[i].clocks[j]
            if state.time not in clocks:
                continue
            occurred = 1 << clocks.index(state.time)
            next_branches = []
            for mass, posterior, visible in branches:
                for observed, mask in ((False, ~occurred), (True, occurred)):
                    p, updated = self.condition(posterior, slot, mask)
                    if p:
                        offers_ride = observed and cid is not None and self.can_board(state, cid)
                        offered = (*visible, cid) if offers_ride else visible
                        next_branches.append((mass * p, updated, offered))
            branches = next_branches
        assert sum(mass for mass, _, _ in branches) == 1
        return tuple((mass, replace(state, initialized=True, visible_connections=tuple(sorted(visible))),
                      posterior) for mass, posterior, visible in branches)

    def actions(self, state):
        if state.stop == self.goal or state.time >= self.cap:
            return ()
        if not state.initialized:
            return (OBSERVE,)
        alight = (ALIGHT,) if state.current_trip is not None else ()
        return (WAIT,) + alight + tuple(('board', cid) for cid in state.visible_connections)

    def _step(self, state, belief, action):
        if action == OBSERVE:
            return self.publish(state, belief)
        if action == ALIGHT:
            return self.publish(replace(state, current_trip=None, last_arrival=state.time), belief)
        if action == WAIT:
            staying = tuple(cid for cid in state.visible_connections
                            if self.graph.connections[cid].trip_id == state.current_trip)
            if staying:
                return self.step(state, belief, ('board', min(staying)))
            return self.publish(replace(state, time=state.time + 1), belief)

        cid = action[1]
        c = self.graph.connections[cid]
        i, j = slot = self.slots[cid][1]
        branches = []
        for index, arrival in enumerate(self.laws[i].clocks[j]):
            p, posterior = self.condition(belief, slot, 1 << index)
            if not p:
                continue
            time = min(self.cap, arrival)
            at = replace(state, stop=c.arr_stop if arrival <= self.cap else state.stop, time=time,
                         current_trip=c.trip_id, last_connection=cid, last_arrival=time,
                         local_since=time, used_connections=state.used_connections | {cid})
            branches.extend((p * prob, next_state, updated) for prob, next_state, updated in self.publish(at, posterior))
        assert sum(mass for mass, _, _ in branches) == 1
        return tuple(branches)

    def _value(self, state, belief, reveal_tail=False, fixed_menu=None):
        actions = self.actions(state)
        if not actions:
            return state.time if state.stop == self.goal else self.cap
        if reveal_tail and state.used_connections:
            return self.reveal(state, belief, range(len(self.laws)))
        if fixed_menu is not None and state.initialized:
            boards = tuple(a for a in actions if a[0] == 'board' and a[1] in fixed_menu)
            actions = (boards[0],) if boards else ((ALIGHT,) if ALIGHT in actions else (WAIT,))
        return min(sum(p * self.value(at, posterior, reveal_tail, fixed_menu)
                       for p, at, posterior in self.step(state, belief, a)) for a in actions)

    def reveal(self, state, belief, factors, reveal_tail=False):
        choices = [tuple((mass / self.mass(i, belief[i]), path)
                         for mass, path in self.laws[i].deterministic_masks(belief[i])) for i in factors]
        value = Q(0)
        for paths in product(*choices):
            full = list(belief)
            for i, (_, masks) in zip(factors, paths, strict=True):
                full[i] = masks
            value += prod(mass for mass, _ in paths) * self.value(state, tuple(full), reveal_tail, None)
        return value

    def scenes(self):
        """Small-test oracle only: expand the exact world product independently."""
        for combination in product(*(tuple(law.deterministic_masks(masks))
                                     for law, masks in zip(self.laws, self.prior, strict=True))):
            events, labels = {}, []
            for law, (mass, masks) in zip(self.laws, combination, strict=True):
                clocks = [times[bit.bit_length() - 1] for times, bit in zip(law.clocks, masks, strict=True)]
                labels.append(tuple(clocks))
                for position, cid in enumerate(law.legs):
                    dep, arr = clocks[2 * position:2 * position + 2]
                    events[cid] = VehicleEvent(0, False, dep, arr)
            yield prod(mass for mass, _ in combination), JourneyScenario(events, publications={}), tuple(labels)

    def independent_check(self, worlds, reveal_tail):
        graph_ = self.graph
        for stop in graph_.stops.values():
            stop.min_transfer_time = self.mct
        model = build_platform_model(graph_, tuple((float(p), w) for p, w in worlds), 0,
                                     self.goal, 0, self.cap, poll_interval=1)
        if not reveal_tail:
            return ExactPlanner(model).solve_compressed().value

        @lru_cache(None)
        def exact_tail(i, state):
            world = replace(model.scenarios[i], probability=1.)
            return ExactPlanner(replace(model, scenarios=(world,), initial_state=state)).solve_compressed().value

        return causal_prefix_lower(model, exact_tail, elapsed_horizon=self.cap).value


def one_leg(cid, dep, arr, prior=(1,), duration_rows=((1,),)):
    return TripLaw.create((cid,), (dep, arr), prior, (duration_rows,))


def alias_reference(third=False, remote_factors=0):
    timetable = [('A', 0, 1, 2, 3), ('B', 0, 1, 4, 10)]
    laws = [one_leg(0, (2, 3), (3, 4, 22, 23), (1, 1), ((1, 0, 1, 0), (0, 1, 0, 1))),
            one_leg(1, (4,), (10,))]
    if third:
        timetable.append(('R', 0, 1, 1, 4))
        laws.append(one_leg(2, (1,), (4, 18), duration_rows=((1, 1),)))
    for index in range(remote_factors):
        cid = len(timetable)
        timetable.append((f'remote_{index}', 10 + 2 * index, 11 + 2 * index, 1, 4))
        laws.append(one_leg(cid, (1,), (4, 18), duration_rows=((1, 1),)))
    return PublicBelief(timetable, laws)


def transfer_reference():
    timetable = [('A', 0, 1, 1, 2), ('A', 1, 2, 4, 6), ('B', 1, 2, 6, 10)]
    a = TripLaw.create((0, 1), ((1,), (2, 3), (4, 5), (6, 7, 12, 14)), (1,),
                       (((1, 1),), ((1, 0), (0, 1)), ((1, 0, 1, 0), (0, 1, 0, 1))))
    return PublicBelief(timetable, (a, one_leg(2, (6,), (10,))), goal=2, mct=1)


def grouped_worlds(reference, revealed):
    groups = defaultdict(list)
    for mass, scene, labels in reference.scenes():
        groups[tuple(labels[i] for i in revealed)].append((mass, scene))
    return groups


def finite_bank_values(reference, revealed, draws=4):
    """Enumerate four IID draws conditional on each deliberately revealed rest."""
    expected = Q(0)
    count = 0
    exact_r = exact_tau = 0.
    for worlds in grouped_worlds(reference, revealed).values():
        mass = sum(p for p, _ in worlds)
        population = tuple((p / mass, w) for p, w in worlds)
        exact_r += float(mass) * reference.independent_check(population, False)
        exact_tau += float(mass) * reference.independent_check(population, True)
        for atoms in combinations_with_replacement(range(len(worlds)), draws):
            weight = Q(factorial(draws), prod(factorial(n) for n in Counter(atoms).values()))
            weight *= prod(population[i][0] for i in atoms)
            bank = tuple((Q(1, draws), population[i][1]) for i in atoms)
            raw = reference.independent_check(bank, True) * draws
            total = round(raw)
            # An optimal deterministic public policy has an integer loss in
            # each equally weighted world. Undo the reference solver's float
            # roundoff before summing exact multinomial weights.
            assert isclose(total, raw, abs_tol=1e-10)
            expected += mass * weight * Q(total, draws)
            count += 1
    return expected, exact_r, exact_tau, count


def solve_fixture(reference, menu, revealed=()):
    started = perf_counter()
    state, prior = reference.initial, reference.prior
    causal = reference.value(state, prior, False, None)
    causal_nodes = reference.value.cache_info().currsize
    causal_factor_evaluations = reference.factor_work
    causal_seconds = perf_counter() - started
    j = reference.value(state, prior, False, frozenset(menu))
    v_r = reference.reveal(state, prior, revealed)
    v_tau = reference.reveal(state, prior, revealed, reveal_tail=True)
    started = perf_counter()
    exact_worlds = tuple((mass, scene) for mass, scene, _ in reference.scenes())
    causal_check = reference.independent_check(exact_worlds, False)
    z, check_r, check_tau, banks = finite_bank_values(reference, revealed)
    assert isclose(causal, causal_check) and isclose(v_r, check_r) and isclose(v_tau, check_tau)
    assert j >= causal >= v_r >= v_tau >= z
    terms = dict(A=j-causal, B=causal-v_r, C=v_r-v_tau, D=v_tau-z)
    assert sum(terms.values()) == j-z
    return dict(J=j, V_star=causal, V_R=v_r, V_R_tau=v_tau, ell_4=z, terms=terms,
                causal_work=dict(nodes=causal_nodes, factor_mass_evaluations=causal_factor_evaluations,
                                 full_world_rows=0, seconds=causal_seconds),
                reference_check=dict(worlds=len(exact_worlds), empirical_bank_multisets=banks,
                                     seconds=perf_counter()-started))


def observed_branch(reference, stop, time, cid, belief=None):
    state = PlatformState(stop, time)
    posterior = reference.prior if belief is None else belief
    rows = [(p, evidence, cid in observed.visible_connections)
            for p, observed, evidence in reference.publish(state, posterior)]
    result = {seen: (p, evidence) for p, evidence, seen in rows}
    assert len(result) == len(rows)
    return result


def remote_factor_growth():
    rows = []
    for count in (0, 4, 12, 24):
        reference = alias_reference(remote_factors=count)
        start = perf_counter()
        value = reference.value(reference.initial, reference.prior, False, None)
        assert value == 10
        rows.append(dict(unreachable_independent_factors=count, value=value,
                         cartesian_joint_support_rows=4 * 2**count, expanded_joint_rows=0,
                         causal_nodes=reference.value.cache_info().currsize,
                         factor_mass_evaluations=reference.factor_work, seconds=perf_counter()-start))
    assert {row['causal_nodes'] for row in rows} == {rows[0]['causal_nodes']}
    assert {row['factor_mass_evaluations'] for row in rows} == {rows[0]['factor_mass_evaluations']}
    return rows


def unannounced_service_filter():
    # Clock 31 is a never-in-window sentinel. With no service message, delayed
    # beyond the deadline and canceled service have identical public evidence.
    timetable = [('A', 0, 1, 1, 2), ('A', 1, 2, 4, 6), ('B', 1, 2, 5, 10)]
    a = TripLaw.create((0, 1), ((1,), (2,), (4, 31), (6, 31)), (1,),
                       (((1,),), ((1, 1),), ((1, 0), (0, 1))))
    reference = PublicBelief(timetable, (a, one_leg(2, (5,), (10,))), goal=2, mct=1)
    later_departure, later_arrival = reference.slots[1]
    at_four = observed_branch(reference, 1, 4, 1)
    prior = reference.marginal(reference.prior, later_departure)
    missed = reference.marginal(at_four[False][1], later_arrival)
    assert prior == {4: Q(1, 2), 31: Q(1, 2)}
    assert missed == {31: Q(1)}
    causal = reference.value(reference.initial, reference.prior, False, None)
    rows = [(mass, scene) for mass, scene, _ in reference.scenes()]
    assert causal == reference.independent_check(rows, False) == 8
    return dict(deadline=30, out_of_window=31, departure_prior=prior,
                arrival_after_no_departure_at_4=missed, causal_value=causal)


def main():
    alias = alias_reference()
    a = solve_fixture(alias, {1})
    assert (a['J'], a['V_star'], a['ell_4']) == (10, 10, Q(2183, 256))
    r = alias_reference(third=True)
    b = solve_fixture(r, {0, 1}, (2,))
    assert (b['J'], b['V_star'], b['V_R']) == (13, 10, 7)
    assert b['terms']['D'] > 0
    transfer = transfer_reference()
    c = solve_fixture(transfer, {0, 1, 2})
    assert (c['V_star'], c['V_R_tau'], c['ell_4']) == (Q(19, 2), Q(33, 4), Q(33, 4))

    local = observed_branch(r, 0, 2, 0)
    assert set(local) == {False, True}
    suffix = r.slots[0][1]
    remote = r.slots[2][1]
    arrive_seen = r.marginal(local[True][1], suffix)
    arrive_absent = r.marginal(local[False][1], suffix)
    assert arrive_seen == {3: Q(1, 2), 22: Q(1, 2)}
    assert arrive_absent == {4: Q(1, 2), 23: Q(1, 2)}
    remote_dist = r.marginal(r.prior, remote)
    assert all(r.marginal(evidence, remote) == remote_dist for _, evidence in local.values())

    first_arrival = transfer.slots[0][1]
    future_departure, future_arrival = transfer.slots[1]
    _, after_ride = transfer.condition(transfer.prior, first_arrival, 1)
    _, after_later_ride = transfer.condition(transfer.prior, first_arrival, 2)
    assert transfer.marginal(after_ride, future_departure) == {4: Q(1)}
    assert transfer.marginal(after_ride, future_arrival) == {6: Q(1, 2), 12: Q(1, 2)}
    assert transfer.marginal(after_later_ride, future_departure) == {5: Q(1)}
    assert transfer.marginal(after_later_ride, future_arrival) == {7: Q(1, 2), 14: Q(1, 2)}
    phase_labels = []
    for tick, posterior in ((2, after_ride), (3, after_later_ride)):
        public = PlatformState(1, 3, 'A', tick, 0, frozenset({0}), initialized=True, local_since=tick)
        phase_labels.append(transfer.value(public, posterior, False, None) - public.time)
    assert phase_labels == [6, 7]

    out = dict(scope='Exact declared small laws. Causal solve branches public events on original '
                     'trip kernels; relaxed comparators deliberately reveal factors or suffixes.',
               alias=a, third_vehicle=b, transfer_tail=c,
               remote_factor_growth=remote_factor_growth(),
               unannounced_service_filter=unannounced_service_filter(),
               local_event_filter=dict(tick=2, departure_seen_probability=local[True][0],
                                       A_arrival_if_seen=arrive_seen, A_arrival_if_absent=arrive_absent,
                                       R_arrival_before_and_after=remote_dist),
               same_trip_filter_after_first_ride=[dict(observed_arrival=tick,
                    optimal_remaining_loss_at_common_clock_3=loss,
                    conditional_next_departure=transfer.marginal(posterior, future_departure),
                    conditional_next_arrival=transfer.marginal(posterior, future_arrival))
                    for tick, posterior, loss in ((2, after_ride, phase_labels[0]),
                                                  (3, after_later_ride, phase_labels[1]))])
    output = Path(__file__).with_name('generative_reference.json')
    output.write_text(json.dumps(out, indent=2, default=float) + '\n')
    print(output)


if __name__ == '__main__':
    main()
