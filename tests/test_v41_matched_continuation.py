from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v41.matched_continuation import (
    FrozenCombinedReference, MatchedContinuationPlanner,
)
from src.evolving_trip_belief import _ChainLaw
from src.full_menu_opportunity import FullMenuOpportunity
from src.full_menu_waiting_dp import FullMenuWaitingDP
from src.plan_waiting_controls import CombinedWaitingOpportunity, SingleTargetOpportunity
from src.transit_graph import Connection, Stop, TransitGraph


class Compiler:
    def __init__(self, rows):
        self.rows = rows

    def departure_distribution(self, cid):
        times, mass, cancel, _ = self.rows[cid]
        return np.array(times), np.array(mass), cancel

    def clock_nominal_score(self, cid, clock):
        times, _, _, marks = self.rows[cid]
        return float(np.interp(clock, times, marks))


def enumerate_combined(reference):
    """Independent path enumeration; the controller receives only event prefixes."""
    control = CombinedWaitingOpportunity(SingleTargetOpportunity(reference))
    atoms = []
    for cid in reference.cids:
        times, mass, cancel = reference.distributions[cid]
        atoms.append([(float(t), float(p)) for t, p in zip(times, mass) if p > 0]
                     + ([(None, cancel)] if cancel else []))
    total, probability = 0., 0.
    for world in product(*atoms):
        weight = float(np.prod([p for _, p in world]))
        events = {}
        for cid, (clock, _) in zip(reference.cids, world):
            if clock is not None and reference.entry <= clock < reference.deadline:
                events.setdefault(clock, []).append(cid)
        seen, loss = set(), reference.deadline
        for clock, visible in sorted(events.items()):
            seen.update(visible)
            if clock < reference.ready:
                continue
            chosen, _ = control.choose(tuple(visible), seen, clock)
            if chosen is not None:
                loss = reference.score(chosen, clock)
                break
        total += weight*loss
        probability += weight
    assert probability == pytest.approx(1.)
    return total


@pytest.mark.parametrize('entry,ready', [(0., 0.), (5., 15.), (5., 20.5), (40., 50.), (101., 110.)])
def test_exact_combined_matches_public_prefix_enumeration_with_ties_absence_and_mct(entry, ready):
    rows = {0: ([0., 20., 40.], [.2, .5, .2], .1, [85., 70., 60.]),
            1: ([10., 20., 50.], [.25, .5, .25], 0., [80., 35., 70.]),
            2: ([20., 30.], [.4, .3], .3, [45., 55.])}
    reference = FullMenuOpportunity(Compiler(rows), {0: (0, 0), 1: (1, 1), 2: (2, 2)},
                                    entry, ready, 100.)
    exact = FrozenCombinedReference(reference)
    assert exact.value == pytest.approx(enumerate_combined(reference), abs=1e-10)
    assert 0 <= exact.value <= 100


def test_refusal_value_is_policy_evaluation_not_minimum_of_integrated_commitments():
    # At 10, an early expensive A is refused in favour of B. If A has not
    # appeared by B's offer at 20, B is refused in favour of late cheap A.
    rows = {0: ([10., 30.], [.5, .5], 0., [90., 40.]),
            1: ([20.], [1.], 0., [70.])}
    reference = FullMenuOpportunity(Compiler(rows), {0: (0, 0), 1: (1, 1)}, 0., 0., 100.)
    exact = FrozenCombinedReference(reference)
    assert exact.first_offer_value == pytest.approx(80.)
    assert exact.value == pytest.approx(55.)
    assert exact.value_after(10., (0,)) == pytest.approx(70.)
    assert exact.value_after(20., (1,)) == pytest.approx(40.)
    # The shallow envelope is 65, not 55: no expectation/min interchange.
    assert min(80., .5*90+.5*40, 70.) == 65.


def test_fixed_combined_is_not_replaced_by_optimal_waiting():
    rows = {0: ([10., 30.], [.5, .5], 0., [90., 40.]),
            1: ([20.], [1.], 0., [70.]),
            2: ([0.], [1.], 0., [60.])}
    reference = FullMenuOpportunity(Compiler(rows), {cid: (cid, cid) for cid in rows}, 0., 0., 100.)
    # Combined takes C at 60 because its wait reference is 65. An optimal
    # waiting solver values the two-stage A/B refusal policy at 55 instead.
    exact = FrozenCombinedReference(reference)
    optimal = FullMenuWaitingDP(reference)
    assert exact.value == pytest.approx(enumerate_combined(reference)) == 60.
    assert optimal.values[0, optimal.full_mask] == pytest.approx(55.)


class History:
    def __init__(self, laws, blocked=(), revision=0):
        self.laws, self.blocked, self.revision = laws, set(blocked), revision
        self._physical_bases = laws

    def law(self, trip):
        return self.laws[trip]

    def signature(self, relevant):
        return tuple((trip, self.revision, tuple(sorted(self.blocked))) for trip in sorted(relevant))

    def announced_unavailable(self, cid, kind):
        return (cid, kind) in self.blocked


def fixture():
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    graph.add_connection(Connection(0, 'R0', 'source', 0, 1, 0, 5))
    graph.add_connection(Connection(1, 'R1', 'A', 1, 2, 10, 90))
    graph.add_connection(Connection(2, 'R2', 'B', 1, 2, 20, 70))
    positions, laws = {}, {}
    for cid, trip, departures, arrivals, probability in (
        (0, 'source', [0.], [5.], [1.]),
        (1, 'A', [10., 30.], [90., 40.], [.5, .5]),
        (2, 'B', [20.], [70.], [1.]),
    ):
        times = [np.array(departures), np.array(arrivals)]
        # States need sorted clock support. Keep matching departure/arrival
        # states via the explicit permutation, not interpolated future truth.
        order = np.argsort(times[1])
        times[1] = times[1][order]
        matrix = np.eye(len(departures))[:, order]
        with np.errstate(divide='ignore'):
            transition = np.log(matrix)
        laws[trip] = _ChainLaw(times, np.log(probability), lambda slot, m=transition: m,
            [np.zeros(len(t)) for t in times], [np.zeros(len(t), bool) for t in times], (True,))
        positions[cid, 'departure'], positions[cid, 'arrival'] = (trip, 0), (trip, 1)
    platform = Platform(1, {cid: Ride(cid, (cid,), {cid: None}) for cid in (1, 2)})
    ride = Ride(0, (0,), {0: platform})
    return graph, SimpleNamespace(_positions=positions), History(laws), platform, ride


def test_shallow_planner_prices_target_wait_without_changing_child_policy_or_menu():
    graph, model, history, platform, ride = fixture()
    planner = MatchedContinuationPlanner(graph, model, {}, 100.)
    components = planner.selective_waiting_components(platform, history, np.array([5., 20.5, 31.]))
    assert components['first_offer'] == pytest.approx([80., 70., 100.])
    assert components['single_target'] == pytest.approx([65., 70., 100.])
    assert components['value'] == pytest.approx([65., 70., 100.])
    assert set(components['by_target']) == set(platform.rides) == {1, 2}
    compiled = planner.compile_ride(ride, history, 'rolling_fixed')
    assert compiled.score(0.) == pytest.approx(65.)
    assert 'approximate' in compiled.conditional_scope
    exact = planner.frozen_combined_reference(platform, history, 5., 5.)
    assert exact.value == pytest.approx(55.)
    assert planner.tail_base.compile_ride(ride, history, 'rolling_fixed').score(0.) == pytest.approx(80.)
    assert planner.stats['tail_base_ride_compiles'] > 0


def test_tail_keeps_notice_masks_conditional_revision_and_terminal_deadline():
    graph, model, history, platform, ride = fixture()
    planner = MatchedContinuationPlanner(graph, model, {}, 100.)
    first = planner.compile_ride(ride, history, 'rolling_fixed')
    assert planner.compile_ride(ride, history, 'rolling_fixed') is first
    blocked_departure = History(history.laws, {(2, 'departure')})
    assert planner.compile_ride(ride, blocked_departure, 'rolling_fixed').score(0) == pytest.approx(65.)
    blocked_arrival = History(history.laws, {(0, 'arrival')})
    assert planner.compile_ride(ride, blocked_arrival, 'rolling_fixed').score(0) == pytest.approx(100.)
    assert planner.compile_ride(ride, History(history.laws, revision=1), 'rolling_fixed') is not first
