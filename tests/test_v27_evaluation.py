from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v26.run import replay_world
from experiments.reconstruction_v27.evaluate import integrate
from experiments.reconstruction_v27.run import assert_reproduction
from src.full_menu_opportunity import FullMenuOpportunity
from src.full_menu_waiting_dp import FullMenuWaitingDP
from src.plan_waiting_controls import (
    CombinedWaitingOpportunity, SchedulePriority, SingleTargetOpportunity,
)
from src.transfer_opportunity import RouteOption
from src.transit_graph import Connection, Stop, TransitGraph


class Scores:
    def __init__(self, laws, values):
        self.laws, self.values = laws, values

    def departure_distribution(self, cid):
        clocks, mass, canceled = self.laws[cid]
        return np.array(clocks), np.array(mass), canceled

    def clock_nominal_score(self, cid, clock):
        return self.values[cid]


def prepare(laws, values, planned, entry, ready, deadline):
    reference = FullMenuOpportunity(Scores(laws, values), {cid: (cid, cid) for cid in laws},
                                     entry, ready, deadline)
    single = SingleTargetOpportunity(reference)
    controls = dict(schedule_priority=SchedulePriority(planned), single_target_nominal=single,
                    combined_wait_nominal=CombinedWaitingOpportunity(single))
    return reference, controls


def enumerate_values(laws, reference, controls):
    """Whole-world evaluation dispatches only observed past and visible offers."""
    cids = tuple(laws)
    atoms = [tuple(zip(times, mass))+((None, canceled),)
             for times, mass, canceled in laws.values()]
    totals, total_mass = dict.fromkeys(controls, 0.), 0.
    for draw in product(*atoms):
        probability = float(np.prod([p for _, p in draw]))
        if not probability:
            continue
        world = dict(zip(cids, (time for time, _ in draw)))
        legal = sorted({t for t in world.values()
                        if t is not None and reference.ready <= t < reference.deadline})
        for method, control in controls.items():
            value = reference.deadline
            for clock in legal:
                visible = tuple(cid for cid in cids if world[cid] == clock)
                seen = tuple(cid for cid in cids if world[cid] is not None
                             and reference.entry <= world[cid] <= clock)
                chosen, _ = control.choose(visible, seen, clock)
                if chosen is not None:
                    value = reference.score(chosen, clock)
                    break
            totals[method] += probability*value
        total_mass += probability
    assert total_mass == pytest.approx(1.)
    return totals


@pytest.mark.parametrize('laws,values,planned,entry,ready,deadline', [
    ({0: ([40], [1.], 0.), 1: ([50], [1.], 0.),
      2: ([60], [1.], 0.), 3: ([60], [1.], 0.)},
     {0: 110., 1: 100., 2: 140., 3: 80.}, {0: 130, 1: 125, 2: 90, 3: 100}, 0, 30, 200),
    ({0: ([0, 10, 20], [.2, .4, .3], .1),
      1: ([8, 20, 35], [.2, .5, .2], .1),
      2: ([20, 30, 70], [.5, .3, .1], .1)},
     {0: 40., 1: 45., 2: 40.}, {0: 35, 1: 30, 2: 40}, 5, 15, 50),
])
def test_new_policy_integration_matches_independent_world_enumeration(
        laws, values, planned, entry, ready, deadline):
    reference, controls = prepare(laws, values, planned, entry, ready, deadline)
    # Only a small oracle fixture is constructed here; production reads V26's table.
    oracle = FullMenuWaitingDP(reference)
    archive = dict(times=oracle.times, supported=oracle.supported, dp=oracle.values)
    expected = enumerate_values(laws, reference, controls)
    result, tables = integrate(reference, controls, archive)
    assert result['expected_arrival'] == pytest.approx(expected)
    assert all(np.all(np.isfinite(table[oracle.supported])) for table in tables.values())
    assert all(excess <= 1e-10 for excess in result['dp_max_excess_over_control'].values())


def physical_fixture():
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'origin'))
    graph.add_stop(Stop(1, 'goal'))
    departures, arrivals = [40, 50, 60, 60], [120, 90, 100, 150]
    for cid, (dep, arr) in enumerate(zip(departures, arrivals, strict=True)):
        graph.add_connection(Connection(cid, str(cid), str(cid), 0, 1, dep, arr))
    options = {cid: RouteOption((cid,), {}) for cid in range(4)}
    events = {cid: SimpleNamespace(departure=dep, arrival=arr, canceled=False)
              for cid, (dep, arr) in enumerate(zip(departures, arrivals, strict=True))}
    laws = {cid: ([dep], [1.], 0.) for cid, dep in enumerate(departures)}
    reference, controls = prepare(laws, {0: 110., 1: 100., 2: 140., 3: 80.},
                                  {0: 130, 1: 125, 2: 90, 3: 100}, 0, 30, 200)
    return graph, options, events, reference, controls


def test_new_sample_replay_keeps_successive_refusals_and_physical_cost_separate():
    graph, options, events, reference, controls = physical_fixture()
    result = replay_world(graph, events, dict(start=0, deadline=200), options, reference.ranks,
                          controls)['outcomes']
    assert result['schedule_priority']['action'] == 2
    assert result['schedule_priority']['loss'] == 100
    for method in ('single_target_nominal', 'combined_wait_nominal'):
        assert [step['choice'] for step in result[method]['trace']] == [None, None, 3]
        assert result[method]['loss'] == 150
        assert reference.score(3, 60) == 80  # Its proxy advantage is not an actual loss guarantee.


def test_reconstruction_check_catches_changed_continuation_with_same_first_action():
    graph, options, events, reference, _ = physical_fixture()
    old = replay_world(graph, events, dict(start=0, deadline=200), options, reference.ranks,
                       dict(full_first_offer=None))['outcomes']['full_first_offer']
    changed = dict(events)
    changed[0] = SimpleNamespace(departure=40, arrival=130, canceled=False)
    new = replay_world(graph, changed, dict(start=0, deadline=200), options, reference.ranks,
                       dict(full_first_offer=None))['outcomes']['full_first_offer']
    assert old['action'] == new['action'] == 0
    assert_reproduction(old, old)
    with pytest.raises(AssertionError, match='Reconstructed V26'):
        assert_reproduction(new, old)
