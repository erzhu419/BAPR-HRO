from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v26.evaluate import METHODS, integrate
from experiments.reconstruction_v26.run import replay_world
from src.full_menu_opportunity import FullMenuOpportunity
from src.full_menu_waiting_dp import FullMenuWaitingDP
from src.transfer_opportunity import RouteOption
from src.transit_graph import Connection, Stop, TransitGraph


class Scores:
    def __init__(self, laws, values):
        self.laws, self.values = laws, values

    def departure_distribution(self, cid):
        times, mass, cancel = self.laws[cid]
        return np.array(times), np.array(mass, dtype=float), cancel

    def clock_nominal_score(self, cid, clock):
        return self.values[cid]


def controllers(laws, values, entry, ready, deadline):
    ranks = {cid: (cid, cid) for cid in laws}
    one = FullMenuOpportunity(Scores(laws, values), ranks, entry, ready, deadline)
    dp = FullMenuWaitingDP(one)
    return one, dp, dict(zip(METHODS, (None, one, dp)))


def enumerate_policy_values(laws, reference, controls):
    """Enumerate full worlds, exposing only each successive observed offer."""
    cids = tuple(laws)
    atoms = [tuple(zip(times, mass))+((None, cancel),) for times, mass, cancel in laws.values()]
    totals = dict.fromkeys(METHODS, 0.)
    total_mass = 0.
    for draw in product(*atoms):
        weight = float(np.prod([probability for _, probability in draw]))
        if weight == 0:
            continue
        world = dict(zip(cids, (clock for clock, _ in draw)))
        clocks = sorted({clock for clock in world.values()
                         if clock is not None and reference.ready <= clock < reference.deadline})
        for method, controller in controls.items():
            loss = reference.deadline
            for clock in clocks:
                visible = tuple(cid for cid in cids if world[cid] == clock)
                seen = tuple(cid for cid in cids if world[cid] is not None
                             and reference.entry <= world[cid] <= clock)
                chosen = (min(visible, key=reference.ranks.__getitem__) if controller is None else
                          controller.choose(visible, seen, clock)[0])
                if chosen is not None:
                    loss = reference.score(chosen, clock)
                    break
            totals[method] += weight*loss
        total_mass += weight
    assert total_mass == pytest.approx(1.)
    return totals


@pytest.mark.parametrize('laws,values,entry,ready,deadline', [
    ({0: ([10], [1.], 0.), 1: ([20], [1.], 0.),
      2: ([30], [1.], 0.), 3: ([30], [1.], 0.)},
     {0: 70., 1: 60., 2: 90., 3: 40.}, 0, 0, 100),
    ({0: ([0, 10, 20], [.2, .4, .3], .1),
      1: ([8, 20, 35], [.2, .5, .2], .1),
      2: ([20, 30, 70], [.5, .3, .1], .1)},
     {0: 40., 1: 45., 2: 40.}, 5, 15, 50),
])
def test_integration_matches_full_world_policy_enumeration(laws, values, entry, ready, deadline):
    one, dp, controls = controllers(laws, values, entry, ready, deadline)
    independent = enumerate_policy_values(laws, one, controls)
    result, tables = integrate(one, dp)
    assert result['expected_arrival'] == pytest.approx(independent)
    assert np.isfinite(result['dp_evaluation_max_error'])
    assert result['dp_evaluation_max_error'] < 1e-10
    assert np.array_equal(tables['supported'], dp.supported)


def test_integration_rejects_nan_in_a_supported_dp_state():
    one, dp, _ = controllers({0: ([10], [1.], 0.)}, {0: 30.}, 0, 0, 100)
    dp.values[0, dp.full_mask] = np.nan
    with pytest.raises(AssertionError, match='support'):
        integrate(one, dp)


def physical_inputs(departures, arrivals, deadline=200):
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'origin'))
    graph.add_stop(Stop(1, 'goal'))
    for cid, (departure, arrival) in enumerate(zip(departures, arrivals, strict=True)):
        graph.add_connection(Connection(cid, str(cid), str(cid), 0, 1, departure, arrival))
    options = {cid: RouteOption((cid,), {}) for cid in range(len(departures))}
    events = {cid: SimpleNamespace(departure=departure, arrival=arrival, canceled=False)
              for cid, (departure, arrival) in enumerate(zip(departures, arrivals, strict=True))}
    return graph, options, events, dict(start=0, deadline=deadline)


def test_sample_replay_reconsiders_after_two_refusals_and_keeps_physical_losses():
    laws = {0: ([40], [1.], 0.), 1: ([50], [1.], 0.),
            2: ([60], [1.], 0.), 3: ([60], [1.], 0.)}
    one, dp, controls = controllers(laws, {0: 110., 1: 100., 2: 140., 3: 80.}, 0, 30, 200)
    graph, options, events, root = physical_inputs([40, 50, 60, 60], [120, 90, 140, 150])
    result = replay_world(graph, events, root, options, one.ranks, controls)['outcomes']
    assert result['full_first_offer']['action'] == 0
    assert result['one_step_nominal']['action'] == 1
    assert result['multistep_nominal']['action'] == 3
    trace = result['multistep_nominal']['trace']
    assert [event['choice'] for event in trace] == [None, None, 3]
    assert [set(event['seen']) for event in trace] == [{0}, {0, 1}, {0, 1, 2, 3}]
    assert result['one_step_nominal']['loss'] == 90
    assert result['multistep_nominal']['loss'] == 150
    # Proxy optimality is not a guarantee about a sampled physical realization.
    assert dp.values[0, dp.full_mask] == 80


def test_sample_replay_observation_prefix_is_unchanged_by_future_departure():
    laws = {0: ([40], [1.], 0.), 1: ([50, 70], [.5, .5], 0.),
            2: ([60], [1.], 0.), 3: ([60], [1.], 0.),
            4: ([-5], [1.], 0.), 5: ([20], [1.], 0.)}
    one, _, controls = controllers(laws, {0: 110., 1: 100., 2: 140., 3: 80., 4: 20., 5: 30.},
                                   0, 30, 200)
    graph, options, events, root = physical_inputs([40, 50, 60, 60, -5, 20], [120, 90, 140, 150, 20, 30])
    later = dict(events)
    later[1] = SimpleNamespace(departure=70, arrival=100, canceled=False)
    a = replay_world(graph, events, root, options, one.ranks, controls)['outcomes']
    b = replay_world(graph, later, root, options, one.ranks, controls)['outcomes']
    for method in METHODS:
        assert a[method]['trace'][0] == b[method]['trace'][0]
        assert a[method]['trace'][0]['visible'] == (0,)
        assert set(a[method]['trace'][0]['seen']) == {0, 5}
