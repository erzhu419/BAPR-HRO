from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v34.forecast import ForecastPlanner
from experiments.reconstruction_v38.planner import FastPlanner, rolling_fixed_policy
from src.evolving_trip_belief import _ChainLaw
from src.onboard_stopping import OnboardStopping
from src.transit_graph import Connection, Stop, TransitGraph


class History:
    def __init__(self, laws, revisions=None, blocked=()):
        self.laws, self.revisions, self.blocked = laws, revisions or {}, set(blocked)

    def law(self, trip):
        return self.laws[trip]

    def signature(self, relevant):
        return tuple((trip, self.revisions.get(trip, 0), tuple(sorted(self.blocked))) for trip in sorted(relevant))

    def announced_unavailable(self, cid, kind):
        return (cid, kind) in self.blocked


def make_law(times, matrices, initial, canceled=None, emissions=None):
    times = [np.asarray(t, dtype=float) for t in times]
    with np.errstate(divide='ignore'):
        matrices = [np.log(np.asarray(matrix, dtype=float)) for matrix in matrices]
        initial = np.log(initial)
    return _ChainLaw(times, initial, lambda slot: matrices[slot],
        [np.zeros(len(t)) for t in times] if emissions is None else emissions,
        [np.zeros(len(t), bool) for t in times] if canceled is None else canceled,
        (True,)*len(matrices))


def terminal_fixture():
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    positions = {}
    for cid in range(3):
        graph.add_connection(Connection(cid, 'R', 'A', cid, cid+1, 240*cid, 240*cid+120))
        positions[cid, 'departure'], positions[cid, 'arrival'] = ('A', 2*cid), ('A', 2*cid+1)
    times = [120*slot+np.array([0, 60, 0, 60]) for slot in range(6)]
    clock = np.array([[.85, .15], [.25, .75]])
    service = np.array([[.8, .2], [.35, .65]])
    matrices = [np.kron(np.eye(2) if slot % 2 == 0 else service, clock) for slot in range(5)]
    law = make_law(times, matrices, np.array([.56, .24, .14, .06]),
                   canceled=[np.array([False, False, True, True]) for _ in times])
    model = SimpleNamespace(_positions=positions)
    return graph, model, History({'A': law}), Ride(0, (0, 1, 2), {2: None}), clock


def test_rolling_only_keeps_commitment_decisions_and_evaluates_the_executable_policy():
    chain = OnboardStopping(([0, 1], [10, 11], [20, 21]),
        ([60., 50.], [25., 80.], [70., 20.]),
        (np.array([[.3, .4], [.5, .2]]), np.array([[.5, .1], [.2, .6]])), 100.)
    initial = np.array([.35, .4])
    expected, actual = chain.solve(initial).policies['rolling_fixed'], rolling_fixed_policy(chain, initial)
    assert actual.values[0] == pytest.approx([60., 50.])
    assert actual.values[1] == pytest.approx([25., 46.])
    assert actual.root_value == pytest.approx(66.)
    assert chain.solve(initial).policies['optimal'].values[0][0] == pytest.approx(55.9)
    for left, right in zip(actual.values, expected.values):
        assert np.array_equal(left, right)
    for left, right in zip(actual.actions, expected.actions):
        assert np.array_equal(left, right)
    for left, right in zip(actual.decision_costs, expected.decision_costs):
        assert np.array_equal(left, right)


@pytest.mark.parametrize('method', ['nominal', 'rolling_fixed', 'one_step_rollout', 'optimal'])
def test_terminal_contraction_matches_the_original_and_kills_intermediate_cancellation(method):
    graph, model, history, ride, clock = terminal_fixture()
    old = ForecastPlanner(graph, model, {}, 1000).compile_ride(ride, history, method)
    planner = FastPlanner(graph, model, {}, 1000)
    new = planner.compile_ride(ride, history, method)
    expected = 1000-.8**2*(np.linalg.matrix_power(clock, 5) @ np.array([400., 340.]))
    assert new.board_scores == pytest.approx(expected, abs=1e-10)
    assert new.board_scores == pytest.approx(old.board_scores, abs=1e-10)
    assert new.policy.root_value == pytest.approx(old.policy.root_value, abs=1e-10)
    assert new.score(30) == pytest.approx(old.score(30), abs=1e-10)
    # Recovery after an intermediate cancellation must not restore that path.
    assert new.score(0) > 1000-(.8**2+.2*.35)*(np.linalg.matrix_power(clock, 5) @ np.array([400., 340.]))[0]
    assert planner.stats['kernel_compiles'] == 0
    assert planner.stats['terminal_vector_compiles'] == 1 and planner.stats['terminal_vector_steps'] == 5


def test_terminal_mask_and_relevant_history_cache_have_the_same_scope():
    graph, model, history, ride, _ = terminal_fixture()
    planner = FastPlanner(graph, model, {}, 1000)
    old = planner.compile_ride(ride, history, 'rolling_fixed')
    unrelated = History(history.laws, revisions={'outside': 1})
    assert planner.compile_ride(ride, unrelated, 'rolling_fixed') is old
    blocked = History(history.laws, blocked={(2, 'arrival')})
    assert planner.compile_ride(ride, blocked, 'rolling_fixed').board_scores == pytest.approx([1000., 1000.])
    assert planner.compile_ride(ride, history, 'rolling_fixed') is old


def test_future_forecast_conditioning_remains_inside_the_terminal_edges():
    graph, model, history, ride, _ = terminal_fixture()
    prior = history.law('A')
    emissions = list(prior.emissions)
    emissions[-1] = -.5*((prior.times[-1]-660)/20.)**2
    history.laws['A'] = _ChainLaw(prior.times, prior.initial, prior._transition, emissions, prior.canceled, prior.stochastic)
    old = ForecastPlanner(graph, model, {}, 1000).compile_ride(ride, history, 'rolling_fixed')
    new = FastPlanner(graph, model, {}, 1000).compile_ride(ride, history, 'rolling_fixed')
    assert new.board_scores == pytest.approx(old.board_scores, abs=1e-10)
    assert new.policy.root_value == pytest.approx(old.policy.root_value, abs=1e-10)


def platform_fixture():
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'start', min_transfer_time=0))
    graph.add_stop(Stop(1, 'goal', min_transfer_time=0))
    graph.add_connection(Connection(0, 'R', 'A', 0, 1, 0, 10))
    graph.add_connection(Connection(1, 'R', 'B', 0, 1, 20, 35))
    model = SimpleNamespace(_positions={(0, 'departure'): ('A', 0), (0, 'arrival'): ('A', 1),
                                        (1, 'departure'): ('B', 0), (1, 'arrival'): ('B', 1)})
    history = History({'A': make_law([[0, 20, 40], [10, 30, 50]], [np.eye(3)], [.2, .5, .3]),
                       'B': make_law([[20, 40], [35, 55]], [np.eye(2)], [.6, .4])})
    platform = Platform(0, {0: Ride(0, (0,), {0: None}), 1: Ride(1, (1,), {1: None})})
    return graph, model, history, platform


def test_batched_platform_scores_keep_mct_ties_past_mass_and_offgrid_public_clocks():
    graph, model, history, platform = platform_fixture()
    old, new = ForecastPlanner(graph, model, {}, 100), FastPlanner(graph, model, {}, 100)
    ready = np.array([[0., 5., 20.], [20.5, 40., 41.]])
    assert new.first_offer_values(platform, history, ready, 'rolling_fixed') == pytest.approx(
        old.first_offer_values(platform, history, ready, 'rolling_fixed'))
    a, b = old.platform_reference(platform, history, 5, 15, 'rolling_fixed'), new.platform_reference(platform, history, 5, 15, 'rolling_fixed')
    for cid in platform.rides:
        assert b.score(cid, 25.) == pytest.approx(a.score(cid, 25.))
        assert b.absent[cid] == pytest.approx(a.absent[cid])
        for left, right in zip(b.marks[cid], a.marks[cid]):
            assert np.array_equal(left, right)
    for visible, seen, clock in (((0, 1), (0, 1), 20), ((0,), (0,), 25), ((1,), (0, 1), 40)):
        assert b.choose(visible, seen, clock) == a.choose(visible, seen, clock)
    with pytest.raises(UnsupportedHistory):
        b.compiler.clock_nominal_scores(0, np.array([-1., 25.]))
    assert b.compiler.clock_nominal_scores(0, np.array([])).shape == (0,)
