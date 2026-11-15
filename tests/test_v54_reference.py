"""Refinement preserves causal targets while spending one cumulative allowance."""
import pytest
import numpy as np

from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v52.reference import CausalReference, point
from experiments.reconstruction_v54.reference import RefiningReference
from test_v52_reference import information_case, frozen_chain
from test_v46_shared import shared_case
from experiments.reconstruction_v46.small_oracle import evaluate
from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from src import evolving_trip_belief as evolving
from src.transit_graph import Connection, Stop, TransitGraph


def refining(base, *, cutoff=.01, max_states=None, max_cpu=None):
    solver = RefiningReference(
        base.graph, base.model, base.parameters, base.deadline, base.snapshot,
        base.observations, max_states=max_states, max_process_cpu=max_cpu,
        impact_cutoff=cutoff, keep_decisions=True)
    solver.root_law.roots.update(base.root_law.roots)
    return solver


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('observed,mct', [(False, 0), (True, 0), (True, 11)])
def test_refinement_reaches_hand_enumerated_causal_values(mode, observed, mct):
    base, platform, _ = information_case(observed=observed, mct=mct)
    solver = refining(base, cutoff=30.)
    result = solver.solve(platform, dict(kind='platform_start', clock=0., entry=0.), mode=mode)
    # A arrives at 30; B's two worlds arrive at 25/55. The observation-only
    # B departure identifies its world before A's offer. Consistent scoring
    # waits for B only in the early world, whereas frozen W boards A.
    expected = 40. if mct else (27.5 if observed and mode != 'cheap_working' else 30.)
    assert result['value']['lower'] == pytest.approx(expected, abs=1e-8)
    assert result['value']['upper'] == pytest.approx(expected, abs=1e-8)
    if result['value']['complete']:
        assert result['value']['mean'] == pytest.approx(expected, abs=1e-8)
    else:
        # The policy value may be determined while an irrelevant alternative
        # remains partial. Do not relabel that traversal as complete.
        assert result['value']['mean'] is None
    assert result['future_public_frames'] == 0
    assert result['observation_contract'] == base.observations.describe()
    if observed:
        assert result['counters']['impact_pruned_branches'] > 0
        assert solver.impact_cutoff < 30.
    assert solver.reach == 1.
    assert all(record['clock'] >= mct for record in result['decisions'])


@pytest.mark.parametrize('reverse,cancel', [(False, 0.), (True, .05)])
def test_working_value_matches_independent_private_world_enumeration(monkeypatch, reverse, cancel):
    graph, model, parameters, history, platform, entry = shared_case(
        monkeypatch, reverse=reverse, forecasts=True, cancel=cancel)
    row = dict(kind='platform_start', clock=entry, entry=entry)
    expected = evaluate(graph, model, parameters, 60, platform, row, history,
                        action='policy', max_evaluations=100000)
    solver = RefiningReference(graph, model, parameters, 60, history,
        ObservationContract.from_menu(platform), max_states=None,
        max_process_cpu=None, impact_cutoff=30.)
    actual = solver.solve(platform, row, mode='cheap_working')
    assert expected['unresolved_probability'] == 0
    assert actual['value']['complete']
    assert actual['value']['mean'] == pytest.approx(expected['mean'], abs=1e-8)


def test_lower_cutoff_reopens_settled_partial_but_retains_exact_cache():
    base, platform, history = information_case(observed=True)
    solver = refining(base, cutoff=30.)
    solver._prepare(platform)
    first = solver._platform(platform, 0., history, depth=-1)
    assert not first[2] and first[0] <= 27.5 <= first[1]
    states = solver.stats['states']
    memo_before = dict(solver.memo)
    assert solver.settled_reach
    solver._set_cutoff(30.)
    repeated = solver._platform(platform, 0., history, depth=-1)
    assert repeated == first and solver.stats['states'] == states
    solver._set_cutoff(0.)
    assert solver.memo == memo_before  # interval evidence survives refinement
    assert not solver.settled_reach
    exact = solver._platform(platform, 0., history, depth=-1)
    assert exact == pytest.approx((27.5, 27.5, True))
    assert solver.stats['states'] > states
    states = solver.stats['states']
    assert solver._platform(platform, 0., history, depth=-1) == exact
    assert solver.stats['states'] == states


def test_budget_partial_can_resume_without_lowering_cutoff():
    base, platform, history = information_case(observed=True)
    solver = refining(base, cutoff=0., max_states=1000)
    solver._prepare(platform)
    solver._state_limit = 2
    first = solver._platform(platform, 0., history, depth=-1)
    assert not first[2] and solver.stats['budget_unknowns'] > 0
    states = solver.stats['states']
    solver._state_limit = 1000
    second = solver._platform(platform, 0., history, depth=-1)
    assert second == pytest.approx((27.5, 27.5, True))
    assert solver.stats['states'] > states
    assert solver.impact_cutoff == 0.


def work_arms(solver, *, cpu=None):
    """Eight resumable work units versus one; work is independent of scheduling."""
    pending = [8]
    calls = []

    def unit():
        if cpu is None:
            solver.stats['states'] += 1
        else:
            cpu[0] += 1.

    def slow():
        calls.append('slow')
        while pending[0] and solver._available():
            pending[0] -= 1
            unit()
        if pending[0]:
            return solver._unknown(0.)
        return point(20.)

    def fast():
        calls.append('fast')
        if not solver._available():
            return solver._unknown(0.)
        unit()
        return point(50.)

    return dict(slow=slow, fast=fast), pending, calls


def test_later_root_unused_state_share_returns_to_earlier_unfinished_action():
    base, _, _ = information_case()
    solver = refining(base, cutoff=0., max_states=10)
    solver.mode = 'cheap_p0'
    options, pending, calls = work_arms(solver)
    result = solver._solve_actions(options, 0., 'slow')
    assert calls[:2] == ['slow', 'fast'] and calls.count('slow') >= 2
    assert pending == [0]
    assert result['slow'] == point(20.) and result['fast'] == point(50.)
    assert solver.stats['states'] == 9
    assert solver.stats['states'] <= solver.max_states


def test_reflow_does_not_reset_cumulative_state_limit_when_work_remains():
    base, _, _ = information_case()
    solver = refining(base, cutoff=0., max_states=6)
    solver.mode = 'cheap_p0'
    options, pending, calls = work_arms(solver)
    result = solver._solve_actions(options, 0., 'slow')
    assert calls.count('slow') >= 2
    assert pending[0] > 0 and not result['slow'][2]
    assert solver.stats['states'] == 6
    assert result['slow'][0] <= 20. <= result['slow'][1]


def test_cpu_share_reflows_without_restarting_query_clock(monkeypatch):
    import experiments.reconstruction_v52.reference as v52
    import experiments.reconstruction_v53.reference as v53
    import experiments.reconstruction_v54.reference as v54
    cpu = [0.]
    for module in (v52, v53, v54):
        monkeypatch.setattr(module, 'process_time', lambda: cpu[0])
    base, _, _ = information_case()
    solver = refining(base, cutoff=0., max_cpu=10.)
    solver.mode = 'cheap_p0'
    began = solver.cpu_start
    options, pending, calls = work_arms(solver, cpu=cpu)
    result = solver._solve_actions(options, 0., 'slow')
    assert calls.count('slow') >= 2 and pending == [0]
    assert result['slow'] == point(20.)
    assert solver.cpu_start == began and cpu[0] - began == 9.


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('limit', [0, 2, 8])
def test_total_state_cap_contains_same_policy_value_during_refinement(mode, limit):
    base, platform, _ = information_case(observed=True)
    solver = refining(base, cutoff=30., max_states=limit)
    result = solver.solve(platform, dict(kind='platform_start', clock=0., entry=0.), mode=mode)
    expected = 30. if mode == 'cheap_working' else 27.5
    assert result['value']['lower'] - 1e-8 <= expected <= result['value']['upper'] + 1e-8
    assert result['counters']['states'] <= limit
    if not result['value']['complete']:
        assert result['value']['mean'] is None


@pytest.mark.parametrize('mode', CausalReference.MODES)
def test_root_wait_does_not_reopen_current_offer_after_refinement(mode):
    base, platform, _ = information_case(observed=False)
    base.snapshot.observe_platform((0, 2), 0., 10., {0:10.})
    solver = refining(base, cutoff=30.)
    row = dict(kind='platform', clock=10., entry=0., visible=[0], observed=[0])
    result = solver.solve(platform, row, mode=mode)
    assert result['actions']['0']['mean'] == pytest.approx(20.)
    wait = result['actions']['wait']
    assert wait['lower'] - 1e-8 <= 30. <= wait['upper'] + 1e-8
    if wait['complete']:
        assert wait['mean'] == pytest.approx(30.)
    assert result['selected_action'] == 0
    assert result['certified_action'] in (0, None)
    assert all(record['clock'] > 10. for record in result['decisions'])


def later_onboard_root(monkeypatch, *, cancel_next=False):
    """The root is H's second segment; its next segment reaches a useful exit."""
    monkeypatch.setattr(evolving, 'DELAYS', np.array([0.]))
    graph = TransitGraph()
    for stop in range(5):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for cid, trip, origin, target, dep, arr in (
            (0, 'H', 0, 1, 0, 5), (1, 'H', 1, 2, 6, 10),
            (2, 'H', 2, 3, 11, 20), (3, 'H', 3, 4, 21, 35),
            (4, 'A', 2, 4, 12, 40), (5, 'B', 3, 4, 22, 25)):
        graph.add_connection(Connection(cid, 'R', trip, origin, target, dep, arr))
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False,
                                        travel_sd=.1, dwell_sd=.1)
    parameters = {'R': dict(mean=0., std=.1, cancel_rate=0.)}
    fit = dict(route_kind={}, pooled={kind:dict(bias_seconds=0., sd_seconds=5.)
                                     for kind in ('departure', 'arrival')})
    history = PhysicalHistory(model, parameters, fit=fit)
    history.message_clock = 10.
    for cid, dep, arr in ((0, 0., 5.), (1, 6., 10.)):
        history.observe(cid, 'departure', dep)
        history.observe(cid, 'arrival', arr)
    first_exit = Platform(2, {4:Ride(4, (4,), {4:None})})
    next_exit = Platform(3, {5:Ride(5, (5,), {5:None})})
    root = Ride(0, (0, 1, 2, 3), {1:first_exit, 2:next_exit, 3:None})
    solver = RefiningReference(graph, model, parameters, 60., history,
        ObservationContract.from_menu(root), max_states=None,
        max_process_cpu=None, impact_cutoff=0., keep_decisions=True)
    law = frozen_chain([[0], [5], [6], [10], [11], [20], [21], [35]], [1.])
    law.canceled[4][0] = cancel_next
    solver.root_law.roots.update(H=law, A=frozen_chain([[12], [40]], [1.]),
                                B=frozen_chain([[22], [25]], [1.]))
    return solver, root, dict(kind='onboard', cid=1, clock=10.)


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('cancel_next', [False, True])
def test_onboard_continuation_keeps_next_segment_and_its_transfer(monkeypatch, mode, cancel_next):
    solver, root, row = later_onboard_root(monkeypatch, cancel_next=cancel_next)
    result = solver.solve(root, row, mode=mode)
    # Alight now reaches the goal at40. Continuing must execute segment2:
    # it reaches B at20 and B reaches the goal at25, or cancellation costs60.
    # Starting one segment too late would bypass the transfer/failure and
    # falsely report H's final arrival35. All receipts are relative to clock10.
    assert result['actions']['alight']['mean'] == pytest.approx(30.)
    assert result['actions']['continue']['mean'] == pytest.approx(50. if cancel_next else 15.)
    if not cancel_next:
        assert any(record.get('cid') == 2 for record in result['decisions'])
    if mode == 'cheap_working' and not cancel_next:
        independent = evaluate(solver.graph, solver.model, solver.parameters, 60.,
            root, row, solver.snapshot, action='continue', max_evaluations=1000)
        assert independent['unresolved_probability'] == 0
        assert result['actions']['continue']['mean'] == pytest.approx(independent['mean'])
