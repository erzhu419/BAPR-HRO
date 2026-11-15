"""Terminal subset sums retain the five causal targets and original budgets."""
import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v52.reference import CausalReference
from experiments.reconstruction_v53.reference import CompressedReference
from experiments.reconstruction_v55.reference import StructuralReference
from experiments.reconstruction_v55.terminal import terminal_platform, _physical_marks, _slots
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from src import evolving_trip_belief as evolving
from src.transit_graph import TransitGraph, Stop, Connection
from test_v52_reference import frozen_chain, information_case
from test_v53_chain import case as chain_case, enumerate_value


def host(base, *, history=None, limit=None):
    result = StructuralReference(base.graph, base.model, base.parameters, base.deadline,
        base.snapshot if history is None else history, base.observations,
        max_states=limit, max_process_cpu=None)
    result.root_law.roots.update(base.root_law.roots)
    return result


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('mct', [0, 11, 21])
def test_full_terminal_value_matches_chronological_all_modes(mode, mct):
    base, platform, _ = information_case(observed=False, mct=mct)
    new = host(base)
    row = dict(kind='platform_start', clock=0., entry=0.)
    expected = base.solve(platform, row, mode=mode)
    actual = new.solve(platform, row, mode=mode)
    assert actual['value']['complete']
    assert actual['value']['mean'] == pytest.approx(expected['value']['mean'], abs=1e-8)
    assert actual['counters']['terminal_platform_calls'] == 1


@pytest.mark.parametrize('mode', CausalReference.MODES)
def test_root_wait_removes_seen_offer_and_keeps_strict_after(mode):
    base, platform, _ = information_case(observed=False)
    history = base.snapshot.clone()
    history.observe_platform((0, 2), 0., 10., {0:10.})
    reference = host(base, history=history)
    row = dict(kind='platform', clock=10., entry=0., visible=[0], observed=[0])
    actual = reference.solve(platform, row, mode=mode)
    assert actual['actions']['wait']['complete']
    assert actual['actions']['wait']['mean'] == pytest.approx(30.)
    assert actual['actions']['0']['mean'] == pytest.approx(20.)
    assert actual['counters']['terminal_singleton_integrations'] == 1


@pytest.mark.parametrize('mode', CausalReference.MODES)
@pytest.mark.parametrize('limit', [0, 1, 2, 4, 8, 12])
def test_budget_bound_contains_the_same_policy_without_extra_allowance(mode, limit):
    base, platform, _ = information_case(observed=False)
    expected = base.solve(platform, dict(kind='platform_start', clock=0., entry=0.), mode=mode)
    new = host(base, limit=limit)
    actual = new.solve(platform, dict(kind='platform_start', clock=0., entry=0.), mode=mode)
    value = actual['value']
    assert value['lower']-1e-8 <= expected['value']['mean'] <= value['upper']+1e-8
    assert actual['counters']['states'] <= limit
    if not value['complete']:
        assert value['mean'] is None


def test_related_nonaction_observation_rejects_subset_compression():
    base, platform, history = information_case(observed=True)
    new = host(base)
    assert terminal_platform(new, platform, 0., history, -1) is None
    actual = new.solve(platform, dict(kind='platform_start', clock=0., entry=0.))
    assert actual['value']['complete']
    assert actual['value']['mean'] == pytest.approx(27.5)


def test_repeated_trip_and_nonterminal_choice_are_rejected():
    base, platform, history = information_case(observed=False)
    repeated = Platform(0, {1:Ride(1, (1,), {1:None}), 2:platform.rides[2]})
    assert terminal_platform(host(base), repeated, 0., history, -1) is None
    platform.rides[0].exits[0] = Platform(1, {})
    assert terminal_platform(host(base), platform, 0., history, -1) is None


@pytest.mark.parametrize('options', [{}, {'canceled':True}, {'hidden':True}, {'deadline':51.}])
def test_batch_terminal_marks_keep_hidden_slots_future_emissions_and_failures(options):
    base, history, ride = chain_case(**options)
    times, mass, cancel, marks = _physical_marks(base, ride, history, _slots(base.model, ride))
    actual = base.deadline + mass @ (marks-base.deadline)
    assert actual == pytest.approx(enumerate_value(base, history, ride, 0, 0.), abs=1e-9)


@pytest.mark.parametrize('mode', CausalReference.MODES)
def test_singleton_integrates_long_support_past_cancel_and_ready(mode):
    base, platform, _ = information_case(observed=False, mct=11)
    platform.rides.pop(2)
    base.observations = ObservationContract({0:(0,)})
    # Includes pre-entry, pre-ready, legal, at-deadline and canceled mass.
    law = frozen_chain([[-5, 10, 20, 60, 25], [0, 20, 35, 65, 40]], [.1, .2, .3, .15, .25])
    law.canceled[0][-1] = True
    base.root_law.roots['A'] = law
    new = host(base)
    new.mode = mode
    actual = terminal_platform(new, platform, 0., new.snapshot, 2)
    assert actual == pytest.approx((52.5, 52.5, True))
    assert new.stats.get('terminal_platform_board_scores', 0) == 0


@pytest.mark.parametrize('mode', CausalReference.MODES)
def test_simultaneous_offers_use_original_score_and_ranks(mode):
    base, platform, _ = information_case(observed=False)
    base.root_law.roots['A'] = frozen_chain([[10, 20], [30, 35]], [.5, .5])
    base.root_law.roots['B'] = frozen_chain([[5, 15], [6, 16], [10, 20], [25, 55]], [.5, .5])
    row = dict(kind='platform_start', clock=0., entry=0.)
    expected = base.solve(platform, row, mode=mode)
    actual = host(base).solve(platform, row, mode=mode)
    assert actual['value']['complete']
    assert actual['value']['mean'] == pytest.approx(expected['value']['mean'], abs=1e-8)


@pytest.mark.parametrize('mode', CausalReference.MODES)
def test_three_candidates_keep_forecast_reselection_and_finite_depth(monkeypatch, mode):
    monkeypatch.setattr(evolving, 'DELAYS', np.array([0., 8/60]))
    graph = TransitGraph()
    for stop in (0, 1):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for cid, trip, departure, arrival in ((0, 'A', 10, 35), (1, 'B', 15, 25), (2, 'C', 19, 30)):
        graph.add_connection(Connection(cid, 'R', trip, 0, 1, departure, arrival))
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False, travel_sd=.1, dwell_sd=.1)
    parameters = {'R':dict(mean=0., std=.1, cancel_rate=.05)}
    fit = dict(route_kind={}, pooled={kind:dict(bias_seconds=0., sd_seconds=5.)
                                    for kind in ('departure', 'arrival')})
    history = PhysicalHistory(model, parameters, fit=fit)
    history.apply_frame(dict(receipt_seconds=1., trip_statuses=[], endpoint_updates=[
        [0, 'arrival', 'FORECAST', 43.], [1, 'arrival', 'FORECAST', 33.],
        [2, 'arrival', 'FORECAST', 38.]]), 1.)
    history.apply_frame(dict(receipt_seconds=2., trip_statuses=[], endpoint_updates=[
        [0, 'departure', 'FORECAST', 10.], [1, 'departure', 'FORECAST', 15.],
        [2, 'departure', 'FORECAST', 19.]]), 2.)
    platform = Platform(0, {cid:Ride(cid, (cid,), {cid:None}) for cid in (0, 1, 2)})
    base = CausalReference(graph, model, parameters, 60., history,
        ObservationContract.from_menu(platform), max_states=None, max_process_cpu=None)
    assert history.selected_forecast('A')[1] == 'departure'
    observed = history.clone()
    observed.observe(0, 'departure', 10.)
    assert observed.selected_forecast('A')[1] == 'arrival'
    row = dict(kind='platform_start', clock=2., entry=2.)
    expected = base.solve(platform, row, mode=mode)
    actual = host(base).solve(platform, row, mode=mode)
    assert expected['value']['complete'] and actual['value']['complete']
    assert actual['value']['mean'] == pytest.approx(expected['value']['mean'], abs=1e-8)
    if mode != 'optimal':
        assert actual['counters']['terminal_platform_board_scores'] > 0
