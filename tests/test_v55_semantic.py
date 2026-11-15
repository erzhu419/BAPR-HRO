"""Goal domination preserves the rolling scorer, not arbitrary heuristics."""
import numpy as np
import pytest

from experiments.reconstruction_v32.run import nominal_policy
from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v53.relevance import RemainingReadSets
from experiments.reconstruction_v55.semantic import BarrierPlanner, BarrierReadSets, GoalBarrierViews
from src import evolving_trip_belief as evolving
from src.onboard_stopping import OnboardStopping
from src.transit_graph import Connection, Stop, TransitGraph


def barrier_case(monkeypatch, *, cancel=0., forecast=None, masked=False):
    monkeypatch.setattr(evolving, 'DELAYS', np.array([0., 8/60]))
    graph = TransitGraph()
    for stop in range(4):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=0))
    for cid, trip, origin, target, dep, arr in (
            (0,'A',0,1,10,12), (1,'A',1,2,14,20), (2,'A',2,3,22,35),
            (3,'B',1,2,17,30), (4,'C',3,2,40,45), (5,'D',0,2,11,32)):
        graph.add_connection(Connection(cid,'R',trip,origin,target,dep,arr))
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False,
        travel_sd=.1, dwell_sd=.1, cancel_persistence=.8)
    params = {'R':dict(mean=0., std=.1, cancel_rate=cancel)}
    fit = dict(route_kind={}, pooled={kind:dict(bias_seconds=0., sd_seconds=5.)
                                      for kind in ('departure','arrival')})
    history = PhysicalHistory(model, params, fit=fit)
    updates = []
    if forecast is not None:
        updates.append([2, 'arrival', 'FORECAST', forecast])
    if masked:
        updates.append([1, 'arrival', 'SKIPPED', None])
    history.apply_frame(dict(receipt_seconds=1., endpoint_updates=updates), 1.)
    before = Platform(1, {3:Ride(3,(3,),{3:None})})
    after = Platform(3, {4:Ride(4,(4,),{4:None})})
    ride = Ride(0, (0,1,2), {0:before, 1:None, 2:after})
    platform = Platform(0, {0:ride, 5:Ride(5,(5,),{5:None})})
    views = GoalBarrierViews(model, history)
    baseline = MatchedContinuationPlanner(graph, model, params, 60.)
    barrier = BarrierPlanner(graph, model, params, 60., views=views)
    return model, history, platform, views, baseline, barrier


@pytest.mark.parametrize('cancel,forecast', [(0.,None), (.05,43.)])
def test_unmasked_barrier_preserves_scores_pre_goal_actions_and_ranks(monkeypatch, cancel, forecast):
    model, history, platform, views, baseline, barrier = barrier_case(
        monkeypatch, cancel=cancel, forecast=forecast)
    ride = platform.rides[0]
    original = baseline.compile_ride(ride, history, 'rolling_fixed')
    reduced = barrier.compile_ride(ride, history, 'rolling_fixed')
    assert original.arrival_cids == (0,1,2)
    assert reduced.arrival_cids == (0,1)
    np.testing.assert_allclose(reduced.departure_times, original.departure_times)
    np.testing.assert_allclose(reduced.board_scores, original.board_scores, atol=1e-10)
    np.testing.assert_allclose(reduced.policy.decision_costs[0], original.policy.decision_costs[0], atol=1e-10)
    np.testing.assert_array_equal(reduced.policy.actions[0], original.policy.actions[0])
    np.testing.assert_allclose(reduced.policy.values[0], original.policy.values[0], atol=1e-10)
    for clock in original.times[0]:
        assert reduced.decision(0,clock)[0] == original.decision(0,clock)[0]
    np.testing.assert_allclose(
        barrier.first_offer_values(platform, history, np.array([0.,12.,19.]), 'rolling_fixed'),
        baseline.first_offer_values(platform, history, np.array([0.,12.,19.]), 'rolling_fixed'), atol=1e-10)
    assert barrier.ranks(views.platform(platform)) == baseline.ranks(platform)
    assert views.original(views.ride(ride)) is ride
    assert ride.path == (0,1,2) and 2 in ride.exits  # Original menu stays intact.
    assert barrier.relevant_trips(ride) == {'A','B'}
    assert baseline.relevant_trips(ride) == {'A','B','C'}
    tail_original = baseline.tail_base.compile_ride(ride, history, 'rolling_fixed')
    tail_reduced = barrier.tail_base.compile_ride(ride, history, 'rolling_fixed')
    np.testing.assert_allclose(tail_reduced.board_scores, tail_original.board_scores, atol=1e-10)


def test_masked_first_goal_keeps_scoring_suffix(monkeypatch):
    _, history, platform, views, baseline, barrier = barrier_case(monkeypatch, masked=True)
    ride = platform.rides[0]
    assert views.ride(ride).path == ride.path
    assert views.stats['goal_barriers'] == 0
    old = baseline.compile_ride(ride, history, 'rolling_fixed')
    new = barrier.compile_ride(ride, history, 'rolling_fixed')
    np.testing.assert_allclose(new.board_scores, old.board_scores)
    # A masked goal has cost H, so a later permitted exit can change action.
    full = OnboardStopping([[0],[10],[20]], [[50],[100],[20]], [[[1]],[[1]]], 100).solve()
    cut = OnboardStopping([[0],[10]], [[50],[100]], [[[1]]], 100).solve()
    assert full.policies['rolling_fixed'].action(0,0) == 'continue'
    assert cut.policies['rolling_fixed'].action(0,0) == 'alight'


def test_late_same_trip_evidence_is_retained_and_changes_early_score(monkeypatch):
    _, history, platform, views, baseline, barrier = barrier_case(monkeypatch, forecast=43.)
    ride = platform.rides[0]
    contract = ObservationContract({0:(0,2,4,5)})
    reads = BarrierReadSets(barrier.model, contract, views)
    old_reads = RemainingReadSets(barrier.model, contract)
    assert old_reads.ride_trips(ride) == {'A','B','C'}
    assert reads.ride_trips(ride) == {'A','B'}
    assert reads.observation_cids(0, reads.ride_trips(ride)) == (0,2)
    assert history.selected_forecast('A')[:2] == (2,'arrival')
    old = baseline.compile_ride(ride, history, 'rolling_fixed')
    new = barrier.compile_ride(ride, history, 'rolling_fixed')
    np.testing.assert_allclose(new.board_scores, old.board_scores, atol=1e-10)
    assert len(history.law('A').times) == 6  # No chain-slot marginal truncation.
    no_message = history.clone()
    no_message._drop_forecast(2, 'arrival')
    changed = barrier.compile_ride(ride, no_message, 'rolling_fixed')
    assert np.max(np.abs(changed.board_scores-new.board_scores)) > 1e-3


def test_nominal_counterexample_and_explicit_rolling_scope(monkeypatch):
    # Mean-before-interpolation is not an order-preserving expectation of the
    # clipped arrival. A late target can change the old nominal comparison.
    times = [[0],[10,90,200],[20]]
    costs = [[80],[10,90,100],[20]]
    kernels = [[[.49995,.0001,.49995]], [[1],[0],[0]]]
    full = nominal_policy(OnboardStopping(times, costs, kernels, 100), [1.])
    cut = nominal_policy(OnboardStopping(times[:2], costs[:2], kernels[:1], 100), [1.])
    assert full.action(0,0) == 'continue'
    assert cut.action(0,0) == 'alight'
    _, history, platform, _, baseline, barrier = barrier_case(monkeypatch)
    baseline.compile_ride(platform.rides[0], history, 'nominal')
    with pytest.raises(ValueError, match='restricted to rolling_fixed'):
        barrier.compile_ride(platform.rides[0], history, 'nominal')
