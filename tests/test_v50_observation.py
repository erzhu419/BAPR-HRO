"""Common public sight must not depend on the controller's action menu."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v34.execute import Executor as NativeExecutor
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v50.execute import ObservationContract, Executor, SnapshotExecutor
from src import evolving_trip_belief as evolving
from src.transit_graph import Connection, Stop, TransitGraph


def case(monkeypatch):
    monkeypatch.setattr(evolving, 'DELAYS', np.array([0., 8/60]))
    graph = TransitGraph()
    graph.add_stop(Stop(0, 'platform', min_transfer_time=5))
    graph.add_stop(Stop(1, 'destination', min_transfer_time=0))
    clocks = {0:(12, 18), 1:(20, 25), 2:(20, 27), 3:(30, 35), 4:(17, 23)}
    for cid, (departure, arrival) in clocks.items():
        graph.add_connection(Connection(cid, 'R', str(cid), 0, 1, departure, arrival))
    model = evolving.EvolvingTripBelief(graph, public_occurrences=False,
        travel_sd=.1, dwell_sd=.1, cancel_persistence=.8)
    params = {'R':dict(mean=0., std=.1, cancel_rate=0.)}
    fit = dict(route_kind={}, pooled={kind:dict(bias_seconds=0., sd_seconds=5.)
        for kind in ('departure', 'arrival')})
    frames = [dict(receipt_seconds=t, trip_statuses=[], endpoint_updates=[
        [1, 'arrival', 'FORECAST', float(25+t)]]) for t in (11., 19., 21.)]
    history = PhysicalHistory(model, params, fit=fit, frames=frames)
    wide = Platform(0, {cid:Ride(cid, (cid,), {cid:None}) for cid in clocks})
    narrow = Platform(0, {1:wide.rides[1]})

    class World:
        def event(self, cid, kind):
            return clocks[cid][kind == 'arrival'], False, str(cid)

    return graph, model, history, wide, narrow, World()


class ScriptedExecutor(Executor):
    def choose_platform(self, platform, entry, ready, history, visible, observed, clock):
        self.calls.append(dict(entry=entry, ready=ready, clock=clock,
            visible=tuple(visible), observed=dict(observed), history=history.public_state()))
        return (self.target if self.target in visible else None), dict(method='test_script')


class ScriptedNative(NativeExecutor):
    choose_platform = ScriptedExecutor.choose_platform


def actual(cls, graph, contract=None, target=1):
    kwargs = dict(method='rolling_fixed', waiting='combined')
    if issubclass(cls, Executor):
        kwargs['observation_contract'] = contract
    instance = cls(graph, SimpleNamespace(), 60, **kwargs)
    instance.calls, instance.target = [], target
    return instance


def test_common_observations_match_across_action_menus_and_keep_current_choices_legal(monkeypatch):
    graph, _, history, wide, narrow, world = case(monkeypatch)
    contract = ObservationContract.from_menu(wide)
    controllers = []
    for menu in (narrow, wide):
        controller = actual(ScriptedExecutor, graph, contract)
        result = controller.platform(menu, 10., history.clone(), world)
        assert result == (25, 'arrived')
        controllers.append(controller)
    left, right = controllers
    a, b = left.calls[-1], right.calls[-1]
    assert a['history'] == b['history']
    assert a['clock'] == b['clock'] == 20
    assert a['visible'] == (1,)
    assert b['visible'] == (1, 2)
    assert all(call['clock'] >= 15 for c in controllers for call in c.calls)
    assert [r['clock'] for r in left.trace if r['kind'] == 'observation'] == [12, 17]
    at20 = next(r for r in left.trace if r['kind'] == 'platform')
    assert at20['observed_visible'] == (1, 2)
    assert set(at20['observed']) == {0, 1, 2, 4}
    assert set(a['observed']) == {1}


def test_only_received_remote_frames_are_consumed_and_absence_stops_on_departure(monkeypatch):
    graph, _, history, wide, narrow, world = case(monkeypatch)
    controller = actual(ScriptedExecutor, graph, ObservationContract.from_menu(wide))
    controller.platform(narrow, 10., history, world)
    assert history.frame_index == 2  # 21-second publication is still in the future at BOARD.
    before = next(r for r in history.public_state() if r.get('cid') == 3 and r['evidence'] == 'unseen')
    assert before['windows'] == ((10., 20.),)
    controller.decision_history(history, 40.)  # A later onboard/public clock.
    after = next(r for r in history.public_state() if r.get('cid') == 3 and r['evidence'] == 'unseen')
    assert after == before
    assert history.frame_index == 3


def test_native_mode_delegates_original_trace_and_aggregation(monkeypatch):
    graph, _, history, wide, _, world = case(monkeypatch)
    old = actual(ScriptedNative, graph)
    new = actual(ScriptedExecutor, graph)
    assert old.platform(wide, 10., history.clone(), world) == new.platform(wide, 10., history.clone(), world)
    assert old.calls == new.calls
    assert old.trace == new.trace
    assert old.counts == new.counts


class ScriptedEngine:
    def __init__(self, target):
        self.target, self.calls = target, []

    def platform(self, platform, entry, ready, history, visible, observed, clock, **kwargs):
        self.calls.append((entry, ready, clock, tuple(visible), history.public_state()))
        return (self.target if self.target in visible else None), {}


def test_snapshot_wait_keeps_original_entry_ready_and_does_not_consume_frames(monkeypatch):
    graph, _, history, wide, _, world = case(monkeypatch)
    contract = ObservationContract.from_menu(wide)
    engine = ScriptedEngine(3)
    controller = SnapshotExecutor(graph, SimpleNamespace(), 60,
        engine=engine, observation_contract=contract)
    menu = Platform(0, {1:wide.rides[1], 3:wide.rides[3]})
    result = controller.platform(menu, 10., history, world, after=20.)
    assert result == (35, 'arrived')
    assert engine.calls[0][:3] == (10., 15., 30)
    assert history.frame_index == 0
    assert history.message_clock == float('-inf')
    assert not any(r['evidence'] == 'forecast' for r in history.public_state())
    assert all(r['clock'] > 20 for r in controller.trace)


def test_snapshot_without_trace_preserves_decisions_and_history_without_serializing_evidence(monkeypatch):
    graph, _, history, wide, narrow, world = case(monkeypatch)
    contract = ObservationContract.from_menu(wide)
    results = []
    for trace in (True, False):
        current = history.clone()
        engine = ScriptedEngine(1)
        controller = SnapshotExecutor(graph, SimpleNamespace(), 60, engine=engine,
            observation_contract=contract, trace=trace)
        result = controller.platform(narrow, 10., current, world)
        results.append((result, engine.calls, current.public_state(), current.frame_index, controller.counts))
        assert bool(controller.trace) is trace
    assert results[0] == results[1]
    silent = SnapshotExecutor(graph, SimpleNamespace(), 60, engine=ScriptedEngine(1),
        observation_contract=contract, trace=False)
    # This direct evidence hook must not serialize history when no trace uses
    # it; normal controller policy calls remain free to read public history.
    class NoSerialization:
        def public_state(self):
            raise AssertionError('Unused public evidence was serialized')
    assert silent.observation_evidence(NoSerialization(), 20.) == {}


def test_contract_includes_recursive_scoring_tails_and_original_root_trip_metadata(monkeypatch):
    _, model, _, wide, narrow, _ = case(monkeypatch)
    root = Ride(10, (10, 11), {10:None, 11:wide})
    contract = ObservationContract.from_menu(root)
    assert contract.cids_at(0) == (0, 1, 2, 3, 4)
    assert contract.cids_at(99) == ()
    assert contract.trips(model) == {'0', '1', '2', '3', '4'}
    assert set(contract.paths()) == {(0,), (1,), (2,), (3,), (4,)}
    assert contract.describe()['station_to_cids'] == [dict(stop=0, cids=[0, 1, 2, 3, 4])]
    assert set(narrow.rides).issubset(contract.cids_at(0))


def test_action_domain_must_be_observable(monkeypatch):
    graph, _, history, wide, _, world = case(monkeypatch)
    controller = actual(ScriptedExecutor, graph, ObservationContract({0:(0,)}))
    with pytest.raises(ValueError, match='Every action-menu'):
        controller.platform(wide, 10., history, world)


def test_observation_only_vehicle_cannot_be_selected_for_boarding(monkeypatch):
    graph, _, history, wide, narrow, world = case(monkeypatch)
    class Invalid(Executor):
        def choose_platform(self, *args):
            return 2, {}
    controller = Invalid(graph, SimpleNamespace(), 60, observation_contract=ObservationContract.from_menu(wide))
    with pytest.raises(ValueError, match='not a current legal offer'):
        controller.platform(narrow, 10., history, world)
