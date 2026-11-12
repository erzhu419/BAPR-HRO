from types import SimpleNamespace

import pytest

from experiments.reconstruction_v33.nodes import Platform, Ride
from experiments.reconstruction_v34.conditional_reference import (
    AnchoredWorld, SnapshotExecutor, locate_node, restore_snapshot, summarize_pairs,
)
from experiments.reconstruction_v34.forecast import MessageHistory
from src.transit_graph import Connection, Stop, TransitGraph


def model_fixture():
    graph = TransitGraph()
    for index in range(3):
        graph.add_stop(Stop(index, str(index), min_transfer_time=0))
    graph.add_connection(Connection(0, 'R', 'a', 0, 1, 0, 10))
    graph.add_connection(Connection(1, 'R', 'b', 1, 2, 15, 30))
    positions = {(cid, kind): (trip, slot) for cid, trip in ((0, 'a'), (1, 'b'))
                 for slot, kind in enumerate(('departure', 'arrival'))}
    return SimpleNamespace(graph=graph, _positions=positions), {'R': {'std': 1.}}


def test_restore_snapshot_preserves_receipts_status_latches_and_discards_future_stream():
    model, parameters = model_fixture()
    frames = [dict(receipt_seconds=1, trip_statuses=[['b', 'CANCELED']],
                   endpoint_updates=[[0, 'arrival', 'FORECAST', 12.], [1, 'arrival', 'SKIPPED', None]]),
              dict(receipt_seconds=2, trip_statuses=[], endpoint_updates=[[1, 'arrival', 'NO_DATA', None]]),
              dict(receipt_seconds=99, trip_statuses=[['b', 'SCHEDULED']], endpoint_updates=[])]
    history = MessageHistory(model, parameters, frames=frames)
    history.advance(2)
    history.observe(0, 'departure', 0)
    history.observe_platform([1], 0, 2, {})
    snapshot = restore_snapshot(model, parameters, history.public_state(), 2)
    assert snapshot.signature() == history.signature()
    assert snapshot.public_state() == history.public_state()
    assert snapshot.announced_unavailable(1, 'arrival')
    assert snapshot.frames == ()
    signature = snapshot.signature()
    assert snapshot.advance(100)['frames_consumed'] == 0
    assert snapshot.signature() == signature


def test_location_uses_the_executed_branch_even_when_a_cid_is_reused():
    first_tail, other_tail = Ride(7, (7,), {7: None}), Ride(7, (7,), {7: None})
    first = Platform(2, {7: first_tail})
    other = Platform(2, {7: other_tail})
    ride = Ride(0, (0, 1), {0: first, 1: other})
    trace = [dict(kind='onboard', cid=0, choice='continue'),
             dict(kind='onboard', cid=1, choice='alight'),
             dict(kind='platform', stop=2, choice=7),
             dict(kind='onboard', cid=7, choice='continue')]
    assert locate_node(ride, trace, 2) is other
    assert locate_node(ride, trace, 3) is other_tail


def test_current_wait_does_not_board_the_offer_it_has_just_rejected():
    model, parameters = model_fixture()
    bank = SimpleNamespace(belief=model, events={(1, 'departure'): ([15], [False]),
                                               (1, 'arrival'): ([30], [False])})
    public = [dict(cid=1, kind='departure', evidence='observed', clock=15., trip='b')]
    world = AnchoredWorld(bank, 0, public)
    history = restore_snapshot(model, parameters, public, 15)
    platform = Platform(1, {1: Ride(1, (1,), {1: None})})
    row = dict(kind='platform', entry=10, clock=15, visible=[1])
    waiting = SnapshotExecutor(model.graph, None, 100, method='rolling_fixed', waiting='combined')
    boarding = SnapshotExecutor(model.graph, None, 100, method='rolling_fixed', waiting='combined')
    assert waiting.forced(platform, row, None, history.clone(), world)['loss'] == 85
    assert boarding.forced(platform, row, 1, history.clone(), world)['loss'] == 15


def test_offgrid_anchor_cannot_silently_shift_a_future_draw_or_become_service_failure():
    model, parameters = model_fixture()
    bank = SimpleNamespace(belief=model, events={(1, 'departure'): ([15], [False]),
                                               (1, 'arrival'): ([16], [False])})
    public = [dict(cid=1, kind='departure', evidence='observed', clock=17., trip='b')]
    world = AnchoredWorld(bank, 0, public)
    assert world.event(1, 'departure')[0] == 17
    assert world.event(1, 'arrival')[0] == 16
    history = restore_snapshot(model, parameters, public, 17)
    platform = Platform(1, {1: Ride(1, (1,), {1: None})})
    executor = SnapshotExecutor(model.graph, None, 100)
    result = executor.forced(platform, dict(kind='platform', clock=17, visible=[1]), 1, history, world)
    assert result['loss'] is None
    assert result['status'].startswith('projection_incompatible:')


def test_unresolved_draws_retain_their_weight_in_full_bank_bounds():
    summary = summarize_pairs([10., None, 20.], [5., 30., None], 100.)
    assert summary['complete_pairs'] == 1 and summary['unresolved_pairs'] == 2
    assert summary['complete_pair_statistics']['mean_gain'] == 5.
    assert summary['full_bank_mean_gain_interval'] == pytest.approx([-35., 95/3])
