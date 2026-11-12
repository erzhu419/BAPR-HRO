from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from experiments.reconstruction_v33.nodes import Ride, platform_from_options
from experiments.reconstruction_v34.execute import ArchivedWorld, Executor
from experiments.reconstruction_v34.forecast import MessageHistory
from src.evolving_trip_belief import _ChainLaw
from src.transit_graph import Connection, Stop, TransitGraph


def fixture():
    graph = TransitGraph()
    for stop in range(3):
        graph.add_stop(Stop(stop, str(stop), min_transfer_time=5 if stop == 1 else 0))
    rows = [('A', 0, 1, 10, 20), ('B', 1, 2, 25, 40), ('C', 1, 2, 30, 50)]
    metadata, positions, laws = {}, {}, {}
    grid = np.arange(0, 151, 5)
    kernel = np.full((len(grid), len(grid)), -np.log(len(grid)))
    for cid, (trip, origin, destination, departure, arrival) in enumerate(rows):
        graph.add_connection(Connection(cid, trip, trip, origin, destination, departure, arrival))
        metadata[cid] = dict(trip_id=trip,
            scheduled_departure_seconds=departure, scheduled_arrival_seconds=arrival,
            departure=dict(status='REAL', time_seconds=departure, official_journey=trip),
            arrival=dict(status='REAL', time_seconds=arrival, official_journey=trip))
        positions[cid, 'departure'], positions[cid, 'arrival'] = (trip, 0), (trip, 1)
        laws[trip] = _ChainLaw([grid, grid], np.full(len(grid), -np.log(len(grid))),
            lambda index: kernel, [np.zeros(len(grid)), np.zeros(len(grid))],
            [np.zeros(len(grid), bool), np.zeros(len(grid), bool)], (True,))
    belief = SimpleNamespace(graph=graph, _positions=positions, _law=lambda trip, params: laws[trip])
    platform = platform_from_options(graph, 1, [dict(first_cid=cid, first_path=[cid], tail_paths=[])
                                               for cid in (1, 2)])
    ride = Ride(0, (0,), {0: platform})
    frames = [dict(receipt_seconds=receipt, trip_statuses=[],
                   endpoint_updates=[(1, 'arrival', 'FORECAST', prediction)])
              for receipt, prediction in [(18, 40), (23, 45), (26, 100)]]
    return graph, metadata, belief, ride, frames


class FakePlanner:
    def __init__(self):
        self.calls = []

    def compile_ride(self, ride, history, method):
        # A hypothetical future marginal is evaluated without advancing messages.
        history.clone().law('B').marginal(1)
        self.calls.append(('onboard', history.frame_index, history.message_clock))
        return SimpleNamespace(decision=lambda cid, clock: (True, dict(current_cid=cid)))

    def platform_reference(self, platform, history, entry, ready, method):
        self.calls.append(('platform', history.frame_index, history.message_clock))
        return SimpleNamespace(choose=lambda visible, seen, clock, mode:
                               (min(visible), dict(mode=mode)))


def execute(metadata=None, frames=None):
    graph, original, belief, ride, default_frames = fixture()
    planner = FakePlanner()
    history = MessageHistory(belief, {}, frames=default_frames if frames is None else frames)
    result = Executor(graph, planner, 150, waiting='first').run(
        ride, 0, history, ArchivedWorld(original if metadata is None else metadata))
    return result, planner, history


def test_actual_decisions_reveal_only_received_messages_and_keep_receipts():
    result, planner, history = execute()
    assert result['status'] == 'arrived' and result['loss'] == 40
    assert planner.calls == [('onboard', 1, 20.), ('platform', 2, 25.)]
    assert history.frame_index == 2  # Receipt26 was not known at either decision.
    onboard, platform = result['trace'][:2]
    for row, receipt in [(onboard, 18), (platform, 23)]:
        assert row['message_receipt']['frames_consumed'] == 1
        assert row['message_receipt']['receipts'][0]['receipt_seconds'] == receipt
        assert all(e.get('receipt_seconds', row['clock']) <= row['clock']
                   for e in row['public_history'])
    assert onboard['message_receipt']['receipts'][0]['frame_index'] == 0
    assert platform['message_receipt']['receipts'][0]['frame_index'] == 1
    assert platform['message_receipt']['receipts'][0]['consumed_at_clock'] == 25
    assert any(e.get('predicted_time') == 40 for e in onboard['public_history'])
    assert any(e.get('predicted_time') == 45 for e in platform['public_history'])
    assert any(e.get('cid') == 0 and e.get('kind') == 'arrival' and e.get('clock') == 20
               for e in platform['public_history'])


def test_unreceived_forecast_and_unrealized_arrival_do_not_change_earlier_decision():
    _, metadata, _, _, frames = fixture()
    altered = deepcopy(metadata)
    altered[1]['arrival']['time_seconds'] = 55
    changed_frames = deepcopy(frames)
    changed_frames[1]['endpoint_updates'][0] = (1, 'arrival', 'FORECAST', 90)
    a, _, _ = execute(metadata, frames)
    b, _, _ = execute(altered, changed_frames)
    assert a['trace'][0] == b['trace'][0]
    assert (a['loss'], b['loss']) == (40, 55)


def test_transfer_time_and_unresolved_endpoint_contract_is_preserved():
    _, metadata, _, _, _ = fixture()
    metadata[1]['departure']['time_seconds'] = 24
    result, _, _ = execute(metadata)
    event = next(e for e in result['trace'] if e['kind'] == 'platform')
    assert event['ready'] == 25 and event['visible'] == (2,) and event['choice'] == 2
    assert event['observed'] == {1: 24, 2: 30}
    metadata[1]['departure']['status'] = 'MISSING'
    result, _, _ = execute(metadata)
    assert result['loss'] is None and result['status'] == 'unresolved:1:departure:MISSING'
