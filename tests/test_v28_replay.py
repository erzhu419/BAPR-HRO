from copy import deepcopy

import numpy as np
import pytest

from experiments.reconstruction_v25 import run as old_replay
from experiments.reconstruction_v28 import run as replay
from src.transit_graph import Connection, Stop, TransitGraph


class StaticScores:
    """Fixed model marks; actual endpoint changes cannot enter this fixture."""

    def __init__(self, graph, belief, params, deadline, options):
        self.graph, self.options = graph, options

    def departure_distribution(self, cid):
        return np.array([self.graph.connections[cid].dep_time]), np.array([1.]), 0.

    def clock_nominal_score(self, cid, clock):
        option = self.options[cid]
        path = next(iter(option.tails.values())) if option.tails else option.first_path
        return float(self.graph.connections[path[-1]].arr_time)


def fixture(mct=45):
    graph = TransitGraph()
    for sid in range(4):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=mct if sid == 1 else 0))
    ready = 100+max(30, mct)
    rows = [(0, 1, 40, 80), (1, 2, 99, 240), (1, 2, 100, 240),
            (1, 2, ready-1, 240), (1, 2, ready, 220), (1, 2, 200, 250),
            (2, 3, 260, 290)]
    metadata = {}
    for cid, (dep_stop, arr_stop, departure, arrival) in enumerate(rows):
        graph.add_connection(Connection(cid, str(cid), str(cid), dep_stop, arr_stop,
                                        departure, arrival))
        identity = f'vehicle-{cid}'
        metadata[cid] = dict(scheduled_departure_seconds=departure,
            departure=dict(status='REAL', time_seconds=departure, official_journey=identity),
            arrival=dict(status='REAL', time_seconds=100 if cid == 0 else arrival,
                         official_journey=identity))
    root = dict(root_id='tiny_transfer', day='test', root_kind='first_transfer',
                start=0, deadline=300, stop=1, prefix=[0], full_options=[
                    dict(first_cid=cid, first_path=[cid], tail_paths=[],
                         scheduled_arrival=graph.connections[cid].arr_time)
                    for cid in range(1, 6)])
    return graph, metadata, root


def patch_scores(monkeypatch):
    monkeypatch.setattr(replay, 'TransferOpportunityScores', StaticScores)
    monkeypatch.setattr(old_replay, 'TransferOpportunityScores', StaticScores)


def test_prefix_actual_arrival_and_mct_floor_match_original_baselines(monkeypatch):
    patch_scores(monkeypatch)
    for mct in (10, 45):
        graph, metadata, root = fixture(mct)
        row = replay.run_root(graph, metadata, None, {}, root)
        old = old_replay.run_root(graph, metadata,
            dict(original_full_nominal=None, empirical_full_nominal=None), {}, root)
        assert row['entry'] == 100  # Scheduled prefix arrival was 80.
        assert row['ready'] == 100+max(30, mct)
        first = row['outcomes']['full_first_offer']
        assert (first['action'], first['loss']) == (4, 220)
        replay.assert_same(first, old['outcomes']['full_first_offer'], 'V25 first')
        replay.assert_same(row['outcomes']['one_step_nominal'],
                           old['outcomes']['original_full_nominal'], 'V25 nominal')


def test_prefix_public_history_excludes_unseen_past_and_future_endpoints(monkeypatch):
    patch_scores(monkeypatch)
    graph, metadata, root = fixture()
    later = deepcopy(metadata)
    later[5]['departure']['time_seconds'] = 270
    later[5]['arrival']['time_seconds'] = 295
    a = replay.run_root(graph, metadata, None, {}, root)
    b = replay.run_root(graph, later, None, {}, root)
    for method in replay.METHODS:
        before, after = a['outcomes'][method]['trace'][0], b['outcomes'][method]['trace'][0]
        assert before['clock'] == 145
        assert before['visible'] == (4,)
        assert set(before['seen']) == {2, 3, 4}
        assert {k: v for k, v in before.items() if k != 'wall'} == {
            k: v for k, v in after.items() if k != 'wall'}


def test_empty_menu_cap_and_missing_prefix_root_tail_remain_distinct(monkeypatch):
    patch_scores(monkeypatch)
    graph, metadata, root = fixture()
    empty = dict(root, full_options=[])
    cases = []
    cases.append((metadata, empty, 'no_menu', 300))
    missing_prefix = deepcopy(metadata)
    missing_prefix[0]['arrival'] = dict(status='MISSING')
    cases.append((missing_prefix, root, 'prefix_unresolved_arrival', None))
    missed_prefix = deepcopy(metadata)
    missed_prefix[0]['departure']['time_seconds'] = 20
    cases.append((missed_prefix, root, 'prefix_deadline', 300))
    missing_root = deepcopy(metadata)
    missing_root[5]['departure'] = dict(status='MISSING')
    cases.append((missing_root, root, 'unresolved_root_departure', None))
    missing_tail = deepcopy(metadata)
    missing_tail[6]['departure'] = dict(status='MISSING')
    tail_root = dict(root, full_options=[dict(first_cid=4, first_path=[4],
                                            tail_paths=[[6]], scheduled_arrival=290)])
    cases.append((missing_tail, tail_root, 'unresolved_tail_departure', None))
    for actual, selected, status, loss in cases:
        outcomes = replay.run_root(graph, actual, None, {}, selected)['outcomes']
        assert all(o['status'] == status and o['loss'] == loss for o in outcomes.values())


def test_frozen_outcome_comparison_checks_continuation_and_missingness():
    old = dict(action=4, tail=6, loss=290, arrived=True, status='arrived')
    replay.assert_same(dict(old, online_wall=99.), old, 'V27 construction')
    for changed in (dict(old, tail=7), dict(old, loss=291),
                    dict(old, loss=None, arrived=False, status='unresolved_arrival')):
        with pytest.raises(AssertionError, match='Frozen control outcome changed'):
            replay.assert_same(changed, old, 'V27 construction')
    # V25's prefix-failure records omitted tail; V28 makes that None explicit.
    replay.assert_same(dict(action=None, tail=None, loss=None, arrived=False,
                            status='prefix_unresolved_arrival'),
                       dict(action=None, loss=None, arrived=False,
                            status='prefix_unresolved_arrival'), 'V25 prefix')
