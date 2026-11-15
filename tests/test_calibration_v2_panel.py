from types import SimpleNamespace

from experiments.decision_calibration_v2.run_calibration import initial_state, select_public


def root(index, *, trip=None, route=None, **fields):
    return dict(root_id=index, day='2023-03-09', first_trip=trip or str(index),
        first_route=route or str(index), stop=0, goal=2,
        has_downstream_only_direct_tail=True, shared_tail_trips={},
        has_two_transfer_option=False, **fields)


def test_public_round_robin_keeps_missing_and_negative_roots():
    rows = [root(0, actual_status='MISSING', realized_loss=3600),
            root(1, trip='0', actual_status='REAL', realized_loss=0),
            root(2, actual_status='REAL', realized_loss=0)]
    rows[2]['shared_tail_trips'] = {'tail': [1, 2]}
    result = select_public({'candidates': rows}, ['2023-03-09'], 2)
    assert [r['root_id'] for r in result['selected']] == [0, 2]
    rows[0].update(actual_status='REAL', realized_loss=0)
    rows[2].update(actual_status='MISSING', realized_loss=3600)
    repeated = select_public({'candidates': rows}, ['2023-03-09'], 2)
    assert [r['root_id'] for r in repeated['selected']] == [0, 2]


class History:
    frame_index = 0
    message_clock = 0
    frames = []

    def __init__(self):
        self.own = []
        self.advances = []

    def observe(self, cid, kind, clock):
        self.own.append((cid, kind, clock))

    def advance(self, clock):
        self.advances.append(clock)
        return {'receipts': []}

    def public_state(self):
        return []


def endpoint(clock, status='REAL'):
    return dict(status=status, time_seconds=clock, official_journey='same_train')


def test_initial_prefix_stops_before_any_future_endpoint_or_decision():
    node = SimpleNamespace(path=[0, 1], exits={0: SimpleNamespace(rides={99: None}), 1: None})
    history = History()
    # There is deliberately no metadata for the future segment or outgoing99.
    metadata = {0: dict(departure=endpoint(10), arrival=endpoint(20))}
    state = initial_state(None, dict(start=10, deadline=100), node, history, metadata)
    assert state == dict(status='observed', cid=0, clock=20.)
    assert history.own == [(0, 'departure', 10), (0, 'arrival', 20)]
    assert history.advances == [20]


def test_initial_prefix_missing_is_unresolved_and_does_not_invent_clock():
    node = SimpleNamespace(path=[0], exits={0: SimpleNamespace(rides={99: None})})
    history = History()
    metadata = {0: dict(departure=endpoint(None, 'MISSING'), arrival=endpoint(20))}
    state = initial_state(None, dict(start=10, deadline=100), node, history, metadata)
    assert state['status'].startswith('unresolved:')
    assert state['clock'] is None
    assert history.own == history.advances == []
