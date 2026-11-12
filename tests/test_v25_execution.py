import numpy as np

from experiments.reconstruction_v25.run import execute, public_offers, route_outcome
from src.full_menu_opportunity import FullMenuOpportunity
from src.transfer_opportunity import RouteOption
from src.transit_graph import Connection, Stop, TransitGraph


def graph(connections, transfers=None):
    result = TransitGraph()
    transfers = transfers or {}
    stops = {stop for _, dep, arr, _, _ in connections for stop in (dep, arr)}
    for sid in stops:
        result.add_stop(Stop(sid, str(sid), min_transfer_time=transfers.get(sid, 0)))
    for cid, dep, arr, clock, arrival in connections:
        result.add_connection(Connection(cid, 'R', f'trip{cid}', dep, arr, clock, arrival))
    return result


def endpoints(cid, departure, arrival, *, canceled=False):
    return dict(scheduled_departure_seconds=departure,
                departure=dict(status='CANCELED' if canceled else 'REAL', time_seconds=departure,
                               official_journey=f'vehicle{cid}'),
                arrival=dict(status='REAL' if arrival is not None else 'UNKNOWN', time_seconds=arrival,
                             official_journey=f'vehicle{cid}'))


def test_partial_departure_is_a_public_offer_but_missing_chosen_arrival_stays_unresolved():
    net = graph([(0, 0, 1, 10, 30)])
    metadata = {0: endpoints(0, 10, None)}
    options, ranks = {0: RouteOption((0,), {})}, {0: (20, 0)}
    offers = public_offers(metadata, options, 0, 0, 100)
    assert offers == [dict(clock=10, visible=(0,), seen=(0,))]
    outcome = execute(net, metadata, dict(start=0, deadline=100), options, ranks, offers)
    assert outcome['action'] == 0
    assert outcome['status'] == 'unresolved_arrival'
    assert outcome['loss'] is None and not outcome['arrived']


def test_full_menu_refuses_twice_then_accepts_a_different_public_offer():
    net = graph([(0, 0, 1, 10, 90), (1, 0, 1, 20, 80), (2, 0, 1, 30, 40)])
    metadata = {cid: endpoints(cid, clock, arrival) for cid, clock, arrival in ((0, 10, 90), (1, 20, 80), (2, 30, 40))}
    options = {cid: RouteOption((cid,), {}) for cid in metadata}
    ranks = {cid: (net.connections[cid].arr_time-net.connections[cid].dep_time, cid) for cid in metadata}

    class Scores:
        def departure_distribution(self, cid):
            return np.array([net.connections[cid].dep_time]), np.array([1.]), 0.

        def clock_nominal_score(self, cid, clock):
            return net.connections[cid].arr_time

    controller = FullMenuOpportunity(Scores(), ranks, 0, 0, 100)
    offers = public_offers(metadata, options, 0, 0, 100)
    outcome = execute(net, metadata, dict(start=0, deadline=100), options, ranks, offers, controller)
    assert [event['choice'] for event in outcome['trace']] == [None, None, 2]
    assert [event['seen'] for event in outcome['trace']] == [(0,), (0, 1), (0, 1, 2)]
    assert outcome['loss'] == 40 and outcome['arrived']


def test_tail_execution_respects_transfer_time_cancellation_and_simultaneous_rank():
    net = graph([(0, 0, 1, 10, 20), (1, 1, 2, 24, 35), (2, 1, 2, 30, 32),
                 (3, 1, 2, 30, 55), (4, 1, 2, 30, 45)], {1: 5})
    metadata = {cid: endpoints(cid, clock, arrival, canceled=cid == 2)
                for cid, clock, arrival in ((0, 10, 20), (1, 24, 35), (2, 30, 32), (3, 30, 55), (4, 30, 45))}
    option = RouteOption((0,), {cid: (cid,) for cid in range(1, 5)})
    # Tail1 leaves inside MCT; canceled tail2 cannot win; tail4 wins the tie with3.
    assert route_outcome(net, metadata, option, 10, 100) == (45, 'arrived', 4)
