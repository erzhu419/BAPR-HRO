from types import SimpleNamespace

from src.platform_nominal_index import PlatformNominalIndex
from src.transit_graph import Connection, Stop, TransitGraph


def test_nominal_index_applies_mct_and_keeps_no_pickup_same_clock_segments():
    graph = TransitGraph()
    for sid in range(5):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=120))

    def add(origin, goal, dep, arr, trip, pickup=True):
        connection = Connection(0, trip, trip, origin, goal, dep, arr, pickup_allowed=pickup)
        graph.add_connection(connection)
        return connection.id

    first = add(0, 1, 100, 100, 'A')
    onward = add(1, 2, 100, 100, 'A', pickup=False)
    through = add(2, 4, 100, 250, 'A', pickup=False)
    add(1, 4, 219, 220, 'B')
    transfer = add(2, 4, 220, 245, 'C')
    add(1, 4, 220, 240, 'D', pickup=False)

    index = PlatformNominalIndex(graph, 4, 0, 600)
    nominal = {c.id: SimpleNamespace(departure=c.dep_time, arrival=c.arr_time) for c in graph.connections}
    assert index.conditional_path(first, nominal, 100) == (first, onward, transfer)
    assert index.path(1, 220) is None
    assert index.path(2, 220) == (transfer,)

    for cid in (first, onward, through):
        event = nominal[cid]
        nominal[cid] = SimpleNamespace(departure=event.departure+1, arrival=event.arrival+1)
    assert index.conditional_path(first, nominal, 101) == (first, onward, through)
