"""Clairvoyant arrival on a complete fixed vehicle world for diagnostics."""
from heapq import heappop, heappush
from itertools import count
from math import inf


def earliest_arrival(graph, scene, start, destination, deadline, *, transfer_seconds=None):
    """Return the absolute arrival and path in a relaxed platform network.

    The oracle sees all realized service events. Same-trip reboarding is
    relaxed to any later leg of the same trip at this stop. Other boarding
    requires pickup permission. The default transfer duration is zero.
    There are no footpaths in this diagnostic network. The scene is assumed
    complete on the queried network, so missing actuals cannot be used here.
    """
    if start.stop == destination:
        return start.time, ()
    transfer = transfer_seconds or 0
    serial = count()
    queue = []
    best = {}

    def offer(stop, trip, time, path):
        key = stop, trip
        if time <= deadline and time < best.get(key, inf):
            best[key] = time
            heappush(queue, (time, next(serial), stop, trip, path))

    offer(start.stop, start.current_trip, start.time, ())
    if start.current_trip is not None:
        offer(start.stop, None, max(start.time, start.last_arrival + transfer), ())
    while queue:
        time, _, stop, trip, path = heappop(queue)
        if best[stop, trip] != time:
            continue
        if stop == destination:
            return time, path
        for connection in graph.get_connections_from(stop):
            cid = connection.id
            if cid in start.unavailable_ids:
                continue
            if trip is None:
                if not connection.pickup_allowed:
                    continue
            elif trip != connection.trip_id:
                continue
            event = scene.events[cid]
            if event.canceled or not time <= event.departure <= event.arrival <= deadline:
                continue
            new_trip = connection.trip_id or f'connection:{cid}'
            next_path = path + (cid,)
            offer(connection.arr_stop, new_trip, event.arrival, next_path)
            if connection.arr_stop != destination:
                offer(connection.arr_stop, None, event.arrival + transfer, next_path)
    return deadline, ()
