"""Ordinary trip dependency projection for the fixed V52 reference.

The public observation contract and history objects remain unchanged. These
read sets identify independent factors that an internal expectation can sum
out and hard records that a query-local memo key need not distinguish. They
retain every recursive cheap-score input, including exits after a physical
destination. No posterior, policy value, or observed outcome selects a set.
"""

from experiments.reconstruction_v50.execute import ObservationContract


def history_key(history, trips):
    """Ordered hard evidence of live trips, within one frozen root query.

    Received forecasts, service notices and the root P0 basis are fixed by
    that query, not replaced by this key. Retained trips keep their complete
    hard evidence; a latest observation alone need not be sufficient.
    """
    return tuple((trip, history._records[trip]) for trip in sorted(trips)
                 if trip in history._records)


class RemainingReadSets:
    """Syntactic liveness for physical paths and the unchanged cheap scorer.

    At a platform event use all not-yet-departed action candidates. At an
    action comparison also include every current visible candidate; remove
    them only in the subsequent WAIT state. Relevant non-action observations
    are retained by trip, even when their cid never occurs in a ride path.
    """

    def __init__(self, model, observation_contract):
        self.model = model
        self.observation_contract = observation_contract
        self._rides, self._platforms, self._contracts = {}, {}, {}

    def ride_trips(self, ride, position=0):
        key = ride, position
        if key not in self._rides:
            path = ride.path[position:]
            trips = {self.model._positions[cid, kind][0]
                     for cid in path for kind in ('departure', 'arrival')}
            for cid in path:
                tail = ride.exits.get(cid)
                if tail is not None:
                    trips.update(self.platform_trips(tail))
            self._rides[key] = frozenset(trips)
        return self._rides[key]

    def platform_trips(self, platform, remaining_cids=None):
        remaining = (tuple(platform.rides) if remaining_cids is None
                     else tuple(cid for cid in platform.rides if cid in remaining_cids))
        key = platform, remaining
        if key not in self._platforms:
            self._platforms[key] = frozenset(
                trip for cid in remaining for trip in self.ride_trips(platform.rides[cid]))
        return self._platforms[key]

    def node_trips(self, node, *, position=0, remaining_cids=None):
        if hasattr(node, 'path'):
            return self.ride_trips(node, position)
        return self.platform_trips(node, remaining_cids)

    def observation_cids(self, stop, trips):
        """All local slots of live trips, including observation-only slots."""
        return tuple(cid for cid in self.observation_contract.cids_at(stop)
                     if self.model._positions[cid, 'departure'][0] in trips)

    def contract_for_trips(self, trips):
        trips = frozenset(trips)
        if trips not in self._contracts:
            self._contracts[trips] = ObservationContract({
                stop: self.observation_cids(stop, trips)
                for stop in self.observation_contract.station_to_cids})
        return self._contracts[trips]

    def contract_for(self, node, *, position=0, remaining_cids=None):
        """Internal projected sight; never replace the executed contract."""
        return self.contract_for_trips(self.node_trips(
            node, position=position, remaining_cids=remaining_cids))
