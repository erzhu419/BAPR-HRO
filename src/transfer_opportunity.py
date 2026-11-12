"""Conditional values for one downstream transfer with first-offer recourse.

The menu fixes a same-vehicle first path and, at its endpoint, a menu of
same-vehicle tails. The traveler boards the first legal tail, breaking ties
by scheduled duration and connection ID. There is no arbitrary wait expiry.
"""

from dataclasses import dataclass
import numpy as np

from .risk_bounds import CommitmentBounds


@dataclass(frozen=True)
class RouteOption:
    """An empty tail menu means that ``first_path`` already reaches the goal."""

    first_path: tuple
    tails: dict


@dataclass(frozen=True)
class BoardScore:
    exact: float
    mean_arrival: float
    clock_nominal: float
    success_probability: float  # First-path completion only, before deadline clipping.
    mean_transfer_arrival: float
    method: str


class TransferOpportunityScores:
    """Cache independent-trip messages and values for a frozen route menu.

    ``mean_arrival`` replaces only the successful first-path arrival by its
    ceiling-rounded conditional mean. ``clock_nominal`` translates fixed
    marginal mean durations with the visible departure, retaining the actual
    first-departure distributions of the tail menu. It can be requested
    separately without preparing the exact scores.
    """

    def __init__(self, graph, belief, parameters, deadline, options):
        self.graph, self.belief, self.parameters = graph, belief, parameters
        self.deadline, self.options = deadline, dict(options)
        self.bounds = CommitmentBounds(belief, parameters, graph, None, deadline)
        self._departures, self._durations, self._tail_scores, self._tables = {}, {}, {}, {}
        root_trips = {self._path_slots(option.first_path)[0] for option in options.values()}
        for cid, option in self.options.items():
            if not option.first_path or option.first_path[0] != cid:
                raise ValueError('Each option is indexed by its first departure')
            end = graph.connections[option.first_path[-1]].arr_stop
            tails = [self._path_slots(path)[0] for path in option.tails.values()]
            if len(set(tails)) != len(tails) or root_trips.intersection(tails):
                raise ValueError('Tail trips must be distinct and independent of all root offer trips')
            if any(path[0] != first or graph.connections[first].dep_stop != end
                   for first, path in option.tails.items()):
                raise ValueError('Each tail must start at the first path endpoint')

    def _path_slots(self, path):
        if not path:
            raise ValueError('An offered same-vehicle path must be nonempty')
        positions = [self.belief._positions[cid, 'departure'] for cid in path]
        trip, start = positions[0]
        if positions != [(trip, slot) for slot in range(start, start + 2 * len(path), 2)]:
            raise ValueError('A path must consist of consecutive events on one vehicle')
        if any(self.graph.connections[a].arr_stop != self.graph.connections[b].dep_stop
               for a, b in zip(path, path[1:])):
            raise ValueError('A path must be physically continuous')
        return trip, start, start + 2 * len(path) - 1

    def departure_distribution(self, cid):
        """Return clock, unconditional operating mass, and cancellation mass."""
        if cid not in self._departures:
            trip, slot = self.belief._positions[cid, 'departure']
            law = self.bounds._law(trip)
            operating = ~law.canceled[slot]
            mass = law.marginal(slot)
            self._departures[cid] = (law.times[slot][operating], mass[operating],
                                     float(mass[~operating].sum()))
        return self._departures[cid]

    def _contract(self, path, values):
        """Backward contraction kills any canceled segment on the path."""
        trip, start, end = self._path_slots(path)
        law = self.bounds._law(trip)
        values = np.asarray(values, dtype=float)
        values = values * (~law.canceled[end])[:, None]
        for slot in range(end - 1, start - 1, -1):
            values = self.bounds._posterior_edge(trip, slot) @ values
            values *= (~law.canceled[slot])[:, None]
        return values

    def _duration(self, path):
        path = tuple(path)
        if path not in self._durations:
            trip, start, end = self._path_slots(path)
            law = self.bounds._law(trip)
            means = []
            for slot in (start, end):
                mass = law.marginal(slot) * ~law.canceled[slot]
                means.append(float(mass @ law.times[slot] / mass.sum()) if mass.sum() else self.deadline)
            self._durations[path] = max(0., means[1] - means[0])
        return self._durations[path]

    def _tail_marks(self, path, mode):
        key = tuple(path), mode
        if key not in self._tail_scores:
            clocks, mass, _ = self.departure_distribution(path[0])
            if mode == 'clock_nominal':
                values = np.minimum(self.deadline, clocks + self._duration(path))
            elif mode == 'exact':
                trip, start, end = self._path_slots(path)
                law = self.bounds._law(trip)
                gain = np.maximum(0., self.deadline - law.times[end])[:, None]
                values = self.deadline - self._contract(path, gain)[~law.canceled[start], 0]
            else:
                raise ValueError(f'Unknown tail scoring mode: {mode}')
            self._tail_scores[key] = clocks, mass, values
        return self._tail_scores[key]

    def tail_value(self, cid, ready, mode='exact'):
        """Expected first-legal-tail value at one or several ready clocks.

        For offer i at t, every other trip must not offer in [ready,t), nor
        tie at t with a better scheduled rank. Departures before arrival and
        cancellations remain in the complement; they are never conditioned
        away as if the passenger had witnessed them.
        """
        ready = np.asarray(ready)
        shape = ready.shape
        ready = ready.reshape(-1)
        option = self.options[cid]
        if not option.tails:
            return np.minimum(ready, self.deadline).reshape(shape)
        marks = {first: self._tail_marks(path, mode) for first, path in option.tails.items()}
        rank = {first: (self.graph.connections[path[-1]].arr_time
                        - self.graph.connections[first].dep_time, first)
                for first, path in option.tails.items()}
        gain = np.zeros(ready.size)
        for first, (clocks, mass, scores) in marks.items():
            legal = (clocks[None, :] >= ready[:, None]) & (clocks[None, :] < self.deadline)
            selected = np.broadcast_to(mass, legal.shape).copy() * legal
            for other, (other_clocks, other_mass, _) in marks.items():
                if other == first:
                    continue
                cdf = np.r_[0., np.cumsum(other_mass)]
                lower = cdf[np.searchsorted(other_clocks, ready, side='left')]
                side = 'right' if rank[other] < rank[first] else 'left'
                upper = cdf[np.searchsorted(other_clocks, clocks, side=side)]
                selected *= 1. - np.maximum(0., upper[None, :] - lower[:, None])
            gain += selected @ (self.deadline - scores)
        return (self.deadline - gain).reshape(shape)

    def clock_nominal_score(self, cid, clock):
        """Independent cheap duration control; does not compile exact scores."""
        option = self.options[cid]
        arrival = clock + self._duration(option.first_path)
        if not option.tails:
            return float(min(arrival, self.deadline))
        stop = self.graph.connections[option.first_path[-1]].arr_stop
        ready = np.ceil(arrival) + self.graph.stops[stop].min_transfer_time
        return float(self.tail_value(cid, ready, 'clock_nominal'))

    def _compile(self, cid):
        option = self.options[cid]
        trip, start, end = self._path_slots(option.first_path)
        law = self.bounds._law(trip)
        arrivals = law.times[end]
        moments = self._contract(option.first_path, np.column_stack((np.ones(len(arrivals)), arrivals)))
        success = moments[:, 0]
        means = np.divide(moments[:, 1], success, out=np.full_like(success, self.deadline), where=success > 0)
        if option.tails:
            mct = self.graph.stops[self.graph.connections[option.first_path[-1]].arr_stop].min_transfer_time
            terminal = self.tail_value(cid, arrivals + mct)
            at_mean = self.tail_value(cid, np.ceil(means) + mct)
        else:
            terminal = np.minimum(arrivals, self.deadline)
            at_mean = np.minimum(means, self.deadline)
        gain = self._contract(option.first_path, (self.deadline - terminal)[:, None])[:, 0]
        exact = self.deadline - gain
        mean_arrival = self.deadline - success * (self.deadline - at_mean)
        operating = (~law.canceled[start]) & (law.marginal(start) > 0)
        clocks = law.times[start][operating]
        nominal = [self.clock_nominal_score(cid, clock) for clock in clocks]
        self._tables[cid] = clocks, np.column_stack((exact[operating], mean_arrival[operating], nominal,
                                                    success[operating], means[operating]))

    def board_score(self, cid, clock):
        """Return an exact grid score or bracketed linear clock projection."""
        if cid not in self._tables:
            self._compile(cid)
        clocks, table = self._tables[cid]
        index = np.searchsorted(clocks, clock)
        if index < len(clocks) and clocks[index] == clock:
            row, method = table[index], 'on_grid'
        elif 0 < index < len(clocks):
            weight = (clock - clocks[index - 1]) / (clocks[index] - clocks[index - 1])
            row = (1. - weight) * table[index - 1] + weight * table[index]
            method = 'projection'
        else:
            return None
        return BoardScore(*(float(value) for value in row), method)
