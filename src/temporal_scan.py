"""Batch world draws and exact chronological scans for the relaxed oracle."""

from collections import defaultdict
import ctypes
from math import log, sqrt
from pathlib import Path
import subprocess

import numpy as np

from .evolving_trip_belief import EvolvingTripBelief, install_chain_sampler


def build_temporal_kernel(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['c++', '-std=c++17', '-O3', '-shared', '-fPIC',
                    str(Path(__file__).with_suffix('.cpp')),
                    str(Path(__file__).with_name('chain_sampling.cpp')), '-o', str(path)], check=True)
    kernel = ctypes.CDLL(str(path.resolve()))
    install_chain_sampler(kernel)
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    kernel.temporal_arrival.argtypes = [ctypes.c_int, ctypes.c_int, i32, i32, i32, u8, u8,
                                       *[ctypes.c_int] * 6, i32]
    kernel.temporal_arrival.restype = ctypes.c_int
    return kernel


class TemporalScan:
    """Exact cost of ``earliest_arrival`` on the same relaxed graph/world."""

    def __init__(self, graph, kernel):
        self.kernel = kernel
        self.stops = {stop: i for i, stop in enumerate(graph.stops)}
        self.onboard = {}

        def seat(stop, trip):
            if not trip:
                return -1
            return self.onboard.setdefault((stop, trip), len(self.stops) + len(self.onboard))

        self.arcs = np.empty((len(graph.connections), 4), dtype=np.int32)
        for c in graph.connections:
            self.arcs[c.id] = (self.stops[c.dep_stop] if c.pickup_allowed else -1,
                              seat(c.dep_stop, c.trip_id), self.stops[c.arr_stop],
                              seat(c.arr_stop, c.trip_id))

    def root_labels(self, state, transfers):
        origin = self.stops[state.stop]
        onboard = self.onboard.get((state.stop, state.current_trip), -1)
        ready = state.time
        if state.current_trip is not None:
            ready = max(ready, state.last_arrival + int(transfers[origin]))
        return origin, onboard, ready

    def arrivals(self, departures, arrivals, canceled, state, goal, deadline, transfer=0):
        """Evaluate packed worlds with a scalar or per-stop transfer duration."""
        transfers = np.ascontiguousarray(np.broadcast_to(transfer, (len(self.stops),)), dtype=np.int32)
        return self._arrivals(departures, arrivals, canceled, state.stop, state.time,
            state.unavailable_ids, goal, deadline, transfers, self.root_labels(state, transfers))

    def platform_arrivals(self, departures, arrivals, canceled, state, goal, deadline, transfer=0):
        """Clairvoyant continuation with the platform's current transfer phase.

        Leaving a vehicle now starts transfer now; a passenger already on the
        platform retains the elapsed transfer phase. Staying aboard needs none.
        Subsequent transfers use optimistic immediate alighting as in arrivals.
        """
        transfers = np.ascontiguousarray(np.broadcast_to(transfer, (len(self.stops),)), dtype=np.int32)
        origin = self.stops[state.stop]
        onboard = self.onboard.get((state.stop, state.current_trip), -1)
        ready = state.time
        if state.current_trip is not None:
            ready += int(transfers[origin])
        elif state.last_connection is not None:
            ready = max(ready, state.last_arrival + int(transfers[origin]))
        return self._arrivals(departures, arrivals, canceled, state.stop, state.time,
            state.used_connections, goal, deadline, transfers, (origin, onboard, ready))

    def _arrivals(self, departures, arrivals, canceled, stop, time, used, goal, deadline,
                  transfers, root_labels):
        n = len(self.arcs)
        unavailable = np.zeros(n, dtype=np.uint8)
        unavailable[list(used)] = 1
        origin, onboard, ready = root_labels
        if stop == goal:
            return np.full(len(departures), time, dtype=np.int32)
        return np.array([self.kernel.temporal_arrival(n, len(self.stops) + len(self.onboard),
            self.arcs, dep, arr, cnl, unavailable, origin, onboard,
            self.stops[goal], time, ready, deadline, transfers)
            for dep, arr, cnl in zip(departures, arrivals, canceled)], dtype=np.int32)


def draw_chain_batch(law, uniforms):
    """Forward-filter backward-sample using the original per-trip uniforms."""
    return law.draw_uniforms_batch(uniforms)


def window_trips(belief, parameters, start, deadline):
    """Keep complete trip chains if an operating ride could intersect the window."""
    retained = set()
    for trip, slots in belief._trips.items():
        for slot in slots[::2]:
            departure = belief.hard_support(slot.cid, 'departure', parameters)
            if departure is None or departure[1] < start:
                continue
            arrival = belief.hard_support(slot.cid, 'arrival', parameters)
            if arrival is None:
                continue
            earliest_departure = max(start, departure[0])
            earliest_arrival = max(earliest_departure, arrival[0])
            if earliest_departure <= departure[1] and earliest_arrival <= min(deadline, arrival[1]):
                retained.add(trip)
                break
    return retained


def template_trip_map(belief, parameters):
    """Map unobserved chains to a relative-timetable representative and offset."""
    if type(belief) is not EvolvingTripBelief:
        raise TypeError('Translated templates require the ordered Gaussian trip model')
    representatives, templates = {}, {}
    connections = belief.graph.connections
    for trip, slots in belief._trips.items():
        if trip in belief._revision:
            continue
        start = slots[0].scheduled
        route = connections[slots[0].cid].route
        pars = parameters.get(route, {'mean': 1., 'std': 2., 'cancel_rate': .01})
        clocks = tuple(slot.scheduled - start for slot in slots)
        gaps = tuple(connections[slots[i - 1].cid].arr_stop != connections[slots[i].cid].dep_stop
                     for i in range(2, len(slots), 2))
        key = pars['mean'], pars['std'], pars['cancel_rate'], clocks, gaps
        representative, epoch = representatives.setdefault(key, (trip, start))
        templates[trip] = representative, start - epoch
    return templates


def sample_clocks(belief, parameters, rng, n, *, trips=None, templates=None):
    """Draw full ordered-trip worlds without constructing per-edge objects.

    Each trip receives the same variates as EvolvingTripBelief.sample. The
    clock arrays store potential clocks, ignored by the scan on cancellations.
    Skipped trips consume their original random tape and cannot be traversed.
    Trip templates batch the original independent tapes against a shared law.
    """
    shape = (n, len(belief.graph.connections))
    departures = np.zeros(shape, dtype=np.int32)
    arrivals = np.zeros(shape, dtype=np.int32)
    canceled = np.ones(shape, dtype=np.uint8)
    templates = {} if templates is None else templates
    for trip in belief._revision:
        belief._law(trip, parameters)
    draws = defaultdict(list)
    for trip, slots in belief._trips.items():
        uniforms = rng.random((n, len(slots)))[:, ::-1]
        if trips is not None and trip not in trips:
            continue
        representative, offset = templates.get(trip, (trip, 0))
        draws[representative].append((slots, uniforms, offset))
    for representative, group in draws.items():
        law = belief._law(representative, parameters)
        uniforms = group[0][1] if len(group) == 1 else np.concatenate([row[1] for row in group])
        grouped_states = draw_chain_batch(law, uniforms).reshape(len(group), n, uniforms.shape[1])
        for (slots, _, offset), states in zip(group, grouped_states):
            for index in range(0, len(slots), 2):
                cid = slots[index].cid
                departures[:, cid] = law.times[index][states[:, index]] + offset
                arrivals[:, cid] = law.times[index + 1][states[:, index + 1]] + offset
                canceled[:, cid] = law.canceled[index][states[:, index]]
    return departures, arrivals, canceled


def bounded_mean_lower(losses, cap, alpha):
    """Fixed-n IID lower confidence bound on the expectation of losses in [0, cap].

    Split alpha equally between Hoeffding and a nonnegative product bound.
    The expectation of the product is the nth power of the mean, so Markov
    bounds the probability that the product-derived lower bound exceeds it.
    """
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    beta = alpha / 2
    hoeffding = max(0., float(losses.mean()) - cap * sqrt(log(1. / beta) / (2 * n)))
    product = float(np.exp((log(beta) + np.log(losses).sum()) / n)) if np.all(losses > 0) else 0.
    return dict(hoeffding=hoeffding, product=product, lower=max(hoeffding, product))
