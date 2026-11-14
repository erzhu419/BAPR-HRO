"""V41 exact-anchor finite-chain conditioning for the frozen railway model.

This is an explicit model revision. Original grid points and Gaussian/service
parameters remain; exact integer-second observations refine the finite grid
using EvolvingTripBelief's anchor construction. Both planning and sampling
consume the resulting ordered conditional chain. No sampled clock is patched.
"""

from copy import copy

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from experiments.reconstruction_v37.controller import restore_single_snapshot
from experiments.reconstruction_v38.planner import FastPlanner
from experiments.reconstruction_v39.state import DecisionEngine
from src.evolving_trip_belief import EvolvingTripBelief, _ChainLaw, _TripMemo, logsumexp


MODEL_VERSION = 'V41_exact_integer_anchor_finite_chain_v1'


def posterior_transition(law, slot):
    """P(X[slot+1] | X[slot], all received evidence) on this law's support."""
    backward = law._backward_from(max(slot+1, law._last_effect), slot+1)
    weights = law._transition(slot)+law.emissions[slot+1]+backward
    normalizer = logsumexp(weights, axis=1, keepdims=True)
    return np.exp(weights-np.where(np.isfinite(normalizer), normalizer, 0.))


class PhysicalHistory(SingleMessageHistory):
    """The same public message selector, with internally exact own receipts.

    Supported source clocks are integer seconds, as in the frozen archive and
    trip-world generator. Contrary hard evidence raises UnsupportedHistory;
    neither conditioning nor sampling rejects and redraws individual worlds.
    """
    model_version = MODEL_VERSION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._physical_bases = {}

    def clone(self):
        result = super().clone()
        result._physical_bases = dict(self._physical_bases)
        return result

    def _physical_base(self, trip):
        observed = {(cid, kind): value for cid, kind, mode, value in self._records.get(trip, ())
                    if mode == 'observed'}
        for key, value in observed.items():
            if not float(value).is_integer():
                raise UnsupportedHistory(f'V41 source clock must be integer seconds: {key}={value}')
        base = self._physical_bases.get(trip)
        if base is not None and all(value in base.times[self.belief._positions[key][1]]
                                    for key, value in observed.items()):
            return base
        if type(self.belief) is not EvolvingTripBelief:
            raise TypeError('V41 is defined for the frozen V34 EvolvingTripBelief')
        model = copy(self.belief)
        model._known_times = {key: int(value) for key, value in observed.items()}
        model._excluded, model._forecasts = {}, {}
        model._cache = {trip: _TripMemo()}
        anchored = model._law(trip, self.parameters)
        # The basis fixes support and physical kernels, but carries no current
        # hard/message likelihood. In-grid observations condition this same
        # basis, including after clone(), rather than refining the grid again.
        base = _ChainLaw(anchored.times, anchored.initial, anchored._transition,
            [np.zeros(len(times)) for times in anchored.times], anchored.canceled, anchored.stochastic)
        self._physical_bases[trip] = base
        return base

    def law(self, trip):
        if trip not in self._laws:
            try:
                base = self._physical_base(trip)
                emissions = [np.zeros(len(times)) for times in base.times]
                for cid, kind, prediction, recorded_sd, receipt in self.forecast_records(trip):
                    slot = self.belief._positions[cid, kind][1]
                    emissions[slot] += self.forecast_log_emission(
                        cid, kind, prediction, recorded_sd, base.times[slot])
                for cid, kind, mode, value in self._records.get(trip, ()):
                    slot = self.belief._positions[cid, kind][1]
                    if mode == 'observed':
                        keep = ~base.canceled[slot] & (base.times[slot] == value)
                        emissions[slot] = np.where(keep, emissions[slot], -np.inf)
                    else:
                        for entry, clock in value:
                            keep = base.canceled[slot] | (base.times[slot] < entry) | (base.times[slot] > clock)
                            emissions[slot] = np.where(keep, emissions[slot], -np.inf)
                law = _ChainLaw(base.times, base.initial, base._transition,
                                emissions, base.canceled, base.stochastic)
                if not np.isfinite(logsumexp(law._full_forward(len(law.times)-1))):
                    raise ValueError('Public evidence leaves no positive ordered path')
            except ValueError as error:
                raise UnsupportedHistory(f'{MODEL_VERSION}: trip={trip}: {error}') from error
            law.model_version = MODEL_VERSION
            self._laws[trip] = law
        return self._laws[trip]

    def transition_kernel(self, trip, slot):
        return posterior_transition(self.law(trip), slot)

    def basis_context(self):
        """Process-local basis identity for exact execution-state comparisons."""
        return tuple((trip, id(base)) for trip, base in sorted(self._physical_bases.items()))

    def sample_vehicle(self, trip, uniforms):
        """Return exact clocks and availability using caller-supplied uniforms."""
        law = self.law(trip)
        indices = law.draw_uniforms_batch(np.asarray(uniforms))
        return dict(times=np.column_stack([times[indices[:, slot]]
                    for slot, times in enumerate(law.times)]),
                    canceled=np.column_stack([states[indices[:, slot]]
                    for slot, states in enumerate(law.canceled)]))

    def assert_physical_support(self, trip):
        """Diagnostic verification of support, not a filter on sampled paths."""
        law = self.law(trip)
        for slot in range(len(law.times)-1):
            positive = law.marginal(slot)[:, None]*posterior_transition(law, slot) > 0
            if np.any(positive & (law.times[slot+1][None, :] < law.times[slot][:, None])):
                raise AssertionError(f'Backward physical transition in {trip}:{slot}')
        for cid, kind, mode, value in self._records.get(trip, ()):
            if mode == 'observed':
                _, slot = self.belief._positions[cid, kind]
                mass = law.marginal(slot)
                if not np.isclose(mass[(law.times[slot] == value) & ~law.canceled[slot]].sum(), 1.):
                    raise AssertionError(f'Own receipt is not an internal point mass: {cid}:{kind}')
        return True


def restore_physical_snapshot(model, parameters, fit, public_state, clock):
    parsed = restore_single_snapshot(model, parameters, fit, public_state, clock)
    history = PhysicalHistory(model, parameters, fit=fit)
    for name in ('_records', '_forecasts', '_endpoint_statuses', '_trip_statuses', '_service_statuses'):
        setattr(history, name, getattr(parsed, name))
    history.message_clock = float(clock)
    return history


class PhysicalPlanner(FastPlanner):
    """Use the same anchor-refined law as the sampler for every transition.

    The old prior-future shortcut shares matrices on the original support.
    Exact anchors can change that support even at later slots, so this version
    uses the complete current trip key and never substitutes that old matrix.
    """
    model_version = MODEL_VERSION

    def _node_key(self, node, history, method):
        trips = self.relevant_trips(node)
        for trip in trips:
            history.law(trip)
        bases = tuple((trip, id(history._physical_bases[trip])) for trip in sorted(trips))
        return self.structure(node), history.signature(trips), method, bases

    def _trip_key(self, trip, history):
        history.law(trip)
        return trip, history.signature((trip,)), id(history._physical_bases[trip])

    def _transition_key(self, trip, left, history):
        return self._trip_key(trip, history), False


class PhysicalDecisionEngine(DecisionEngine):
    """Keep action-cache scope consistent with the physical planner's basis."""

    def _history(self, kind, node, history):
        public = super()._history(kind, node, history)
        trips = self.planner.relevant_trips(node)
        for trip in trips:
            history.law(trip)
        bases = tuple((trip, id(history._physical_bases[trip])) for trip in sorted(trips))
        return public, bases


class CheckingWorld:
    """Read sampled physical clocks; known receipts are checked, not overlaid."""

    def __init__(self, bank, index, public_state):
        self.bank, self.index = bank, index
        self.known = {(int(row['cid']), row['kind']): row['clock']
                      for row in public_state if row['evidence'] == 'observed'}

    def event(self, cid, kind):
        times, canceled = self.bank.events[cid, kind]
        value, unavailable = int(times[self.index]), bool(canceled[self.index])
        key = cid, kind
        if key in self.known and (unavailable or value != self.known[key]):
            raise AssertionError(f'Sampled law disagrees with exact public receipt {key}')
        return value, unavailable, self.bank.belief._positions[key][0]
