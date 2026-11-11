"""Product-factor information relaxation with a public transfer boundary."""

from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from math import isclose, prod

from .causal_prefix import causal_prefix_lower
from .simulator import LazyTripScenario


class _TripOverlay(Mapping):
    def __init__(self, common, changes):
        self.common, self.changes = common, changes

    def __getitem__(self, cid):
        return self.changes[cid] if cid in self.changes else self.common[cid]

    def __iter__(self):
        return iter(self.common)

    def __len__(self):
        return len(self.common)


def product_trip_bank(graph, common_scene, trip_samples):
    """Reveal the common rest and enumerate independent whole-trip atoms.

    ``trip_samples`` maps each hidden named trip to independently sampled event
    dictionaries, each covering that trip's complete graph chain. Draws across
    hidden factors and the common rest must be independent under the frozen
    root posterior. Identical sampled atoms remain separate empirical draws.
    """
    names = tuple(trip_samples)
    if not names or any(not trip_samples[name] for name in names):
        raise ValueError('Each hidden trip needs a nonempty independently sampled support')
    for name in names:
        expected = {c.id for c in graph.connections if c.trip_id == name}
        if not expected or any(set(events) != expected for events in trip_samples[name]):
            raise ValueError('Each factor atom must replace its whole named trip chain')
    combinations = tuple(product(*(range(len(trip_samples[name])) for name in names)))
    labels = {name: tuple(indices[i] for indices in combinations) for i, name in enumerate(names)}
    bank = []
    for indices in combinations:
        changes = {cid: event for name, index in zip(names, indices)
                   for cid, event in trip_samples[name][index].items()}
        scene = LazyTripScenario(_TripOverlay(common_scene.events, changes), minute=common_scene.minute,
                                 publications=common_scene.publications)
        bank.append((1. / len(combinations), scene))
    return bank, labels


def factored_prefix_lower(model, oracle_terminal, *, graph, factor_labels,
                          elapsed_horizon, max_nodes=50000):
    """Keep product-factor uncertainty until an actual change of vehicle.

    Use ``product_trip_bank`` to construct the platform model. The first
    successful ride on a trip different from the root incoming trip ends the
    causal prefix; ALIGHT alone and rides on the incoming trip do not. All
    physical actions and local observations are retained. The alternative
    boundary is the declared elapsed clock horizon.
    """
    names = tuple(factor_labels)
    count = len(model.scenarios)
    if not names or any(len(factor_labels[name]) != count for name in names):
        raise ValueError('Every product world needs one atom label for each hidden trip')
    combinations = tuple(zip(*(factor_labels[name] for name in names)))
    sizes = tuple(len(set(factor_labels[name])) for name in names)
    if len(set(combinations)) != count or prod(sizes) != count:
        raise ValueError('The bank must contain the complete Cartesian product of factor atoms')
    if any(not isclose(world.probability, 1. / count, abs_tol=1e-12) for world in model.scenarios):
        raise ValueError('This diagnostic uses uniformly sampled empirical factor supports')
    root = model.initial_state

    def changed_vehicle(state):
        return any((graph.connections[cid].trip_id or f'connection:{cid}') != root.current_trip
                   for cid in state.used_connections - root.used_connections)

    result = causal_prefix_lower(model, oracle_terminal, elapsed_horizon=elapsed_horizon,
        max_nodes=max_nodes, stop_when=changed_vehicle, factor_labels=factor_labels)
    kinds = {('changed_vehicle' if kind == 'public_boundary' else kind): mass
             for kind, mass in result.boundary_kind_mass.items()}
    return replace(result, boundary_kind_mass=kinds)
