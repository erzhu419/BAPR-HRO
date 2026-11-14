"""Ordinary dynamic relevance merging, on the unchanged V48 scheduler.

The dependency analysis walks the original recursive menu rather than using
the named shared-observation interface. A past departure clock is retained
only while its trip occurs in a candidate still available for a future score
or physical continuation. Entry evidence is never removed from the history.
"""
import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v46.oracle import structural, unknown
from experiments.reconstruction_v46.shared_oracle import shared_structure
from experiments.reconstruction_v49.bulk import IncrementalBulkOracle


def recursive_trip_dependencies(model, ride):
    """All physical and scoring inputs, including post-destination exits."""
    trips = {model._positions[cid, kind][0]
             for cid in ride.path for kind in ('departure', 'arrival')}
    for tail in ride.exits.values():
        if tail is not None:
            for child in tail.rides.values():
                trips.update(recursive_trip_dependencies(model, child))
    return frozenset(trips)


class RemainingDependencies:
    """A generic menu liveness index; it contains no observed outcomes."""

    def __init__(self, model, platform):
        self.cids = tuple(platform.rides)
        self.root_trips = {cid: model._positions[cid, 'departure'][0]
                           for cid in self.cids}
        self.supports = {cid: recursive_trip_dependencies(model, ride)
                         for cid, ride in platform.rides.items()}
        self.users = {cid: sum(1 << i for i, candidate in enumerate(self.cids)
                              if self.root_trips[cid] in self.supports[candidate])
                      for cid in self.cids}

    def project(self, mask, observed):
        return tuple((cid, clock) for cid, clock in observed
                     if self.users[cid] & mask)

    def describe(self):
        return dict(cids=list(self.cids),
                    candidate_trip_dependencies=[dict(cid=cid, trips=sorted(self.supports[cid]))
                                                 for cid in self.cids],
                    observed_root_users=[dict(cid=cid, candidates=[other for i, other in enumerate(self.cids)
                                                                  if self.users[cid] & (1 << i)])
                                         for cid in self.cids])


class OrdinaryRelevanceMixin:
    """Replace only shared-platform state keys, retaining all old kernels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relevance_profiles = {}
        self._ordinary_dependency_cache = {}
        self.stats.update(ordinary_shared_platforms=0, ordinary_projection_calls=0,
                          ordinary_clock_records_in=0, ordinary_clock_records_kept=0,
                          ordinary_static_records_removed=0, ordinary_event_queries=0,
                          ordinary_event_hits=0, ordinary_policy_queries=0,
                          ordinary_policy_hits=0, ordinary_support_derivations=0)

    def _platform_node(self, platform, entry, history):
        # The independent-platform bulk implementation is identical in both
        # arms of the comparison; relevance merging applies to shared events.
        if structural(self.model, platform)['eligible']:
            return super()._platform_node(platform, entry, history)
        self._prepare(platform)
        context = self._key(platform, history), float(entry)
        key = ('platform', context)
        if key in self.nodes:
            self.stats['retained_node_hits'] += 1
            return self.nodes[key]
        if entry >= self.deadline:
            return self._point(key, self.deadline)
        check = shared_structure(self.model, platform)
        if not check['eligible']:
            self.stats['structure_refusals'] += 1
            return self._new(key, entry, constant=unknown(entry, self.deadline))
        static_interface = set(check['interface_cids'])
        self.stats['shared_interfaces_max'] = max(self.stats['shared_interfaces_max'], len(static_interface))
        structure = context[0][0]
        if structure not in self._ordinary_dependency_cache:
            dependency = RemainingDependencies(self.model, platform)
            self._ordinary_dependency_cache[structure] = dependency
            self.stats['ordinary_support_derivations'] += 1
            description = dependency.describe()
            description.update(stop=platform.stop, static_interface_cids=sorted(static_interface))
            self.relevance_profiles[structure] = description
        dependency = self._ordinary_dependency_cache[structure]

        def project(mask, z):
            kept = dependency.project(mask, z)
            self.stats['ordinary_projection_calls'] += 1
            self.stats['ordinary_clock_records_in'] += len(z)
            self.stats['ordinary_clock_records_kept'] += len(kept)
            self.stats['ordinary_static_records_removed'] += sum(
                cid in static_interface and not (dependency.users[cid] & mask) for cid, _ in z)
            return kept

        def build_platform():
            self.stats['shared_platform_solves'] += 1
            self.stats['ordinary_shared_platforms'] += 1
            cids = tuple(platform.rides)
            nc = len(cids)
            physical = {cid: self.root_law.departure_distribution(cid, history) for cid in cids}
            times = np.array(sorted({float(t) for cid in cids for t, p in zip(*physical[cid][:2], strict=True)
                                     if p > 0 and entry <= t < self.deadline}))
            hazards = np.zeros((len(times), nc))
            for i, cid in enumerate(cids):
                clocks, p, cancel = physical[cid]
                tails = np.r_[np.cumsum(p[::-1])[::-1], 0.]
                survival = cancel + p[clocks < entry].sum() + tails[np.searchsorted(clocks, times, side='left')]
                included = (p > 0) & (clocks >= entry) & (clocks < self.deadline)
                np.add.at(hazards[:, i], np.searchsorted(times, clocks[included]), p[included])
                np.divide(hazards[:, i], survival, out=hazards[:, i], where=survival > 0)
            ready = entry + self.graph.stops[platform.stop].min_transfer_time
            bits = [[i for i in range(nc) if mask & (1 << i)] for mask in range(1 << nc)]
            subsets = []
            for active in bits:
                choices = np.array([0], dtype=int)
                for i in active:
                    choices = np.r_[choices, choices | (1 << i)]
                subsets.append(choices)

            def event(j, mask, z):
                z = project(mask, z)
                event_key = ('event', context, j, mask, z)
                self.stats['ordinary_event_queries'] += 1
                self.stats['ordinary_event_hits'] += event_key in self.nodes
                if j == len(times) or not mask:
                    return self._point(event_key, self.deadline)
                clock = float(times[j])

                def build_event():
                    probability = np.ones(1)
                    for i in bits[mask]:
                        hazard = hazards[j, i]
                        probability = np.r_[probability * (1-hazard), probability * hazard]
                    children = []
                    order = sorted(range(len(probability)),
                                   key=lambda k: (-probability[k], int(subsets[mask][k])))
                    for k in order:
                        mass = float(probability[k])
                        if mass <= 0:
                            continue
                        departed = int(subsets[mask][k])
                        left = mask ^ departed
                        visible = tuple(cids[i] for i in bits[departed])
                        next_z = tuple(sorted((*z, *((cid, clock) for cid in visible))))
                        if not departed or clock < ready:
                            child = event(j+1, left, next_z)
                        else:
                            child = policy(j, mask, departed, left, visible, next_z)
                        children.append((mass, child))
                        self.stats['positive_event_branches'] += 1
                    return None, children

                return self._new(event_key, clock, build=build_event)

            def policy(j, mask, departed, left, visible, z):
                # Both current offers and remaining candidates are relevant
                # to the comparison. Waiting below projects again to left.
                z = project(mask, z)
                clock = float(times[j])
                policy_key = ('policy', context, j, mask, departed, z)
                self.stats['ordinary_policy_queries'] += 1
                self.stats['ordinary_policy_hits'] += policy_key in self.nodes

                def build_policy():
                    remaining = {cids[i] for i in bits[left]}
                    h = self.event_history(platform, history, entry, clock, remaining, visible, z)
                    seen = tuple(cid for cid in cids if cid not in remaining)
                    try:
                        chosen, _ = self.engine.platform(platform, entry, ready, h, visible, seen, clock,
                                                        choose=self.policy.choose, method='rolling_fixed')
                    except UnsupportedHistory:
                        self.stats['unsupported_scores'] += 1
                        return unknown(clock, self.deadline), []
                    child = (event(j+1, left, z) if chosen is None else
                             self._board_node(platform.rides[chosen], clock, history, h))
                    return None, [(1., child)]

                return self._new(policy_key, clock, build=build_policy, kind='shared_policy_compiles')

            return None, [(1., event(0, (1 << nc)-1, ()))]

        return self._new(key, entry, build=build_platform, kind='platform_solves')


class OrdinaryMergedOracle(OrdinaryRelevanceMixin, IncrementalBulkOracle):
    """Generic relevance quotient with V49 reduction and V48 scheduling."""
