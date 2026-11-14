"""Constructive first-alighting compression for the unchanged V44 target.

Markov128 retains every original sampled world. Absorbing integrates the root
physical chain exactly while querying the same policy's working belief. Both
require the current physical trip to be absent from all exit tail menus.
"""

from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import (
    CheckingWorld, PhysicalDecisionEngine, posterior_transition,
)
from experiments.reconstruction_v43.reference import (
    DualMeasureCombinedReference, PrefixExecutor, RootConditionalLaw,
    _loss_receipt, _moments, _point,
)
from experiments.reconstruction_v44 import matched


def structure_gate(model, node, row):
    """Check actual input structure, without seeing any future physical clock."""
    trip = model._positions[row['cid'], 'arrival'][0]
    positions = [model._positions[cid, kind] for cid in node.path for kind in ('departure', 'arrival')]
    continuous = (all(t == trip for t, _ in positions)
                  and all(right[1] == left[1]+1 for left, right in zip(positions, positions[1:])))
    repeated = [cid for cid, platform in node.exits.items()
                if platform is not None and trip in platform.trips(model)]
    return dict(eligible=continuous and not repeated, current_trip=trip,
                contiguous_slots=continuous, same_trip_tail_exits=repeated,
                scope='All original exits checked; no realized suffix or outcome selection')


class CanonicalDecisionEngine(PhysicalDecisionEngine):
    """Use a sufficient representative only at internal prefix decisions."""

    def __init__(self, planner, root_snapshot):
        super().__init__(planner, variant='frontier')
        self.root_snapshot = root_snapshot
        self.histories = {}
        self.stats.update(canonical_queries=0, canonical_hits=0, canonical_histories=0)

    def canonical_history(self, cid, clock):
        self.stats['canonical_queries'] += 1
        key = cid, float(clock)
        if key in self.histories:
            self.stats['canonical_hits'] += 1
        else:
            history = self.root_snapshot.clone()
            history.observe(cid, 'arrival', clock)
            self.histories[key] = history
            self.stats['canonical_histories'] += 1
        return self.histories[key]

    def onboard(self, ride, cid, clock, history, method='rolling_fixed'):
        representative = self.canonical_history(cid, clock)
        return super().onboard(ride, cid, clock, representative, method=method)


class _Query:
    def __init__(self, graph, model, parameters, deadline, node, row, history):
        self.graph, self.model, self.parameters = graph, model, parameters
        self.deadline, self.node, self.row = deadline, node, row
        self.snapshot = freeze_snapshot(history, row['clock'])
        trips = sorted(node.trips(model))
        for trip in trips:
            self.snapshot.law(trip)
        self.root_law = RootConditionalLaw(self.snapshot, trips)
        self.planner = MatchedContinuationPlanner(graph, model, parameters, deadline)
        self.reference_planner = MatchedContinuationPlanner(graph, model, parameters, deadline)
        self.engine = CanonicalDecisionEngine(self.planner, self.snapshot)
        self.cache = {}
        self.cache_stats = dict(queries=0, hits=0, compiles=0)
        self.independence = dict(platform_occurrences=0, max_candidates=0, repeated_root_trips=0)
        self.reference_cpu = 0.

    def local_value(self, platform, entry):
        """Tail evidence is root evidence by the checked trip-disjoint gate."""
        started = process_time()
        try:
            trips = [self.model._positions[cid, 'departure'][0] for cid in platform.rides]
            self.independence['platform_occurrences'] += 1
            self.independence['max_candidates'] = max(self.independence['max_candidates'], len(trips))
            if len(trips) != len(set(trips)):
                raise ValueError('Local exact reference requires distinct physical root trips')
            structure = self.reference_planner.structure(platform)
            key = structure, float(entry)
            self.cache_stats['queries'] += 1
            if key in self.cache:
                self.cache_stats['hits'] += 1
                return self.cache[key]
            physical = {cid: self.root_law.departure_distribution(cid, self.snapshot)
                        for cid in platform.rides}
            ready = entry+self.graph.stops[platform.stop].min_transfer_time
            try:
                reference = self.reference_planner.platform_reference(platform, self.snapshot,
                    entry, ready, 'rolling_fixed')
                value = DualMeasureCombinedReference(reference, physical).receipt()
            except UnsupportedHistory as error:
                value = dict(lower=entry, upper=self.deadline, mean=None,
                    unresolved_probability=1., counters={}, reason=str(error))
            value = _loss_receipt(value, self.row['clock'])
            self.cache[key] = value
            self.cache_stats['compiles'] += 1
            return value
        finally:
            self.reference_cpu += process_time()-started


def _paired(rows):
    left, right = rows['alight'], rows['continue']
    differences = [None if a['value']['mean'] is None or b['value']['mean'] is None
        else a['value']['mean']-b['value']['mean'] for a, b in zip(left, right, strict=True)]
    moments = _moments(differences)
    complete = sum(value is not None for value in differences)
    return dict(orientation='alight remaining loss minus continue remaining loss',
        worlds=len(left), complete_pairs=complete, unresolved_pairs=len(left)-complete,
        mean_alight=_moments([r['value']['mean'] for r in left])['mean'],
        mean_continue=_moments([r['value']['mean'] for r in right])['mean'],
        mean_gain=moments['mean'], se=moments['se'],
        full_bank_mean_gain_interval=[
            float(np.mean([a['value']['lower']-b['value']['upper'] for a, b in zip(left, right, strict=True)])),
            float(np.mean([a['value']['upper']-b['value']['lower'] for a, b in zip(left, right, strict=True)]))],
        unresolved_probability={arm: float(np.mean([r['value']['unresolved_probability'] for r in values]))
                                for arm, values in rows.items()},
        interval_scope='Original finite prefix-bank local-proxy bounds; not a population certificate')


def _markov128(query, seed):
    tick, cpu = perf_counter(), process_time()
    bank = TripWorlds(query.snapshot.as_belief(), query.parameters, query.node.paths(), 128, seed)
    generation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-tick)
    public = query.snapshot.public_state()
    rows = {'alight': [], 'continue': []}
    prefix_cpu = 0.
    for index in range(128):
        for arm in rows:
            started = process_time()
            executor = PrefixExecutor(query.graph, query.planner, query.deadline,
                engine=query.engine, method='rolling_fixed', waiting='combined')
            outcome = executor.forced(query.node, query.row, arm, query.snapshot.clone(),
                                      CheckingWorld(bank, index, public))
            prefix_cpu += process_time()-started
            receipt = dict(world_index=index, prefix=dict(status=outcome['status'],
                trace=outcome['trace'], decisions=outcome['decisions']))
            if outcome['status'] != 'next_platform':
                value = (_point(outcome['loss']) if outcome['loss'] is not None else
                    dict(lower=0., upper=query.deadline-query.row['clock'], mean=None,
                         unresolved_probability=1., counters={}))
                receipt['prefix'].update(kind='terminal', entry=None)
            else:
                platform, entry = executor.reached['platform'], executor.reached['entry']
                receipt['prefix'].update(kind='platform', entry=entry,
                    ready=entry+query.graph.stops[platform.stop].min_transfer_time,
                    stop=platform.stop, node_structure=query.reference_planner.structure(platform))
                value = query.local_value(platform, entry)
            receipt['value'] = value
            rows[arm].append(receipt)
    return dict(rows=rows, paired=_paired(rows), generation=generation,
                prefix_process_cpu=prefix_cpu, count=128, generated_count=128,
                estimator='original128_canonical_prefix')


def _absorbing(query):
    started = process_time()
    node, row, model, deadline = query.node, query.row, query.model, query.deadline
    clock = row['clock']
    trip, first_slot = model._positions[row['cid'], 'arrival']
    law = query.root_law.roots[trip]
    atoms = {'alight': [], 'continue': []}
    counters = dict(transition_steps=0, positive_arrival_states=0, policy_queries=0,
                    platform_atoms=0, terminal_atoms=0, unresolved_policy_atoms=0)

    def absorb(arm, probability, kind, *, cid, entry=None, status=None, platform=None):
        if probability <= 0:
            return
        atom = dict(kind=kind, probability=float(probability), cid=cid, entry=entry, status=status)
        if kind == 'platform':
            atom['stop'] = platform.stop
            atom['node_structure'] = query.reference_planner.structure(platform)
            atom['value'] = query.local_value(platform, entry)
            counters['platform_atoms'] += 1
        elif kind == 'unresolved_prefix':
            atom['value'] = dict(lower=entry-clock, upper=deadline-clock, mean=None,
                                unresolved_probability=1., counters={})
            counters['unresolved_policy_atoms'] += 1
        else:
            atom['value'] = _point((entry if status == 'arrived' else deadline)-clock)
            counters['terminal_atoms'] += 1
        atoms[arm].append(atom)

    absorb('alight', 1., 'platform', cid=row['cid'], entry=clock,
           status='next_platform', platform=node.exits[row['cid']])
    mass = law.marginal(first_slot).copy()
    known = (law.times[first_slot] == clock) & ~law.canceled[first_slot]
    if not np.isclose(mass[known].sum(), 1., atol=1e-10, rtol=0):
        raise ValueError('Root current arrival must already be an exact public point mass')
    mass[~known] = 0.
    begin = node.path.index(row['cid'])+1
    for cid in node.path[begin:]:
        for kind in ('departure', 'arrival'):
            _, slot = model._positions[cid, kind]
            mass = mass @ posterior_transition(law, slot-1)
            counters['transition_steps'] += 1
            for state in np.flatnonzero(mass > 0):
                weight, t = float(mass[state]), float(law.times[slot][state])
                if law.canceled[slot][state] or t >= deadline:
                    status = ('service_or_deadline_failure' if kind == 'departure'
                              else 'service_failure' if law.canceled[slot][state] else 'deadline')
                    absorb('continue', weight, 'terminal', cid=cid, entry=t, status=status)
                    mass[state] = 0.
                elif kind == 'arrival' and cid in node.exits:
                    counters['positive_arrival_states'] += 1
                    platform = node.exits[cid]
                    if platform is None:
                        absorb('continue', weight, 'terminal', cid=cid, entry=t, status='arrived')
                        mass[state] = 0.
                    else:
                        counters['policy_queries'] += 1
                        try:
                            alight, _ = query.engine.onboard(node, cid, t, query.snapshot,
                                                           method='rolling_fixed')
                        except UnsupportedHistory:
                            absorb('continue', weight, 'unresolved_prefix', cid=cid,
                                   entry=t, status='unsupported_policy_state')
                            mass[state] = 0.
                            continue
                        if alight:
                            absorb('continue', weight, 'platform', cid=cid, entry=t,
                                   status='next_platform', platform=platform)
                            mass[state] = 0.
    if mass.sum() > 0:
        absorb('continue', float(mass.sum()), 'terminal', cid=node.path[-1],
               entry=None, status='exhausted_ride')
    actions, full_bounds, conservation = {}, {}, {}
    for arm, values in atoms.items():
        conservation[arm] = float(sum(a['probability'] for a in values))
        lower = sum(a['probability']*a['value']['lower'] for a in values)
        upper = sum(a['probability']*a['value']['upper'] for a in values)
        unresolved = sum(a['probability']*a['value']['unresolved_probability'] for a in values)
        actions[arm] = dict(lower=float(lower), upper=float(upper),
            mean=float(lower) if unresolved == 0 else None, unresolved_probability=float(unresolved))
        bound_low, bound_high = 0., 0.
        for atom in values:
            if atom['kind'] == 'terminal':
                lo = hi = atom['value']['mean']
            else:
                lo, hi = atom['entry']-clock, deadline-clock
            bound_low += atom['probability']*lo
            bound_high += atom['probability']*hi
        full_bounds[arm] = dict(lower=float(bound_low), upper=float(bound_high))
    a, c = actions['alight'], actions['continue']
    gap = [a['lower']-c['upper'], a['upper']-c['lower']]
    full_gap = [full_bounds['alight']['lower']-full_bounds['continue']['upper'],
                full_bounds['alight']['upper']-full_bounds['continue']['lower']]
    paired = dict(orientation='alight remaining loss minus continue remaining loss',
        worlds=0, complete_pairs=None, unresolved_pairs=None,
        mean_alight=a['mean'], mean_continue=c['mean'],
        mean_gain=None if a['mean'] is None or c['mean'] is None else a['mean']-c['mean'], se=None,
        full_bank_mean_gain_interval=gap,
        unresolved_probability={arm: value['unresolved_probability'] for arm, value in actions.items()},
        interval_scope='Exact population kappa bounds from full positive prefix support; no Monte Carlo interval')
    return dict(actions=actions, atoms=atoms, paired=paired,
        probability_conservation=conservation, absorption_counters=counters,
        full_pi_bounds=full_bounds, full_pi_difference_interval=full_gap,
        residual_interval=[full_gap[0]-gap[1], full_gap[1]-gap[0]],
        full_pi_bounds_scope=('Population bounds from physical terminal values and [entry,H] at first platform; '
            'unknown policy scores retain all bounded completions, not physical failure point values'),
        count=0, generated_count=0, estimator='exact_population_kappa',
        generation=dict(process_cpu=0., wall=0.),
        prefix_process_cpu=process_time()-started-query.reference_cpu)


def evaluate(graph, model, parameters, deadline, node, row, history, *,
             mode='markov128', seed, count=128):
    if mode not in ('markov128', 'absorbing'):
        raise ValueError(mode)
    if count != 128:
        raise ValueError('The finite comparator remains the original 128-world V44 query')
    began, cpu = perf_counter(), process_time()
    structural = structure_gate(model, node, row)
    if not structural['eligible']:
        result = matched.evaluate(graph, model, parameters, deadline, node, row, history,
                                  count=count, seed=seed)
        result.update(mode=mode, structural=structural, fallback='V44 original128 prefix; structural rejection',
                      estimator='structural_fallback_mc128', requested_count=count,
                      full_pi_bounds=None, full_pi_difference_interval=None, residual_interval=None)
        result['cost']['total_process_cpu'] = process_time()-cpu
        result['cost']['total_wall'] = perf_counter()-began
        return result
    tick, tick_cpu = perf_counter(), process_time()
    query = _Query(graph, model, parameters, deadline, node, row, history)
    preparation = dict(process_cpu=process_time()-tick_cpu, wall=perf_counter()-tick)
    result = _markov128(query, seed) if mode == 'markov128' else _absorbing(query)
    result.update(mode=mode, structural=structural, fallback=None, requested_count=count,
        seed=seed, future_messages=0, public_history=history.public_state(), clock=row['clock'], deadline=deadline,
        scope='Fixed V44 kappa target; canonical prefix score state, no new physical or message model',
        cache_stats=query.cache_stats, root_law_stats=query.root_law.stats,
        root_independence=query.independence,
        canonical_stats={key: value for key, value in query.engine.stats.items() if key.startswith('canonical_')})
    result['cost'] = dict(preparation=preparation, generation=result.pop('generation'),
        prefix_process_cpu=result.pop('prefix_process_cpu'), reference_process_cpu=query.reference_cpu,
        total_process_cpu=process_time()-cpu, total_wall=perf_counter()-began,
        prefix_planner_stats=query.planner.stats, reference_planner_stats=query.reference_planner.stats,
        prefix_decision_stats=query.engine.stats)
    return result
