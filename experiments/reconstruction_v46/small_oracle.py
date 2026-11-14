"""Ordinary finite-world conditioning oracle for the unchanged complete policy.

Private event assignments are used only to integrate P0. The original executor
exposes its original public observations to the original working-belief policy.
Restarting an execution at each requested event is deliberately a small,
independent reference, not a proposed efficient online algorithm.
"""
from collections import defaultdict
from heapq import heappop, heappush
from itertools import count
from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v34.conditional_reference import ProjectionIncompatible
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v39.executor import SnapshotExecutor
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine
from experiments.reconstruction_v43.reference import RootConditionalLaw
from src.evolving_trip_belief import _ChainLaw


class NeedEvent(Exception):
    def __init__(self, trip, slot):
        self.trip, self.slot = trip, slot


class PrivateLaws:
    """Freeze root soft emissions; condition exact private chain state indices."""
    def __init__(self, roots):
        self.roots, self.cache = roots, {}
        self.queries = self.compiles = 0

    def probabilities(self, trip, slot, assigned):
        self.queries += 1
        evidence = tuple(sorted((s, state) for (t, s), state in assigned.items() if t == trip))
        key = trip, evidence
        if key not in self.cache:
            root = self.roots[trip]
            if not evidence:
                law = root
            else:
                emissions = [row.copy() for row in root.emissions]
                for observed_slot, state in evidence:
                    keep = np.arange(len(emissions[observed_slot])) == state
                    emissions[observed_slot] = np.where(keep, emissions[observed_slot], -np.inf)
                law = _ChainLaw(root.times, root.initial, root._transition,
                                emissions, root.canceled, root.stochastic)
                self.compiles += 1
            self.cache[key] = law
        return self.cache[key].marginal(slot)


class PrivateWorld:
    """A private future completion; never handed to a scoring function."""
    def __init__(self, model, roots, assigned):
        self.model, self.roots, self.assigned = model, roots, assigned

    def event(self, cid, kind):
        trip, slot = self.model._positions[cid, kind]
        if (trip, slot) not in self.assigned:
            raise NeedEvent(trip, slot)
        index, law = self.assigned[trip, slot], self.roots[trip]
        return int(law.times[slot][index]), bool(law.canceled[slot][index]), trip


def evaluate(graph, model, parameters, deadline, node, row, history, *, action='policy',
             max_evaluations=100000, keep_terminal_records=False):
    """Bound one complete-policy expected remaining loss under frozen P0.

    ``row.kind == 'platform_start'`` and action ``'policy'`` begin the original
    platform rule at row.entry (default row.clock). Other rows/actions use the
    original ``SnapshotExecutor.forced`` contract, including alight/continue.
    The evaluation limit counts every execution restart, not only completed
    worlds. Every unfinished or undefined branch retains [0, H-root_clock].
    """
    if max_evaluations < 0:
        raise ValueError('max_evaluations must be nonnegative')
    began, cpu = perf_counter(), process_time()
    clock = row['clock']
    snapshot = freeze_snapshot(history, clock)
    roots = RootConditionalLaw(snapshot, sorted(node.trips(model))).roots
    laws = PrivateLaws(roots)
    planner = MatchedContinuationPlanner(graph, model, parameters, deadline)
    engine = PhysicalDecisionEngine(planner, variant='frontier')
    serial = count()
    frontier = [(-1., next(serial), {})]
    resolved_mass = undefined_mass = weighted_loss = 0.
    evaluations = branches = completed_leaves = 0
    peak_frontier = 1
    status_mass, terminals = defaultdict(float), []
    while frontier and evaluations < max_evaluations:
        negative, _, assigned = heappop(frontier)
        probability = -negative
        executor = SnapshotExecutor(graph, planner, deadline, engine=engine,
            method='rolling_fixed', waiting='combined', trace=keep_terminal_records)
        world = PrivateWorld(model, roots, assigned)
        evaluations += 1
        try:
            if row['kind'] == 'platform_start':
                if action != 'policy':
                    raise ValueError('A platform_start evaluates its original policy')
                arrival, status = executor.platform(node, row.get('entry', clock), snapshot.clone(), world)
                result = dict(loss=arrival-clock, status=status, trace=executor.trace)
            else:
                result = executor.forced(node, row, action, snapshot.clone(), world)
        except NeedEvent as event:
            masses = laws.probabilities(event.trip, event.slot, assigned)
            for state in np.flatnonzero(masses > 0):
                child = dict(assigned)
                child[event.trip, event.slot] = int(state)
                heappush(frontier, (-probability*float(masses[state]), next(serial), child))
                branches += 1
            peak_frontier = max(peak_frontier, len(frontier))
            continue
        except (UnsupportedHistory, ProjectionIncompatible) as error:
            result = dict(loss=None, status=type(error).__name__+':'+str(error), trace=executor.trace)
        status_mass[result['status']] += probability
        if result['loss'] is None:
            undefined_mass += probability
        else:
            resolved_mass += probability
            weighted_loss += probability*result['loss']
        completed_leaves += 1
        if keep_terminal_records:
            terminals.append(dict(probability=probability, loss=result['loss'],
                status=result['status'], trace=result['trace']))
    pending_mass = float(sum(-item[0] for item in frontier))
    remaining = deadline-clock
    unresolved = pending_mass+undefined_mass
    result = dict(target='Full original rolling_fixed/combined pi under frozen root P0',
        method='Private-event conditioning enumeration; no policy-state compression',
        clock=clock, deadline=deadline, action=action,
        lower=float(weighted_loss), upper=float(weighted_loss+unresolved*remaining),
        mean=float(weighted_loss) if not frontier and undefined_mass == 0 else None,
        complete=not frontier and undefined_mass == 0, enumeration_exhausted=not frontier,
        resolved_mass=float(resolved_mass), pending_mass=pending_mass,
        undefined_mass=float(undefined_mass), unresolved_probability=float(unresolved),
        probability_mass=float(resolved_mass+unresolved), status_mass=dict(status_mass),
        evaluations=evaluations, max_evaluations=max_evaluations, branches=branches,
        completed_leaves=completed_leaves, pending_branches=len(frontier), peak_frontier=peak_frontier,
        private_law_queries=laws.queries, private_law_compiles=laws.compiles,
        generated_monte_carlo_worlds=0, future_messages=0,
        public_history=snapshot.public_state(),
        cost=dict(total_process_cpu=process_time()-cpu, total_wall=perf_counter()-began,
                  planner_stats=dict(planner.stats), decision_stats=dict(engine.stats)))
    if keep_terminal_records:
        result['terminals'] = terminals
    return result
