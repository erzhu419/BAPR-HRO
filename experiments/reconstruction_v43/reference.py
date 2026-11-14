"""Next-platform attribution under V41's unchanged root physical law.

The executable prefix is the fixed shallow rolling/combined policy. At its
first alighting platform, local frozen-Combined proxies use either root or
prefix-updated policy scores. Their evaluation probabilities always remain
separate from those policy scores. This is exact local policy evaluation,
not exact evaluation of the later replanning policy.
"""

from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v39.executor import SnapshotExecutor
from experiments.reconstruction_v39.state import raw_history_key
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import CheckingWorld, PhysicalDecisionEngine
from src.evolving_trip_belief import _ChainLaw, logsumexp


class RootConditionalLaw:
    """Condition root emissions on new physical facts, without reselection."""

    def __init__(self, root_history, trips):
        self.model = root_history.belief
        self.roots = {trip: root_history.law(trip) for trip in trips}
        self._conditioned = {}
        self.stats = dict(law_queries=0, law_hits=0, law_compiles=0)

    def law(self, trip, history):
        self.stats['law_queries'] += 1
        records = history._records.get(trip, ())
        key = trip, records
        if key in self._conditioned:
            self.stats['law_hits'] += 1
            return self._conditioned[key]
        root = self.roots[trip]
        emissions = [row.copy() for row in root.emissions]
        for cid, kind, mode, value in records:
            slot = self.model._positions[cid, kind][1]
            if mode == 'observed':
                keep = ~root.canceled[slot] & (root.times[slot] == value)
            else:
                keep = np.ones(len(root.times[slot]), dtype=bool)
                for entry, clock in value:
                    keep &= (root.canceled[slot] | (root.times[slot] < entry)
                             | (root.times[slot] > clock))
            emissions[slot] = np.where(keep, emissions[slot], -np.inf)
        result = _ChainLaw(root.times, root.initial, root._transition,
                           emissions, root.canceled, root.stochastic)
        if not np.isfinite(logsumexp(result._full_forward(len(result.times)-1))):
            raise UnsupportedHistory(f'Prefix has no positive root-law probability: {trip}')
        self._conditioned[key] = result
        self.stats['law_compiles'] += 1
        return result

    def departure_distribution(self, cid, history):
        trip, slot = self.model._positions[cid, 'departure']
        return departure_distribution(self.law(trip, history), slot)


def departure_distribution(law, slot):
    mass = law.marginal(slot)
    operating = ~law.canceled[slot]
    return law.times[slot][operating], mass[operating], float(mass[~operating].sum())


class DualMeasureCombinedReference:
    """Evaluate a fixed local Combined rule under a separate physical law.

    The input reference determines all scores and waiting comparisons. The
    supplied independent departure distributions determine event probability.
    Unsupported visible scores retain their probability as [entry, deadline].
    Complexity O(T 3**K), storage O(T 2**K), for the current small menus.
    """

    def __init__(self, reference, physical_distributions):
        self.reference = reference
        self.cids = tuple(reference.cids)
        entry, ready, deadline = reference.entry, reference.ready, reference.deadline
        distributions = (reference.distributions, physical_distributions)
        self.times = np.array(sorted({float(t) for rows in distributions for cid in self.cids
            for t, p in zip(rows[cid][0], rows[cid][1], strict=True)
            if p > 0 and entry <= t < deadline}))
        ntime, ncid = len(self.times), len(self.cids)
        nstates = 1 << ncid
        hazards = []
        for rows in distributions:
            hazard = np.zeros((ntime, ncid))
            for i, cid in enumerate(self.cids):
                times, mass, cancel = rows[cid]
                times, mass = np.asarray(times), np.asarray(mass)
                tails = np.r_[np.cumsum(mass[::-1])[::-1], 0.]
                absent = cancel+mass[times < entry].sum()
                survival = absent+tails[np.searchsorted(times, self.times, side='left')]
                legal = (mass > 0) & (times >= entry) & (times < deadline)
                np.add.at(hazard[:, i], np.searchsorted(self.times, times[legal]), mass[legal])
                np.divide(hazard[:, i], survival, out=hazard[:, i], where=survival > 0)
            hazards.append(hazard)
        policy_hazard, physical_hazard = hazards
        scores = np.full((ntime, ncid), deadline)
        defined = np.ones((ntime, ncid), dtype=bool)
        single_wait = np.full_like(scores, deadline)
        for i, cid in enumerate(self.cids):
            times, mass, cancel = reference.distributions[cid]
            times, mass = np.asarray(times), np.asarray(mass)
            tails = np.r_[np.cumsum(mass[::-1])[::-1], 0.]
            after = cancel+mass[times < entry].sum()+tails[np.searchsorted(times, self.times, side='right')]
            clocks, probability, marks = reference.marks[cid]
            gains = np.r_[np.cumsum((probability*(deadline-marks))[::-1])[::-1], 0.]
            gain = gains[np.searchsorted(clocks, self.times, side='right')]
            single_wait[:, i] -= np.divide(gain, after, out=np.zeros_like(gain), where=after > 0)
            for j, t in enumerate(self.times):
                if t < ready:
                    continue
                try:
                    score = reference.score(cid, t)
                    if score is None or not np.isfinite(score):
                        defined[j, i] = False
                    else:
                        scores[j, i] = score
                except UnsupportedHistory:
                    defined[j, i] = False
        bits, subsets = [], []
        for mask in range(nstates):
            active = [i for i in range(ncid) if mask & (1 << i)]
            outcomes = np.array([0], dtype=int)
            for i in active:
                outcomes = np.r_[outcomes, outcomes | (1 << i)]
            bits.append(active)
            subsets.append(outcomes)
        first = np.full((ntime+1, nstates), deadline)
        lower = np.full_like(first, deadline)
        upper = np.full_like(first, deadline)
        unresolved = np.zeros_like(first)
        rank_order = sorted(range(ncid), key=lambda i: reference.ranks[self.cids[i]])
        transitions = 0
        for j in range(ntime-1, -1, -1):
            legal = self.times[j] >= ready
            best_board = np.full(nstates, deadline)
            ranked_board = np.full(nstates, deadline)
            missing = np.zeros(nstates, dtype=bool)
            wait = np.full(nstates, deadline)
            for mask in range(1, nstates):
                bit = mask & -mask
                i = bit.bit_length()-1
                best_board[mask] = min(best_board[mask ^ bit], scores[j, i])
                ranked_board[mask] = scores[j, next(i for i in rank_order if mask & (1 << i))]
                missing[mask] = missing[mask ^ bit] or not defined[j, i]
                wait[mask] = min(min(single_wait[j, i] for i in bits[mask]), first[j+1, mask])
            for mask in range(nstates):
                probability, physical_probability = np.ones(1), np.ones(1)
                for i in bits[mask]:
                    hp, hw = policy_hazard[j, i], physical_hazard[j, i]
                    probability = np.r_[probability*(1-hp), probability*hp]
                    physical_probability = np.r_[physical_probability*(1-hw), physical_probability*hw]
                departed = subsets[mask]
                remaining = mask ^ departed
                first[j, mask] = probability @ np.where(
                    legal & (departed != 0), ranked_board[departed], first[j+1, remaining])
                positive = physical_probability > 0
                departed, remaining = departed[positive], remaining[positive]
                physical_probability = physical_probability[positive]
                unknown = legal & missing[departed]
                board = legal & (departed != 0) & (best_board[departed] <= wait[remaining]+1e-9)
                lower[j, mask] = physical_probability @ np.where(unknown, entry,
                    np.where(board, best_board[departed], lower[j+1, remaining]))
                upper[j, mask] = physical_probability @ np.where(unknown, deadline,
                    np.where(board, best_board[departed], upper[j+1, remaining]))
                unresolved[j, mask] = physical_probability @ np.where(unknown, 1.,
                    np.where(board, 0., unresolved[j+1, remaining]))
                transitions += len(physical_probability)
        self.lower, self.upper = float(lower[0, -1]), float(upper[0, -1])
        self.unresolved_probability = float(unresolved[0, -1])
        self.value = self.lower if self.unresolved_probability == 0 else None
        self.counters = dict(events=ntime, masks=nstates, positive_event_outcomes=transitions)

    def receipt(self):
        return dict(lower=self.lower, upper=self.upper, mean=self.value,
                    unresolved_probability=self.unresolved_probability, counters=self.counters)


class PrefixExecutor(SnapshotExecutor):
    """Execute only until first alighting; never read its platform's events."""

    def platform(self, platform, entry, history, world, *, after=float('-inf')):
        self.reached = dict(platform=platform, entry=float(entry), history=history.clone())
        return entry, 'next_platform'


def _point(value):
    return dict(lower=float(value), upper=float(value), mean=float(value),
                unresolved_probability=0., counters={})


def _loss_receipt(receipt, root_clock):
    return {**receipt, 'lower': receipt['lower']-root_clock,
            'upper': receipt['upper']-root_clock,
            'mean': None if receipt['mean'] is None else receipt['mean']-root_clock}


def _moments(values):
    if any(value is None for value in values):
        return dict(n=len(values), mean=None, se=None)
    return dict(n=len(values), mean=float(np.mean(values)),
                se=float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None)


def evaluate_case(graph, model, parameters, deadline, node, row, history, *, count=128, seed):
    """Same 128 V41 worlds; fixed-policy prefixes and local dual-law proxies."""
    if count != 128:
        raise ValueError('V43 fixes the original 128 paired worlds')
    began, cpu = perf_counter(), process_time()
    snapshot = freeze_snapshot(history, row['clock'])
    trips = sorted(node.trips(model))
    for trip in trips:
        snapshot.law(trip)
    root_law = RootConditionalLaw(snapshot, trips)
    bank = TripWorlds(snapshot.as_belief(), parameters, node.paths(), count, seed)
    preparation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-began)
    planner = MatchedContinuationPlanner(graph, model, parameters, deadline)
    reference_planner = MatchedContinuationPlanner(graph, model, parameters, deadline)
    engine = PhysicalDecisionEngine(planner, variant='frontier')
    root_public = snapshot.public_state()
    cache = {}
    cache_stats = dict(queries=0, hits=0, compiles=0)
    independence = dict(platform_occurrences=0, max_candidates=0, repeated_root_trips=0)
    prefix_cpu, reference_cpu = 0., 0.
    rows = {'alight': [], 'continue': []}
    for index in range(count):
        for action in rows:
            started = process_time()
            executor = PrefixExecutor(graph, planner, deadline, engine=engine,
                method='rolling_fixed', waiting='combined')
            outcome = executor.forced(node, row, action, snapshot.clone(), CheckingWorld(bank, index, root_public))
            prefix_cpu += process_time()-started
            receipt = dict(world_index=index, prefix=dict(status=outcome['status'],
                trace=outcome['trace'], decisions=outcome['decisions']))
            if outcome['status'] != 'next_platform':
                if outcome['loss'] is None:
                    terminal = dict(lower=0., upper=deadline-row['clock'], mean=None,
                                    unresolved_probability=1., counters={})
                else:
                    terminal = _point(outcome['loss'])
                receipt['prefix'].update(kind='terminal', entry=None)
                receipt.update({level: dict(terminal) for level in ('root', 'updated', 'working')})
                rows[action].append(receipt)
                continue
            reached = executor.reached
            platform, entry, local = reached['platform'], reached['entry'], reached['history']
            root_trips = [model._positions[cid, 'departure'][0] for cid in platform.rides]
            independence['platform_occurrences'] += 1
            independence['max_candidates'] = max(independence['max_candidates'], len(root_trips))
            if len(root_trips) != len(set(root_trips)):
                raise ValueError('Local exact reference requires distinct physical root trips')
            ready = entry+graph.stops[platform.stop].min_transfer_time
            structure = reference_planner.structure(platform)
            receipt['prefix'].update(kind='platform', entry=entry, ready=ready, stop=platform.stop,
                                     node_structure=structure)
            started = process_time()
            relevant = reference_planner.relevant_trips(platform)
            key = structure, entry, raw_history_key(local, relevant)[0], tuple(
                (trip, id(local._physical_bases[trip])) for trip in sorted(relevant))
            cache_stats['queries'] += 1
            if key in cache:
                cache_stats['hits'] += 1
                local_results = cache[key]
            else:
                physical = {cid: root_law.departure_distribution(cid, local) for cid in platform.rides}
                working = {}
                for cid in platform.rides:
                    trip, slot = model._positions[cid, 'departure']
                    working[cid] = departure_distribution(local.law(trip), slot)
                local_results = {}
                for level, policy_history, measure in (
                        ('root', snapshot, physical), ('updated', local, physical), ('working', local, working)):
                    try:
                        reference = reference_planner.platform_reference(platform, policy_history,
                            entry, ready, 'rolling_fixed')
                        exact = DualMeasureCombinedReference(reference, measure)
                        result = exact.receipt()
                    except UnsupportedHistory as error:
                        result = dict(lower=entry, upper=deadline, mean=None,
                            unresolved_probability=1., counters={}, reason=str(error))
                    local_results[level] = _loss_receipt(result, row['clock'])
                cache[key] = local_results
                cache_stats['compiles'] += 1
            reference_cpu += process_time()-started
            receipt.update(local_results)
            rows[action].append(receipt)
    summaries = {}
    for level in ('root', 'updated', 'working'):
        summaries[level] = {}
        for action, values in rows.items():
            summaries[level][action] = {**_moments([r[level]['mean'] for r in values]),
                'lower': float(np.mean([r[level]['lower'] for r in values])),
                'upper': float(np.mean([r[level]['upper'] for r in values])),
                'unresolved_probability': float(np.mean([r[level]['unresolved_probability'] for r in values]))}
        left, right = rows['alight'], rows['continue']
        differences = [None if a[level]['mean'] is None or b[level]['mean'] is None
            else a[level]['mean']-b[level]['mean'] for a, b in zip(left, right, strict=True)]
        summaries[level]['difference'] = {**_moments(differences),
            'lower': float(np.mean([a[level]['lower']-b[level]['upper'] for a, b in zip(left, right, strict=True)])),
            'upper': float(np.mean([a[level]['upper']-b[level]['lower'] for a, b in zip(left, right, strict=True)]))}
    return dict(count=count, seed=seed, generated_count=count, future_messages=0,
        scope='Fixed V41 shallow rolling/combined prefix; exact frozen local proxies, not full G_pi',
        mark_scope='MatchedContinuationPlanner platform marks for both root and updated; not tail_base',
        public_history=history.public_state(), clock=row['clock'], deadline=deadline,
        orientation='alight remaining loss minus continue remaining loss',
        rows=rows, summary=summaries, cache_stats=cache_stats, root_law_stats=root_law.stats,
        root_independence=independence,
        cost=dict(preparation=preparation, prefix_process_cpu=prefix_cpu,
            reference_process_cpu=reference_cpu, total_process_cpu=process_time()-cpu,
            total_wall=perf_counter()-began, prefix_planner_stats=planner.stats,
            reference_planner_stats=reference_planner.stats, prefix_decision_stats=engine.stats))
