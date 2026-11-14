"""Fixed-sample policy evaluation and ordinary same-world suffix reuse.

The suffix table lives for exactly one paired world. It retains complete raw
history, including currently unused forecasts; it never uses a current-action
cache key as a certificate of permanent meeting.
"""

from time import perf_counter, process_time

from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v33.execute import DrawnWorld
from experiments.reconstruction_v34.conditional_reference import summarize_pairs
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v39.executor import SnapshotExecutor
from experiments.reconstruction_v39.state import DecisionEngine, raw_history_key


class MeetingExecutor(SnapshotExecutor):
    """Reuse a completed suffix only at identical complete entry states."""
    def __init__(self, *args, suffixes, meeting_stats, **kwargs):
        super().__init__(*args, **kwargs)
        self.suffixes, self.meeting_stats = suffixes, meeting_stats

    def _suffix(self, phase, node, history, clock, extra, compute):
        began = process_time()
        basis = history.basis_context() if hasattr(history, 'basis_context') else ()
        key = (phase, self.planner.structure(node), float(clock), extra,
               raw_history_key(history)[0], history.message_clock,
               history.frame_index, basis)
        self.meeting_stats['checks'] += 1
        self.meeting_stats['check_cpu'] += process_time()-began
        if key in self.suffixes:
            value, trace, counts = self.suffixes[key]
            self.trace.extend(trace)
            for name, count in counts.items():
                self.counts[name] += count
            self.meeting_stats['hits'] += 1
            self.meeting_stats['avoided_decisions'] += sum(counts.values())
            return value
        start = len(self.trace)
        before = dict(self.counts)
        value = compute()
        self.suffixes[key] = (value, tuple(self.trace[start:]),
            {name: count-before[name] for name, count in self.counts.items()})
        return value

    def platform(self, platform, entry, history, world, *, after=float('-inf')):
        parent = super().platform
        return self._suffix('platform_entry', platform, history, entry, (after,),
            lambda: parent(platform, entry, history, world, after=after))

    def ride(self, ride, ready, history, world):
        parent = super().ride
        return self._suffix('ride_entry', ride, history, ready, (),
            lambda: parent(ride, ready, history, world))


def empirical_pair_summary(left, right, remaining):
    result = summarize_pairs(left, right, remaining)
    stats = result['complete_pair_statistics']
    if stats is not None and stats['n'] < 2:
        stats.update(se=None, ci95=None,
                     uncertainty='Not estimable from fewer than two complete pairs')
    return result


def evaluate(graph, model, parameters, deadline, node, row, history, *,
             planner_type, count, seed, meeting=False, world_type=DrawnWorld):
    """Evaluate both current actions under one fully specified base policy.

    The 16-world configuration uses the first 16 of the same 128 generated
    worlds. All 128 draws are charged. No CPU-dependent sample stopping occurs.
    """
    if count not in (16, 128):
        raise ValueError('V41 freezes 16 or 128 paired worlds')
    began, cpu = perf_counter(), process_time()
    snapshot = freeze_snapshot(history, row['clock'])
    for trip in sorted(node.trips(model)):
        snapshot.law(trip)
    conditioning = dict(wall=perf_counter()-began, process_cpu=process_time()-cpu)
    # Retain the original entire menu for random-number allocation.
    tick, tick_cpu = perf_counter(), process_time()
    bank = TripWorlds(snapshot.as_belief(), parameters, node.paths(), 128, seed)
    generation = dict(wall=perf_counter()-tick, process_cpu=process_time()-tick_cpu)
    tick, tick_cpu = perf_counter(), process_time()
    planner = planner_type(graph, model, parameters, deadline)
    if hasattr(snapshot, 'basis_context'):
        from experiments.reconstruction_v41.physical_posterior import PhysicalDecisionEngine
        engine = PhysicalDecisionEngine(planner, variant='frontier')
    else:
        engine = DecisionEngine(planner, variant='frontier')
    setup = dict(wall=perf_counter()-tick, process_cpu=process_time()-tick_cpu)
    rows = dict(alight=[], continue_=[])
    costs = {arm: dict(process_cpu=0., wall=0.) for arm in rows}
    stats = dict(checks=0, hits=0, avoided_decisions=0, check_cpu=0.)
    for index in range(count):
        suffixes = {}
        for arm, action in (('alight', 'alight'), ('continue_', 'continue')):
            tick, tick_cpu = perf_counter(), process_time()
            kwargs = dict(engine=engine, method='rolling_fixed', waiting='combined')
            if meeting:
                kwargs.update(suffixes=suffixes, meeting_stats=stats)
            executor = (MeetingExecutor if meeting else SnapshotExecutor)(
                graph, planner, deadline, **kwargs)
            outcome = executor.forced(node, row, action, snapshot.clone(), world_type(bank, index))
            outcome['world_index'] = index
            rows[arm].append(outcome)
            costs[arm]['process_cpu'] += process_time()-tick_cpu
            costs[arm]['wall'] += perf_counter()-tick
    rows['continue'] = rows.pop('continue_')
    costs['continue'] = costs.pop('continue_')
    paired = empirical_pair_summary([r['loss'] for r in rows['alight']],
        [r['loss'] for r in rows['continue']], deadline-row['clock'])
    paired['orientation'] = 'alight loss minus continue loss'
    return dict(seed=seed, count=count, generated_count=128,
        public_history=history.public_state(), future_messages=0,
        rows=rows, paired=paired, meeting=meeting, meeting_stats=stats,
        cost=dict(conditioning=conditioning, generation=generation, planner_setup=setup,
            continuations=costs, total_process_cpu=process_time()-cpu,
            total_wall=perf_counter()-began, planner_stats=dict(planner.stats),
            decision_stats=dict(engine.stats)))


def choose(result, base):
    low, high = result['paired']['full_bank_mean_gain_interval']
    return 'alight' if high < 0 else 'continue' if low > 0 else base


def science(result):
    return dict(seed=result['seed'], count=result['count'], paired=result['paired'],
        rows={arm: [{k: r[k] for k in ('loss', 'status', 'trace', 'decisions', 'world_index')}
                    for r in rows] for arm, rows in result['rows'].items()})
