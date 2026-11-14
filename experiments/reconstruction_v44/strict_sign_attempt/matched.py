"""Online V43 updated next-platform proxy, without diagnostic duplicate solves.

Each query generates its own original 128 V41 worlds. The fixed shallow
rolling/combined policy runs only to its first alighting platform, where a
frozen local Combined policy is integrated under the root-conditioned law.
No saved reference answer or future archive message is read here.
"""

from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v39.state import raw_history_key
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import CheckingWorld, PhysicalDecisionEngine
from experiments.reconstruction_v43.reference import (
    DualMeasureCombinedReference, PrefixExecutor, RootConditionalLaw,
    _loss_receipt, _moments, _point,
)


def evaluate(graph, model, parameters, deadline, node, row, history, *, count=128, seed):
    """Compute only V43 κ_updated, including every generated sample's mass."""
    if count != 128:
        raise ValueError('V44 fixes the original 128 paired worlds')
    began, cpu = perf_counter(), process_time()
    snapshot = freeze_snapshot(history, row['clock'])
    trips = sorted(node.trips(model))
    for trip in trips:
        snapshot.law(trip)
    root_law = RootConditionalLaw(snapshot, trips)
    bank = TripWorlds(snapshot.as_belief(), parameters, node.paths(), count, seed)
    preparation = dict(process_cpu=process_time()-cpu, wall=perf_counter()-began)
    tick, tick_cpu = perf_counter(), process_time()
    planner = MatchedContinuationPlanner(graph, model, parameters, deadline)
    reference_planner = MatchedContinuationPlanner(graph, model, parameters, deadline)
    engine = PhysicalDecisionEngine(planner, variant='frontier')
    setup = dict(process_cpu=process_time()-tick_cpu, wall=perf_counter()-tick)
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
                    value = dict(lower=0., upper=deadline-row['clock'], mean=None,
                                 unresolved_probability=1., counters={})
                else:
                    value = _point(outcome['loss'])
                receipt['prefix'].update(kind='terminal', entry=None)
                receipt['value'] = value
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
                value = cache[key]
            else:
                physical = {cid: root_law.departure_distribution(cid, local) for cid in platform.rides}
                try:
                    reference = reference_planner.platform_reference(platform, local,
                        entry, ready, 'rolling_fixed')
                    value = DualMeasureCombinedReference(reference, physical).receipt()
                except UnsupportedHistory as error:
                    value = dict(lower=entry, upper=deadline, mean=None,
                        unresolved_probability=1., counters={}, reason=str(error))
                value = _loss_receipt(value, row['clock'])
                cache[key] = value
                cache_stats['compiles'] += 1
            reference_cpu += process_time()-started
            receipt['value'] = value
            rows[action].append(receipt)
    left, right = rows['alight'], rows['continue']
    differences = [None if a['value']['mean'] is None or b['value']['mean'] is None
        else a['value']['mean']-b['value']['mean'] for a, b in zip(left, right, strict=True)]
    moments = _moments(differences)
    complete = sum(value is not None for value in differences)
    paired = dict(orientation='alight remaining loss minus continue remaining loss',
        worlds=count, complete_pairs=complete, unresolved_pairs=count-complete,
        mean_alight=_moments([r['value']['mean'] for r in left])['mean'],
        mean_continue=_moments([r['value']['mean'] for r in right])['mean'],
        mean_gain=moments['mean'], se=moments['se'],
        full_bank_mean_gain_interval=[
            float(np.mean([a['value']['lower']-b['value']['upper'] for a, b in zip(left, right, strict=True)])),
            float(np.mean([a['value']['upper']-b['value']['lower'] for a, b in zip(left, right, strict=True)]))],
        unresolved_probability={arm: float(np.mean([r['value']['unresolved_probability'] for r in values]))
                                for arm, values in rows.items()},
        interval_scope=('Finite prefix-bank bounds for unsupported local proxy scores; '
                        'not a confidence interval or full-policy-value certificate'))
    return dict(count=count, seed=seed, generated_count=count, future_messages=0,
        public_history=history.public_state(), clock=row['clock'], deadline=deadline,
        scope='V43 updated frozen local Combined proxy after fixed V41 prefixes; not full G_pi',
        mark_scope='MatchedContinuationPlanner platform marks at the legitimate next-platform history',
        rows=rows, paired=paired, cache_stats=cache_stats, root_law_stats=root_law.stats,
        root_independence=independence,
        cost=dict(preparation=preparation, planner_setup=setup,
            prefix_process_cpu=prefix_cpu, reference_process_cpu=reference_cpu,
            total_process_cpu=process_time()-cpu, total_wall=perf_counter()-began,
            prefix_planner_stats=planner.stats, reference_planner_stats=reference_planner.stats,
            prefix_decision_stats=engine.stats))


def choose(result, base):
    """Use strict paired interval sign; otherwise retain the current cheap action."""
    low, high = result['paired']['full_bank_mean_gain_interval']
    return 'alight' if high < 0 else 'continue' if low > 0 else base
