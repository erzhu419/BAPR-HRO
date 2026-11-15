"""Common-trip random tapes and two ordinary wide-menu rollout references."""
from time import process_time, perf_counter
import numpy as np

from experiments.reconstruction_v32.journey import TripWorlds
from experiments.reconstruction_v34.gated_rollout import freeze_snapshot
from experiments.reconstruction_v41.continuation import empirical_pair_summary
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import CheckingWorld, PhysicalDecisionEngine
from experiments.reconstruction_v43.reference import PrefixExecutor, _point
from experiments.reconstruction_v45.prefix import _Query, _paired, structure_gate
from experiments.reconstruction_v50.execute import SnapshotExecutor


class StableTripWorlds(TripWorlds):
    """Global dated trip ordinal, root and world row identify every tape.

    The ordinal is from the full fixed graph, never the traversed menu. No
    hashing, resampling, menu-order RNG consumption or realized labels.
    """
    def __init__(self, belief, parameters, paths, count, seed, root_index, trip_universe):
        self.belief, self.count = belief, count
        self.events, self._paths = {}, {}
        self.trip_ranks = {trip:i for i,trip in enumerate(sorted(trip_universe))}
        self.used_trips = sorted({belief._positions[cid,'departure'][0] for path in paths for cid in path})
        for trip in self.used_trips:
            law=belief._law(trip,parameters)
            rng=np.random.default_rng(np.random.SeedSequence([seed,root_index,self.trip_ranks[trip]]))
            draws=law.draw_uniforms_batch(rng.random((count,len(law.times))))
            for slot,event in enumerate(belief._trips[trip]):
                selected=draws[:,slot]
                self.events[event.cid,event.kind]=law.times[slot][selected],law.canceled[slot][selected]


def platforms(node):
    if hasattr(node,'path'):
        for tail in node.exits.values():
            if tail is not None:yield from platforms(tail)
    else:
        yield node
        for ride in node.rides.values():yield from platforms(ride)


def evaluate(graph,model,parameters,deadline,node,row,history,contract,*,root_index,
             mode='full128',count=128,max_process_cpu=120.,seed=510015):
    tick,cpu=perf_counter(),process_time()
    snapshot=freeze_snapshot(history,row['clock'])
    paths=[*node.paths(),*contract.paths()]
    trips={model._positions[cid,'departure'][0] for path in paths for cid in path}
    for trip in sorted(trips):snapshot.law(trip)
    bank=StableTripWorlds(snapshot.as_belief(),parameters,paths,count,seed,root_index,model._trips)
    preparation=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
    public=snapshot.public_state()
    maximum=max((len(p.rides) for p in platforms(node)),default=0)
    # V45's local frozen Combined integral is exponential in menu width.
    # Retain the entire menu and use the ordinary full rollout when that
    # kernel is outside the declared finite-work admission rule.
    admitted=mode=='matched128' and maximum<=6 and structure_gate(model,node,row)['eligible']
    fallback=None if mode=='full128' or admitted else 'full128: V45 independent-prefix/bulk-width admission failed'
    query=_Query(graph,model,parameters,deadline,node,row,snapshot) if admitted else None
    planner=MatchedContinuationPlanner(graph,model,parameters,deadline)
    engine=PhysicalDecisionEngine(planner,variant='frontier')
    rows={'alight':[],'continue':[]}
    completed=0
    for index in range(count):
        if process_time()-cpu>=max_process_cpu:break
        for arm in rows:
            world=CheckingWorld(bank,index,public)
            executor=(PrefixExecutor(graph,query.planner,deadline,engine=query.engine,
                        method='rolling_fixed',waiting='combined') if admitted else
                      SnapshotExecutor(graph,planner,deadline,engine=engine,method='rolling_fixed',
                        waiting='combined',observation_contract=contract,trace=False))
            outcome=executor.forced(node,row,arm,snapshot.clone(),world)
            if admitted:
                if outcome['status']=='next_platform':
                    value=query.local_value(executor.reached['platform'],executor.reached['entry'])
                elif outcome['loss'] is not None:value=_point(outcome['loss'])
                else:value=dict(lower=0.,upper=deadline-row['clock'],mean=None,unresolved_probability=1.,counters={})
                rows[arm].append(dict(world_index=index,value=value,prefix=dict(status=outcome['status'])))
            else:
                rows[arm].append(dict(world_index=index,loss=outcome['loss'],status=outcome['status'],decisions=outcome['decisions']))
        completed+=1
    for index in range(completed,count):
        for arm in rows:
            rows[arm].append(dict(world_index=index,value=dict(lower=0.,upper=deadline-row['clock'],
                mean=None,unresolved_probability=1.,counters={}),prefix=dict(status='cpu_unresolved')) if admitted else
                dict(world_index=index,loss=None,status='cpu_unresolved',decisions={}))
    paired=(_paired(rows) if admitted else empirical_pair_summary(
        [r['loss'] for r in rows['alight']],[r['loss'] for r in rows['continue']],deadline-row['clock']))
    return dict(mode=mode,used_kernel='V45_canonical_prefix_kappa' if admitted else 'full_common_observation_pi',
        fallback=fallback,maximum_candidates=maximum,rows=rows,paired=paired,count=count,
        completed_pairs=completed,seed=seed,root_index=root_index,future_messages=0,
        common_tape=dict(trips=bank.used_trips,rank_universe_size=len(bank.trip_ranks),
            identity='fixed dated full-graph trip ordinal, root index, world row'),
        scope='Full mode evaluates original shallow pi with common local observations; matched kappa remains a frozen-child-mark proxy, not full pi. No sample interval is a population certificate.',
        cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick,preparation=preparation))
