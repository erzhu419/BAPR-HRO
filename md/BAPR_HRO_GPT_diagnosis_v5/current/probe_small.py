"""Read-only discovery: exact finite-offer values on every natural small menu.

No sampled worlds or realized-label selection. The laws, root menus and replay
are existing frozen objects. Newest segment law is default. Output is exploratory
model attribution and paired historical replay, not confirmation evidence.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph
from experiments.reconstruction_v24.run import options_from, ranks_from
from experiments.reconstruction_v25.run import physical_path, public_offers, execute
from experiments.reconstruction_v31.model import SegmentDurationBelief
from src.empirical_duration_belief import EmpiricalDurationBelief
from src.full_menu_opportunity import FullMenuOpportunity
from src.full_menu_waiting_dp import FullMenuWaitingDP
from src.plan_waiting_controls import SingleTargetOpportunity, CombinedWaitingOpportunity
from src.transfer_opportunity import TransferOpportunityScores

REPO = Path('/workspace/scratch/fde8f7d289c2/repos/BAPR-HRO-86ba89c')
OUT = Path('/workspace/scratch/fde8f7d289c2/audit_86/current')


class ExactBoardCompiler:
    def __init__(self, compiler):
        self.compiler = compiler

    def departure_distribution(self, cid):
        return self.compiler.departure_distribution(cid)

    def clock_nominal_score(self, cid, clock):
        value = self.compiler.board_score(cid, clock)
        if value is None:
            raise ValueError(('unsupported exact board', cid, clock))
        return value.exact


class PairOpportunity:
    """Rolling max-two-target backup, strengthened by the existing combined rule."""
    def __init__(self, combined):
        self.combined=combined
        self.reference=combined.reference
        self.cids=self.reference.cids
        self.pairs=[]
        for i,cid in enumerate(self.cids):
            for j,other in enumerate(self.cids[i+1:],i+1):
                ref=copy.copy(self.reference)
                ref.cids=(cid,other)
                ref.ranks={c:ref.ranks[c] for c in ref.cids}
                self.pairs.append((i,j,FullMenuWaitingDP(ref)))

    def wait_score(self,clock,seen):
        value=self.combined.wait_score(clock,seen)
        seen=set(seen)
        for i,j,dp in self.pairs:
            if self.cids[i] not in seen and self.cids[j] not in seen:
                value=min(value,dp.wait_score(clock,()))
        return value

    def choose(self,visible,seen,clock):
        wait=self.wait_score(clock,set(seen).union(visible))
        if not visible:return None,dict(wait=wait,board=None)
        values={cid:self.reference.score(cid,clock) for cid in visible}
        best=min(visible,key=lambda cid:(values[cid],self.reference.ranks[cid]))
        return (best if values[best]<=wait+1e-9 else None),dict(wait=wait,board=values[best],best_visible=best)


def policy_values(reference, dp, controls, return_tables=False):
    """Vectorized branch integration, independent of the DP min recursion.

First-offer policy's next-event value equals the reference first-future-offer
threshold. Single-target thresholds use the deployed control's implementation.
Check equality at every supported mask/clock to guard semantic drift.
"""
    times, cids = dp.times, dp.cids
    n, T = len(cids), len(times)
    M = 1 << n
    methods = ('first', 'one', 'single', 'combined', 'dp')+ (('pair',) if 'pair' in controls else ())
    v = {m: np.full((T+1, M), np.nan) for m in methods}
    for arr in v.values():
        arr[-1, dp.supported[-1]] = reference.deadline
    bits, subsets = [], []
    for mask in range(M):
        active = [i for i in range(n) if mask & (1 << i)]
        ids = np.array([0], dtype=int)
        for i in active:
            ids = np.r_[ids, ids | (1 << i)]
        bits.append(active)
        subsets.append(ids)
    max_threshold_error = 0.
    max_reference_recursion_difference = 0.
    # Individual target values can be evaluated via one mask per singleton,
    # then subset minima reproduce the controller's unrestricted target scan.
    all_cids = set(cids)
    mask_ids = np.arange(M)
    def deployed_first_wait(clock):
        """Vectorize the exact deployed CDF formula across all unseen masks."""
        survival = {cid:reference.absent[cid]+float(mass[times>clock].sum())
                    for cid,(times,mass,_) in reference.distributions.items()}
        active = [cid for cid in cids if survival[cid]>0]
        gain = np.zeros(M)
        for cid in active:
            clocks,mass,scores=reference.marks[cid]
            future=clocks>clock
            times=clocks[future]
            selected=mass[future]/survival[cid]
            factor=np.ones((M,len(times)))
            for other in active:
                if other==cid:continue
                other_times=reference.distributions[other][0]
                cdf=reference.cdfs[other]
                lower=cdf[np.searchsorted(other_times,clock,side='right')]
                side='right' if reference.ranks[other]<reference.ranks[cid] else 'left'
                upper=cdf[np.searchsorted(other_times,times,side=side)]
                factor[(mask_ids & (1<<cids.index(other)))!=0] *= 1.-np.maximum(0.,upper-lower)/survival[other]
            has=(mask_ids & (1<<cids.index(cid)))!=0
            gain[has] += factor[has] @ (selected*(reference.deadline-scores[future]))
        return reference.deadline-gain
    for j in range(T-1, -1, -1):
        clock = int(times[j]); legal = clock >= reference.ready
        scores = np.full(n, reference.deadline, dtype=float)
        for i, cid in enumerate(cids):
            if dp.hazards[j, i] > 0 and legal:
                scores[i] = reference.score(cid, clock)
        best = np.full(M, reference.deadline, dtype=float)
        first = np.full(M, reference.deadline, dtype=float)
        single_wait = np.full(M, reference.deadline, dtype=float)
        first_wait = deployed_first_wait(clock) if legal else np.full(M,reference.deadline,dtype=float)
        target_cost = [controls['single'].wait_score(clock, all_cids-{cid}) for cid in cids]
        for mask in range(1, M):
            bit = mask & -mask; i = bit.bit_length()-1
            best[mask] = min(best[mask ^ bit], scores[i])
            first_cid = min((cids[k] for k in bits[mask]), key=reference.ranks.__getitem__)
            first[mask] = scores[cids.index(first_cid)]
            single_wait[mask] = min(single_wait[mask ^ bit], target_cost[i])
        if 'pair' in controls:
            pair_wait=np.minimum(single_wait,first_wait)
            for i,k,pair in controls['pair'].pairs:
                both=(1<<i)|(1<<k)
                contains=(mask_ids & both)==both
                pair_wait[contains]=np.minimum(pair_wait[contains],pair.wait_score(clock,()))
        for mask in range(M):
            if not dp.supported[j, mask]:
                continue
            probability = np.ones(1)
            for i in bits[mask]:
                h = dp.hazards[j, i]
                probability = np.r_[probability*(1-h), probability*h]
            positive = probability > 0
            departed = subsets[mask][positive]
            remaining = mask ^ departed
            p = probability[positive]
            has = (departed != 0) & legal
            threshold = dict(one=first_wait[remaining],
                single=single_wait[remaining],
                combined=np.minimum(single_wait[remaining], first_wait[remaining]),
                dp=dp.values[j+1, remaining])
            if 'pair' in controls:threshold['pair']=pair_wait[remaining]
            for method in methods:
                branch = v[method][j+1, remaining].copy()
                if method == 'first':
                    branch[has] = first[departed[has]]
                else:
                    take = has & (best[departed] <= threshold[method]+1e-9)
                    branch[take] = best[departed[take]]
                if not np.all(np.isfinite(branch)):
                    raise AssertionError('unsupported feasible branch')
                v[method][j, mask] = p @ branch
        # Deterministic representative subset of masks validates the recursive
        # first-offer threshold against the production closed form.
        if legal:
            for mask in sorted({0, 1, M-1, M//2, (j*37) % M}):
                if not dp.supported[j+1, mask]:
                    continue
                seen = {cids[i] for i in range(n) if not mask & (1 << i)}
                closed = reference.wait_score(clock, seen)
                err = abs(first_wait[mask]-closed)
                max_threshold_error = max(max_threshold_error, err)
                assert err < 1e-6, ('first threshold', clock, mask, err)
                max_reference_recursion_difference=max(max_reference_recursion_difference,abs(v['first'][j+1,mask]-closed))
                assert abs(single_wait[mask]-controls['single'].wait_score(clock,seen)) < 1e-6
    supported = dp.supported
    err = float(np.max(np.abs(v['dp'][supported]-dp.values[supported])))
    assert err < 1e-6, ('independent dp integration', err)
    for m in methods:
        assert np.min((v[m]-dp.values)[supported]) >= -1e-6, ('feasible beats optimal',m)
    report=dict(expected_arrival={m:float(a[0,-1]) for m,a in v.items()},
        max_state_gap={m:float(np.max((a-dp.values)[supported])) for m,a in v.items()},
        checked_first_threshold_error=max_threshold_error, dp_integration_error=err,
        max_deployed_first_formula_recursion_difference=max_reference_recursion_difference,
        events=T, supported_states=int(supported.sum()))
    return (report,v) if return_tables else report


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--max-k',type=int,default=8)
    arg.add_argument('--min-k',type=int,default=1)
    arg.add_argument('--limit',type=int,default=999)
    arg.add_argument('--law',choices=['segment','route'],default='segment')
    arg.add_argument('--mode',choices=['nominal','full'],default='nominal')
    arg.add_argument('--day',default='')
    arg.add_argument('--root-prefix',default='')
    arg.add_argument('--pair',action='store_true')
    args=arg.parse_args()
    tick=perf_counter(); rows=[]
    source=REPO/'experiments/reconstruction_v25'
    fit=json.loads((source/'model_fit.json').read_text())
    seg=json.loads((REPO/'experiments/reconstruction_v31/segment_fit.json').read_text())
    selection=json.loads((source/'selection.json').read_text())
    data,_=load_city(source/'city_transfer_endpoints.json.gz')
    name=f'probe_{args.law}_{args.mode}_k{args.min_k}-{args.max_k}'
    if args.day:name+='_'+args.day
    if args.root_prefix:name+='_'+args.root_prefix
    if args.pair:name+='_pair'
    def save():
        (OUT/(name+'.json')).write_text(json.dumps(dict(scope=__doc__,arguments=vars(args),
            runtime=perf_counter()-tick,rows=rows),indent=2,allow_nan=False))
    for day,group in selection['by_day'].items():
        if args.day and day!=args.day:continue
        graph,_,stop_ids,_,metadata,_=dated_graph(data,day,'gtfs_station_max',True)
        kwargs=dict(public_occurrences=False,grid_step_seconds=20,**fit['increments'])
        if args.law=='segment':
            model=SegmentDurationBelief(graph,duration_fit=fit['duration_fit'],segment_fit=seg,
                stop_identities={v:k for k,v in stop_ids.items()},**kwargs)
        else:model=EmpiricalDurationBelief(graph,duration_fit=fit['duration_fit'],**kwargs)
        params={c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
        for root in group['roots']:
            if args.root_prefix and not root['root_id'].startswith(args.root_prefix):continue
            if not args.min_k<=len(root['full_options'])<=args.max_k:continue
            if len(rows)>=args.limit:break
            start=perf_counter()
            options,ranks=options_from(root['full_options']),ranks_from(graph,root['full_options'])
            entry,status=root['start'],'arrived'
            if root['prefix']:
                entry,status=physical_path(metadata,root['prefix'],root['start']+30,root['deadline'])
            if status!='arrived':
                rows.append(dict(root_id=root['root_id'],status=status));save();continue
            ready=entry+(max(30,graph.stops[root['stop']].min_transfer_time) if root['prefix'] else 30)
            compiler=TransferOpportunityScores(graph,model,params,root['deadline'],options)
            reference=FullMenuOpportunity(compiler if args.mode=='nominal' else ExactBoardCompiler(compiler),
                ranks,entry,ready,root['deadline'])
            single=SingleTargetOpportunity(reference);combined=CombinedWaitingOpportunity(single)
            controls=dict(first=None,one=reference,single=single,combined=combined)
            if args.pair:controls['pair']=PairOpportunity(combined)
            compiled=perf_counter();dp=FullMenuWaitingDP(reference);dp_time=perf_counter()-compiled
            controls['dp']=dp
            evaluated=perf_counter();exact=policy_values(reference,dp,controls);integration=perf_counter()-evaluated
            offers=public_offers(metadata,tuple(options),entry,ready,root['deadline'])
            actual={m:execute(graph,metadata,root,options,ranks,offers,c) for m,c in controls.items()}
            fields=('action','tail','loss','arrived','status','trace')
            actual={m:{f:o[f] for f in fields} for m,o in actual.items()}
            row=dict(root_id=root['root_id'],day=day,root_kind=root['root_kind'],K=len(options),
                start=root['start'],entry=entry,ready=ready,deadline=root['deadline'],exact=exact,actual=actual,
                compilation_wall=compiled-start,dp_wall=dp_time,integration_wall=integration,total_wall=perf_counter()-start)
            row['root_gaps']={m:v-exact['expected_arrival']['dp'] for m,v in exact['expected_arrival'].items()}
            rows.append(row);save()
            print(root['root_id'],len(options),row['root_gaps'],
                {m:o['loss'] for m,o in actual.items()},row['total_wall'],flush=True)
    print('SAVED',OUT/(name+'.json'),'total',perf_counter()-tick,flush=True)

if __name__=='__main__':main()
