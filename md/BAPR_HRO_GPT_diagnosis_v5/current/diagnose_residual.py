"""Exploratory Bellman-loss attribution of a selected natural OD2 residual."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from probe_small import (REPO,OUT,load_city,dated_graph,SegmentDurationBelief,
    options_from,ranks_from,TransferOpportunityScores,FullMenuOpportunity,
    FullMenuWaitingDP,SingleTargetOpportunity,CombinedWaitingOpportunity,
    PairOpportunity,ExactBoardCompiler,policy_values,public_offers,execute)


def main():
    args=argparse.ArgumentParser()
    args.add_argument('--mode',choices=['nominal','full'],default='nominal')
    args.add_argument('--root',default='2023-07-17_2_31200_origin')
    a=args.parse_args();started=perf_counter()
    source=REPO/'experiments/reconstruction_v25'
    selection=json.loads((source/'selection.json').read_text())
    root=next(r for g in selection['by_day'].values() for r in g['roots'] if r['root_id']==a.root)
    fit=json.loads((source/'model_fit.json').read_text())
    seg=json.loads((REPO/'experiments/reconstruction_v31/segment_fit.json').read_text())
    data,_=load_city(source/'city_transfer_endpoints.json.gz')
    graph,_,stop_ids,_,metadata,_=dated_graph(data,root['day'],'gtfs_station_max',True)
    model=SegmentDurationBelief(graph,duration_fit=fit['duration_fit'],segment_fit=seg,
        stop_identities={v:k for k,v in stop_ids.items()},public_occurrences=False,
        grid_step_seconds=20,**fit['increments'])
    params={c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
    options,ranks=options_from(root['full_options']),ranks_from(graph,root['full_options'])
    compiler=TransferOpportunityScores(graph,model,params,root['deadline'],options)
    ref=FullMenuOpportunity(compiler if a.mode=='nominal' else ExactBoardCompiler(compiler),
        ranks,root['start'],root['start']+30,root['deadline'])
    single=SingleTargetOpportunity(ref);combined=CombinedWaitingOpportunity(single)
    pair=PairOpportunity(combined);dp=FullMenuWaitingDP(ref)
    controls=dict(first=None,one=ref,single=single,combined=combined,pair=pair,dp=dp)
    exact,tables=policy_values(ref,dp,controls,return_tables=True)
    cids=ref.cids;n=len(cids);M=1<<n
    occupancy=np.zeros(M);occupancy[-1]=1.
    records=[];regret=0.;differing_mass=0.
    numeric=dict(changed_actions=0,changed_probability_mass=0.,max_abs_first_difference=0.,
        max_abs_combined_difference=0.,changed_bellman_weighted_loss=0.,worst_states=[])
    def direct_first(clock,seen):
        active=[c for c in cids if c not in seen]
        survival={c:ref.absent[c]+float(ref.distributions[c][1][ref.distributions[c][0]>clock].sum()) for c in active}
        active=[c for c in active if survival[c]>0]
        gain=0.;selected_total=0.
        for c in active:
            times,p,s=ref.marks[c];keep=times>clock
            times,p,s=times[keep],p[keep],s[keep]
            selected=p/survival[c]
            for other in active:
                if c==other:continue
                ot,op,_=ref.distributions[other]
                late=(ot>clock)
                before=ot[None,:] < times[:,None]
                if ref.ranks[other]<ref.ranks[c]:before |= (ot[None,:] == times[:,None])
                interval=(before & late[None,:]) @ op
                selected*=1.-interval/survival[other]
            gain+=selected @ (ref.deadline-s);selected_total+=float(selected.sum())
        return ref.deadline-gain,survival,selected_total
    subsets=[];bits=[]
    for mask in range(M):
        active=[i for i in range(n) if mask&(1<<i)]
        ids=np.array([0],dtype=int)
        for i in active:ids=np.r_[ids,ids|(1<<i)]
        bits.append(active);subsets.append(ids)
    for j,time in enumerate(dp.times):
        clock=int(time);legal=clock>=ref.ready
        next_mass=np.zeros(M);wait_cache={}
        score=np.full(n,float(ref.deadline))
        if legal:
            for i,cid in enumerate(cids):
                if dp.hazards[j,i]>0:score[i]=ref.score(cid,clock)
        board=np.full(M,float(ref.deadline));board_cid=[None]*M
        for mask in range(1,M):
            best=min(bits[mask],key=lambda i:(score[i],ref.ranks[cids[i]]))
            board[mask]=score[best];board_cid[mask]=cids[best]
        for mask,mass in enumerate(occupancy):
            if mass<=0:continue
            probability=np.ones(1)
            for i in bits[mask]:
                h=dp.hazards[j,i];probability=np.r_[probability*(1-h),probability*h]
            for present,p in zip(subsets[mask],probability):
                if p<=0:continue
                remaining=int(mask^present);weight=float(mass*p)
                if not present or not legal:
                    next_mass[remaining]+=weight;continue
                seen=tuple(cids[i] for i in range(n) if not remaining&(1<<i))
                if remaining not in wait_cache:
                    wait_cache[remaining]=(single.wait_score(clock,seen),ref.wait_score(clock,seen))
                sw,fw=wait_cache[remaining];cw=min(sw,fw)
                ew=float(dp.values[j+1,remaining]);b=float(board[present])
                take=b<=cw+1e-9
                recursive_fw=float(tables['first'][j+1,remaining])
                recursive_cw=min(sw,recursive_fw)
                stable_take=b<=recursive_cw+1e-9
                first_diff=abs(fw-recursive_fw)
                combined_diff=abs(cw-recursive_cw)
                numeric['max_abs_first_difference']=max(numeric['max_abs_first_difference'],first_diff)
                numeric['max_abs_combined_difference']=max(numeric['max_abs_combined_difference'],combined_diff)
                if take!=stable_take:
                    numeric['changed_actions']+=1
                    numeric['changed_probability_mass']+=weight
                    numeric['changed_bellman_weighted_loss']+=weight*((b if take else ew)-(b if stable_take else ew))
                if first_diff>1e-5:
                    direct,surv,total=direct_first(clock,seen)
                    item=dict(clock=clock,visible=[cids[i] for i in range(n) if present&(1<<i)],seen=seen,
                        state_probability=weight,first_deployed=fw,first_recursion=recursive_fw,
                        first_direct_interval_sum=float(direct),first_difference=first_diff,
                        direct_vs_recursion=abs(float(direct)-recursive_fw),single_wait=sw,
                        combined_deployed=cw,combined_recursion=recursive_cw,
                        combined_difference=combined_diff,board=b,action_changed=take!=stable_take,
                        survival=surv,direct_selected_probability=total)
                    numeric['worst_states'].append(item)
                if not take:next_mass[remaining]+=weight
                local=(b if take else ew)-min(b,ew)
                if local>1e-7:
                    visible=tuple(cids[i] for i in range(n) if present&(1<<i))
                    pw=pair.wait_score(clock,seen)
                    best_pair=min(((d.wait_score(clock,()),[cids[i],cids[k]]) for i,k,d in pair.pairs
                                  if cids[i] not in seen and cids[k] not in seen),default=(ref.deadline,[]))
                    records.append(dict(clock=clock,visible=visible,seen=seen,
                        remaining=[cid for cid in cids if cid not in seen],board_cid=board_cid[present],
                        board=b,single_wait=sw,first_wait=fw,combined_wait=cw,exact_wait=ew,
                        pair_wait=pw,best_pair=best_pair[1],best_pair_wait=best_pair[0],
                        state_probability=weight,conditional_loss=local,root_contribution=weight*local,
                        combined_action='BOARD' if take else 'WAIT',
                        dp_action='BOARD' if b<=ew+1e-9 else 'WAIT',
                        pair_action='BOARD' if b<=pw+1e-9 else 'WAIT'))
                    regret+=weight*local;differing_mass+=weight
        occupancy=next_mass
    expected=exact['expected_arrival']['combined']-exact['expected_arrival']['dp']
    assert abs(regret-expected)<1e-6,(regret,expected)
    offers=public_offers(metadata,tuple(options),root['start'],root['start']+30,root['deadline'])
    actual={m:execute(graph,metadata,root,options,ranks,offers,c) for m,c in controls.items()}
    out=dict(scope=__doc__,root_id=root['root_id'],mode=a.mode,exact=exact,
        exact_gaps={m:v-exact['expected_arrival']['dp'] for m,v in exact['expected_arrival'].items()},
        numerical_audit={**numeric,'worst_states':sorted(numeric['worst_states'],key=lambda x:x['first_difference'],reverse=True)[:20],
            'direct_max_error':max((s['direct_vs_recursion'] for s in numeric['worst_states']),default=0.),
            'cost_difference_coupling_bound':numeric['changed_probability_mass']*(root['deadline']-root['start'])},
        combined_bellman_regret=regret,decision_state_mass=differing_mass,
        top_contributions=sorted(records,key=lambda x:x['root_contribution'],reverse=True)[:30],
        remaining_no_board_probability=float(occupancy.sum()),
        all_regret_states=len(records),actual={m:{f:o[f] for f in ('action','tail','loss','arrived','status','trace')} for m,o in actual.items()},
        menu=root['full_options'],wall=perf_counter()-started)
    file=OUT/f'residual_{a.mode}_{a.root}.json';file.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps(dict(path=str(file),gaps=out['exact_gaps'],mass=differing_mass,top=out['top_contributions'][:3],wall=out['wall']),indent=2),flush=True)

if __name__=='__main__':main()
