"""Bounded schedule-only discovery outside the original four OD pairs.

Selection accesses only static timetable fields. Actual metadata is loaded only
after the selected root catalog is saved. Development discovery, not independent
confirmation, not a city-wide prevalence estimate. Repository is read-only.
"""
import json
import random
import signal
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

REPO=Path('/workspace/scratch/fde8f7d289c2/repos/BAPR-HRO-86ba89c')
OUT=Path('/workspace/scratch/fde8f7d289c2/audit_86/algorithm')
sys.path.insert(0,str(REPO))
sys.path.insert(0,'/workspace/scratch/fde8f7d289c2/audit_86/current')

class BudgetExpired(Exception): pass
def expire(signum,frame): raise BudgetExpired('180-second discovery wall budget')

def main():
    begun=perf_counter()
    signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,180)
    result=dict(scope=__doc__,commit='86ba89c327e5230f1290e8678bebcb589b2465a9',
        protocol=dict(seed=20261112,day='2023-07-17',start=28500,deadline=30900,
            max_new_od=150,max_roots=12,min_K=2,max_K=6,
            minimum_first_routes=2,minimum_transfer_options=2,no_direct=True,
            law='segment n/(n+20)',board='full conditional fixed-route first-tail cost',
            selection_uses_actual=False,budget_seconds=180),
        stage='imports',funnel={},screened=[],selected=[],rows=[])
    def save():
        result['runtime_seconds']=perf_counter()-begun
        (OUT/'broad_probe.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    try:
        import experiments.reconstruction_v24.screen as screening
        from src.recourse_itinerary_router import build_trip_blocks
        from probe_small import (ExactBoardCompiler,PairOpportunity,policy_values,
            options_from,ranks_from,FullMenuOpportunity,FullMenuWaitingDP,
            SingleTargetOpportunity,CombinedWaitingOpportunity,TransferOpportunityScores,
            SegmentDurationBelief,load_city,dated_graph,public_offers,execute)
        source=REPO/'experiments/reconstruction_v25'
        old=json.loads((source/'selection.json').read_text())
        excluded={(r['stop'],r['goal']) for g in old['by_day'].values()
            for r in g['roots'] if r['root_kind']=='origin'}
        screening.DATES=('2023-07-17',)
        result['stage']='schedule_only_selection'
        static=screening.static_graphs(source/'city_transfer_endpoints.json.gz')['2023-07-17']
        blocks=build_trip_blocks(static)
        stops=sorted(static.stops)
        n=len(stops);rng=random.Random(20261112)
        seen=set();candidates=[]
        while len(candidates)<150:
            idx=rng.randrange(n*(n-1));a,b=divmod(idx,n-1)
            if b>=a:b+=1
            pair=stops[a],stops[b]
            if pair in excluded or pair in seen:continue
            seen.add(pair);candidates.append(pair)
        result['network']=dict(stops=n,connections=len(static.connections),
            excluded_old_pairs=sorted(excluded))
        funnel=Counter()
        for draw,(stop,goal) in enumerate(candidates):
            funnel['sampled_new_od']+=1
            catalog,counts=screening.build_menu(static,blocks,stop,goal,28500,28530,30900)
            menu,removed=screening.independent_menu(static,catalog)
            k=len(menu)
            routes=sorted({static.connections[o['first_cid']].route for o in menu})
            transfers=sum(o['transfer_stop'] is not None for o in menu)
            direct=sum(o['transfer_stop'] is None for o in menu)
            checks=dict(nonempty=bool(k),K_2_to_6=2<=k<=6,
                two_first_routes=len(routes)>=2,two_transfer_options=transfers>=2,no_direct=direct==0)
            for key,value in checks.items():funnel[key]+=int(value)
            eligible=all(checks.values())
            funnel['all_eligible']+=int(eligible)
            row=dict(draw=draw,stop=stop,goal=goal,origin_name=static.stops[stop].name,
                goal_name=static.stops[goal].name,K=k,first_routes=routes,
                transfer_options=transfers,direct_options=direct,checks=checks,
                menu_counts=counts,independence_removals=dict(removed))
            result['screened'].append(row)
            if eligible:
                root=dict(root_id=f'broad_2023-07-17_{draw}_{stop}_{goal}',day='2023-07-17',
                    root_kind='origin',stop=stop,goal=goal,start=28500,deadline=30900,
                    prefix=[],scheduled_entry=28500,scheduled_ready=28530,
                    origin_name=row['origin_name'],goal_name=row['goal_name'],full_options=menu,
                    draw=draw)
                result['selected'].append(root)
            if len(result['selected'])>=12:break
        result['funnel']=dict(funnel)
        result['selection_seconds']=perf_counter()-begun
        result['stage']='selection_frozen_before_actual_load'
        (OUT/'broad_probe_selection.json').write_text(json.dumps(dict(
            protocol=result['protocol'],network=result['network'],funnel=result['funnel'],
            screened=result['screened'],selected=result['selected']),indent=2)+'\n')
        save()
        if not result['selected']:
            result['status']='no_eligible_roots';save();return
        # Selection is now frozen; actual labels are permitted for evaluation.
        result['stage']='model_and_actual_loading'
        data,_=load_city(source/'city_transfer_endpoints.json.gz')
        graph,_,ids,_,metadata,_=dated_graph(data,'2023-07-17','gtfs_station_max',True)
        assert len(graph.connections)==len(static.connections)
        assert all((a.trip_id,a.dep_stop,a.arr_stop,a.dep_time,a.arr_time)==
                   (b.trip_id,b.dep_stop,b.arr_stop,b.dep_time,b.arr_time)
                   for a,b in zip(graph.connections,static.connections))
        fit=json.loads((source/'model_fit.json').read_text())
        seg=json.loads((REPO/'experiments/reconstruction_v31/segment_fit.json').read_text())
        model=SegmentDurationBelief(graph,duration_fit=fit['duration_fit'],segment_fit=seg,
            stop_identities={v:k for k,v in ids.items()},public_occurrences=False,
            grid_step_seconds=20,**fit['increments'])
        params={c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled'])
            for c in graph.connections}
        for root in result['selected']:
            if perf_counter()-begun>165:
                result['status']='budget_stopped_before_next_root';break
            tick=perf_counter();result['stage']='root_'+root['root_id'];save()
            options=options_from(root['full_options']);ranks=ranks_from(graph,root['full_options'])
            compiler=TransferOpportunityScores(graph,model,params,30900,options)
            reference=FullMenuOpportunity(ExactBoardCompiler(compiler),ranks,28500,28530,30900)
            single=SingleTargetOpportunity(reference);combined=CombinedWaitingOpportunity(single)
            controls=dict(first=None,one=reference,single=single,combined=combined)
            controls['pair']=PairOpportunity(combined)
            compiled=perf_counter();dp=FullMenuWaitingDP(reference);dp_wall=perf_counter()-compiled
            controls['dp']=dp
            integrated=perf_counter();exact=policy_values(reference,dp,controls)
            integration_wall=perf_counter()-integrated
            offers=public_offers(metadata,tuple(options),28500,28530,30900)
            actual={m:execute(graph,metadata,root,options,ranks,offers,control)
                for m,control in controls.items()}
            row=dict(root_id=root['root_id'],draw=root['draw'],origin_name=root['origin_name'],
                goal_name=root['goal_name'],K=len(options),exact=exact,actual=actual,
                root_gaps={m:v-exact['expected_arrival']['dp']
                    for m,v in exact['expected_arrival'].items()},
                compilation_wall=compiled-tick,dp_wall=dp_wall,
                integration_wall=integration_wall,total_wall=perf_counter()-tick)
            result['rows'].append(row);save()
            print(row['root_id'],row['origin_name'],'->',row['goal_name'],row['K'],
                row['root_gaps'],{m:o['loss'] for m,o in actual.items()},flush=True)
        else:result['status']='all_selected_roots_complete'
        result['stage']='complete';save()
    except BudgetExpired as exc:
        result['status']='budget_interrupted';result['interruption']=str(exc);save()
    except Exception as exc:
        result['status']='error';result['error']=repr(exc);save();raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        print('SAVED',OUT/'broad_probe.json',result.get('status'),
            len(result['rows']),len(result['selected']),perf_counter()-begun,flush=True)

if __name__=='__main__':main()
