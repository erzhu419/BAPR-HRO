"""Frozen-menu dependency audit; no natural policy value is computed."""
from collections import Counter
from pathlib import Path
import json

from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node
from experiments.reconstruction_v41.run import restored
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v53.relevance import RemainingReadSets

HERE=Path(__file__).resolve().parent
ANCHORS={4,6,18,32,57,62,63}


def structure(node):
    if node is None:return None
    if hasattr(node,'path'):
        return ('ride',tuple(node.path),tuple((cid,structure(p)) for cid,p in node.exits.items()))
    return ('platform',node.stop,tuple((cid,structure(r)) for cid,r in node.rides.items()))


def dependencies(node,model,history,start=0,rule='physical'):
    if node is None:return set()
    if not hasattr(node,'path'):
        return set().union(*(dependencies(r,model,history,rule=rule) for r in node.rides.values()))
    result=set()
    for cid in node.path[start:]:
        result.add(model._positions[cid,'departure'][0])
        if cid not in node.exits:continue
        child=node.exits[cid]
        if child is None:
            if rule=='physical' or (rule=='semantic' and not history.announced_unavailable(cid,'arrival')):break
        else:result.update(dependencies(child,model,history,rule=rule))
    return result


def platforms(node,start=0,physical=True):
    if not hasattr(node,'path'):
        yield node
        for ride in node.rides.values():yield from platforms(ride,physical=physical)
        return
    for cid in node.path[start:]:
        if cid not in node.exits:continue
        child=node.exits[cid]
        if child is None:
            if physical:break
        else:yield from platforms(child,physical=physical)


def rides(node):
    if hasattr(node,'path'):
        yield node
        for child in node.exits.values():
            if child is not None:yield from rides(child)
    else:
        for ride in node.rides.values():yield from rides(ride)


def terminal(ride,physical):
    for cid in ride.path:
        if cid not in ride.exits:continue
        if ride.exits[cid] is not None:return False
        if physical:return True
    return any(child is None for child in ride.exits.values())


def overlap(sets):
    counts=Counter(t for group in sets for t in group)
    shared={t for t,n in counts.items() if n>1}
    pending=set(range(len(sets)));sizes=[]
    while pending:
        component={pending.pop()};front=list(component)
        while front:
            i=front.pop();neighbors={j for j in pending if sets[i]&sets[j]}
            pending-=neighbors;component|=neighbors;front.extend(neighbors)
        sizes.append(len(component))
    return dict(shared_trips=sorted(shared),shared_count=len(shared),component_sizes=sorted(sizes,reverse=True))


def inspect(platform,model,history,reads):
    candidates=[]
    for cid,ride in platform.rides.items():
        score=set(reads.ride_trips(ride));phys=dependencies(ride,model,history)
        sem=dependencies(ride,model,history,rule='semantic')
        candidates.append(dict(cid=cid,trip=model._positions[cid,'departure'][0],score_trips=sorted(score),
            physical_trips=sorted(phys),semantic_trips=sorted(sem),physical_terminal=terminal(ride,True),
            score_terminal=terminal(ride,False),semantic_single_trip=len(sem)==1))
    score_overlap=overlap([set(c['score_trips']) for c in candidates])
    sem_overlap=overlap([set(c['semantic_trips']) for c in candidates])
    all_score=set().union(*(set(c['score_trips']) for c in candidates))
    all_sem=set().union(*(set(c['semantic_trips']) for c in candidates))
    obs=reads.observation_cids(platform.stop,all_score);sem_obs=reads.observation_cids(platform.stop,all_sem)
    extras=[cid for cid in obs if cid not in platform.rides]
    sem_extras=[cid for cid in sem_obs if cid not in platform.rides]
    roots={c['trip'] for c in candidates}
    eligible=(all(c['physical_terminal'] for c in candidates) and len(roots)==len(candidates) and not sem_extras)
    return dict(stop=platform.stop,candidate_count=len(candidates),candidates=candidates,
        distinct_root_trips=len(roots)==len(candidates),pure_physical_terminal=all(c['physical_terminal'] for c in candidates),
        pure_score_terminal=all(c['score_terminal'] for c in candidates),
        semantic_single_trip_candidates=all(c['semantic_single_trip'] for c in candidates),
        score_overlap=score_overlap,semantic_overlap=sem_overlap,
        relevant_observation_cids=list(obs),semantic_observation_cids=list(sem_obs),
        nonaction_observation_cids=extras,semantic_nonaction_observation_cids=sem_extras,
        nonaction_root_trip_cids=[cid for cid in extras if model._positions[cid,'departure'][0] in roots],
        terminal_dp_eligible=eligible)


def run():
    panel=read(HERE.parent/'reconstruction_v52/panel.json.gz')
    data=read(HERE.parent/'reconstruction_v34/data.json.gz');fit=read(HERE.parent/'reconstruction_v34/model_fit.json')
    emission=read(HERE.parent/'reconstruction_v35/fitted_emission.json');dated={};rows=[]
    for item in panel['rows']:
        state=item['state'];day=state['day']
        if day not in dated:
            graph=context(data,day)[0];model,params=setup(graph,fit);dated[day]=(graph,model,params)
        graph,model,params=dated[day];node=restore_node(state['node']);history=restored('shallow',model,params,emission,state)
        contract=ObservationContract({r['stop']:r['cids'] for r in item['context']['observation_contract']['station_to_cids']})
        reads=RemainingReadSets(model,contract);start=node.path.index(state['cid']) if state['kind']=='onboard' else 0
        physical=list(platforms(node,start));scored=list(platforms(node,start,False));unique={structure(p):p for p in physical}
        details=[inspect(p,model,history,reads) for p in unique.values()]
        score=set(reads.node_trips(node,position=start));phys=dependencies(node,model,history,start)
        sem=dependencies(node,model,history,start,rule='semantic')
        goals=[]
        for ride in rides(node):
            goal=next((cid for cid in ride.path if cid in ride.exits and ride.exits[cid] is None),None)
            if goal is None:continue
            later=ride.path[ride.path.index(goal)+1:]
            goals.append(dict(cid=goal,masked=history.announced_unavailable(goal,'arrival'),
                later_segments=len(later),later_exits=sum(cid in ride.exits for cid in later)))
        root_overlap=None
        if state['kind']=='onboard':
            a=set(reads.node_trips(node.exits[state['cid']]));c=set(reads.node_trips(node,position=start+1))
            sa=dependencies(node.exits[state['cid']],model,history,rule='semantic')
            sc=dependencies(node,model,history,start+1,rule='semantic')
            root_overlap=dict(alight_trips=len(a),continue_trips=len(c),shared_count=len(a&c),shared_trips=sorted(a&c),
                semantic_alight_trips=len(sa),semantic_continue_trips=len(sc),semantic_shared_count=len(sa&sc),semantic_shared_trips=sorted(sa&sc))
        rows.append(dict(query_index=item['query_index'],root_index=state['root_index'],kind=state['kind'],
            physical_platform_occurrences=len(physical),scored_platform_occurrences=len(scored),unique_physical_platforms=len(unique),
            root_score_trips=len(score),root_physical_trips=len(phys),root_semantic_trips=len(sem),
            root_score_only_trips=sorted(score-phys),root_semantically_dropped_trips=sorted(score-sem),
            root_action_overlap=root_overlap,goals=goals,platforms=details))
    plats=[p for row in rows for p in row['platforms']];goals=[g for row in rows for g in row['goals']]
    hist=lambda values:dict(sorted(Counter(values).items()))
    aggregate=dict(states=len(rows),roots=len({r['root_index'] for r in rows}),
        physical_platform_occurrences=sum(r['physical_platform_occurrences'] for r in rows),
        scoring_platform_occurrences=sum(r['scored_platform_occurrences'] for r in rows),unique_physical_platforms=len(plats),
        candidate_count=hist(p['candidate_count'] for p in plats),
        pure_physical_terminal=sum(p['pure_physical_terminal'] for p in plats),pure_score_terminal=sum(p['pure_score_terminal'] for p in plats),
        terminal_dp_eligible=sum(p['terminal_dp_eligible'] for p in plats),
        terminal_dp_candidate_count=hist(p['candidate_count'] for p in plats if p['terminal_dp_eligible']),
        queries_with_terminal_dp_platform=sum(any(p['terminal_dp_eligible'] for p in r['platforms']) for r in rows),
        semantic_single_trip_candidates=sum(p['semantic_single_trip_candidates'] for p in plats),
        shared_score_trip_count=hist(p['score_overlap']['shared_count'] for p in plats),
        shared_semantic_trip_count=hist(p['semantic_overlap']['shared_count'] for p in plats),
        score_terminal_candidate_count=hist(p['candidate_count'] for p in plats if p['pure_score_terminal']),
        physical_terminal_candidate_count=hist(p['candidate_count'] for p in plats if p['pure_physical_terminal']),
        platforms_with_related_nonaction_observations=sum(bool(p['nonaction_observation_cids']) for p in plats),
        semantic_platforms_with_related_nonaction_observations=sum(bool(p['semantic_nonaction_observation_cids']) for p in plats),
        goal_occurrences=len(goals),goal_masked=sum(g['masked'] for g in goals),early_goal_occurrences=sum(g['later_segments']>0 for g in goals),
        root_score_trips=sum(r['root_score_trips'] for r in rows),root_semantic_trips=sum(r['root_semantic_trips'] for r in rows),
        roots_with_semantic_drop=sum(r['root_semantic_trips']<r['root_score_trips'] for r in rows),
        root_shared_trip_count=hist(r['root_action_overlap']['shared_count'] for r in rows if r['root_action_overlap']),
        root_semantic_shared_trip_count=hist(r['root_action_overlap']['semantic_shared_count'] for r in rows if r['root_action_overlap']))
    result=dict(scope='Input-only syntactic dependency audit. Semantic sets are conditional on the unmasked first-goal dominance proof; they are not activated in the solver. Structural platforms are not positive-probability belief states. No value or policy occupancy computed.',aggregate=aggregate,rows=rows)
    (HERE/'structure.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(aggregate,indent=2))
    for r in rows:
        if r['root_index'] in ANCHORS:print('anchor',r['root_index'],r['query_index'],r['root_score_trips'],r['root_semantic_trips'],r['root_action_overlap'])


if __name__=='__main__':run()
