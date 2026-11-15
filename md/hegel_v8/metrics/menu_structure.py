"""Inspect saved positive-support menus, without executing any policy."""
from pathlib import Path
import os,json,gzip,collections
from experiments.reconstruction_v34.discover import context,read
from experiments.reconstruction_v34.run import setup

ROOT=Path(os.environ.get('BAPR_HRO_REPO','/workspace/scratch/fde8f7d289c2/repos/BAPR-HRO-0022df2'))
OUT=Path(__file__).resolve().parent
V=ROOT/'experiments/reconstruction_v45';V34=ROOT/'experiments/reconstruction_v34'
data=read(V34/'data.json.gz');fit=read(V34/'model_fit.json');selection=read(V34/'selection.json')['selected']
contexts={day:context(data,day) for day in {r['day'] for r in selection}}
models={day:setup(c[0],fit)[0] for day,c in contexts.items()}
states=json.loads((OUT/'states.json').read_text())

def depths(n):
    if n is None:return [0]
    if n[0]=='platform':return [1+d for _,r in n[2] for d in depths(r)]
    return [d for _,child in n[3] for d in depths(child)]
def trips(n,g):
    if n is None:return set()
    if n[0]=='platform':return set().union(*(trips(r,g) for _,r in n[2]))
    return {g.connections[cid].trip_id for cid in n[2]}|set().union(*(trips(child,g) for _,child in n[3]))
def inspect(n,g,m,meta):
    candidates=[]
    for cid,r in n[2]:
        slots=[m._positions[c,k] for c in r[2] for k in ('departure','arrival')]
        roottrip=g.connections[cid].trip_id
        future=set().union(*(trips(child,g) for _,child in r[3]))
        candidates.append(dict(cid=cid,route=meta[cid]['route_name'],vehicle=meta[cid]['vehicle_number'],root_trip=roottrip,
            path=r[2],exits=[{'cid':c,'child_stop':child[1] if child is not None else None} for c,child in r[3]],
            pure_terminal=all(child is None for _,child in r[3]),
            same_trip=len({x[0] for x in slots})==1,
            contiguous=all(b[1]==a[1]+1 for a,b in zip(slots,slots[1:])),
            min_boardings=1+min(depths(r)),max_boardings=1+max(depths(r)),future_trips=sorted(future)))
    roottrips={r['root_trip'] for r in candidates}
    occurrences=collections.Counter(t for r in candidates for t in r['future_trips'])
    return dict(stop=n[1],station=g.stops[n[1]].name,candidates=candidates,
        independent_root_trips=len(roottrips)==len(candidates),
        pure_terminal_candidate_count=sum(r['pure_terminal'] for r in candidates),
        pure_terminal_platform=all(r['pure_terminal'] for r in candidates),
        same_trip_contiguous_candidates=all(r['same_trip'] and r['contiguous'] for r in candidates),
        min_boardings=min(depths(n)),max_boardings=max(depths(n)),
        unique_root_trips=len(roottrips),unique_future_trips=len(occurrences),
        future_trips_shared_between_candidates={t:n for t,n in occurrences.items() if n>1},
        root_trips_reused_as_future=sorted(roottrips & set(occurrences)))

records=[]
for s in states:
    root=selection[s['root_index']];g,_,_,_,meta,_=contexts[root['day']];m=models[root['day']]
    d=read(V/'banks'/f"{s['index']:03}_absorbing.json.gz")
    a=inspect(d['atoms']['alight'][0]['node_structure'],g,m,meta)
    record=dict(index=s['index'],root_index=s['root_index'],day=root['day'],changed=s['base_action']!=s['selected']['markov128'],alight=a)
    if record['changed']:
        grouped={}
        for at in d['atoms']['continue']:
            if at['kind']!='platform':continue
            key=json.dumps(at['node_structure'])
            if key not in grouped:grouped[key]=dict(probability=0.,**inspect(at['node_structure'],g,m,meta))
            grouped[key]['probability']+=at['probability']
        record['continue_platforms']=list(grouped.values())
    records.append(record)
(OUT/'menu_structure.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print('eligible terminal shape plus physical root independence and continuity',sum(r['alight']['pure_terminal_platform'] and r['alight']['independent_root_trips'] and r['alight']['same_trip_contiguous_candidates'] for r in records))
for r in records:
    if r['changed']:
        a=r['alight'];print(json.dumps(dict(index=r['index'],root=r['root_index'],station=a['station'],terminal_candidates=a['pure_terminal_candidate_count'],root_trips=a['unique_root_trips'],future_trips=a['unique_future_trips'],shared_future=a['future_trips_shared_between_candidates'],root_future_reuse=a['root_trips_reused_as_future'],max_boardings=a['max_boardings'],candidate_routes=[(x['cid'],x['route'],x['vehicle'],x['pure_terminal']) for x in a['candidates']]),ensure_ascii=False))
