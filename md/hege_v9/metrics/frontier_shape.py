"""Static residual-menu eligibility only; no new policy/value computation."""
import json
import os
import sys
from pathlib import Path
REPO=Path(os.environ.get('BAPR_HRO_REPO','/workspace/scratch/fde8f7d289c2/repos/BAPR-HRO-6e46267'))
sys.path.insert(0,str(REPO))
from experiments.reconstruction_v34.discover import context,read
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node
from experiments.reconstruction_v33.nodes import Platform
from experiments.reconstruction_v46.oracle import structural

OUT=Path(__file__).resolve().parent
manifest=read(REPO/'experiments/reconstruction_v38/seed_manifest.json')
data=read(REPO/'experiments/reconstruction_v34/data.json.gz')
fit=read(REPO/'experiments/reconstruction_v34/model_fit.json')
records=[]
for index,root_index in [(120,62),(25,13)]:
    state=manifest[index]['state']
    graph=context(data,state['day'])[0]
    model,_=setup(graph,fit)
    ride=restore_node(state['node'])
    platform=ride.exits[state['cid']]
    cids=tuple(platform.rides)
    trace=read(OUT/'diagnostic_replay'/f'root{root_index}_frontier.json')
    fs=trace['checkpoints'][-1]['arms']['alight']['frontier']
    events=[f for f in fs if f['key_kind']=='event']
    checked=[]
    for f in events:
        mask=f['unseen_mask']
        keep=[cid for i,cid in enumerate(cids) if mask&(1<<i)]
        subset=Platform(platform.stop,{cid:platform.rides[cid] for cid in keep})
        checked.append(dict(frontier=f,remaining_cids=keep,eligible_shape=structural(model,subset)))
    records.append(dict(index=index,root=root_index,original_cids=cids,
        original_shape=structural(model,platform),original_entry=state['clock'],
        original_ready=state['clock']+graph.stops[platform.stop].min_transfer_time,
        event_frontier_count=len(events),eligible_count=sum(x['eligible_shape']['eligible'] for x in checked),
        eligible_width=sum(x['frontier']['width_contribution'] for x in checked if x['eligible_shape']['eligible']),
        eligible_mass=sum(x['frontier']['probability'] for x in checked if x['eligible_shape']['eligible']),
        events=checked))
output=dict(scope='Shape eligibility only. Original Ride objects and every scoring tail retained. No expectation, action or timing recomputed. Dynamic switch must separately preserve original entry/ready, all prior and absence evidence, seen IDs, P0 conditional law and W semantics.',
    nearest_existing_project_result='V39/V40 remaining menu reduction and relevant-history contexts. This is a possible fast-path interface for deterministic evaluation, not a new menu-reduction principle.',records=records)
(OUT/'frontier_shape.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
for r in records:print({k:v for k,v in r.items() if k!='events'})
