"""Read-only checks: root4 scheduled-early exclusion and V51/V52 panel mapping."""
import json,os,sys
from pathlib import Path
REPO=Path(os.environ['BAPR_HRO_REPO']).resolve();sys.path.insert(0,str(REPO))
from experiments.reconstruction_v34.discover import read,context
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v50.execute import ObservationContract
HERE=Path(__file__).resolve().parent


def main():
    data=read(REPO/'experiments/reconstruction_v34/data.json.gz')
    menu=read(REPO/'experiments/reconstruction_v50/menus/004.json.gz')['wide_root']
    graph,_,_,_,meta,_=context(data,menu['day']);node=root_ride(graph,menu)
    contract=ObservationContract.from_menu(node);platform=node.exits[111]
    cheap=read(REPO/'experiments/reconstruction_v51/roots/004_cheap.json.gz')
    full=read(REPO/'experiments/reconstruction_v51/roots/004_full128.json.gz')
    ready=cheap['states'][1]['ready']
    assert 145 not in platform.rides and 145 not in contract.cids_at(platform.stop)
    assert meta[145]['departure']['time_seconds']>=ready
    assert meta[145]['departure']['official_journey']==meta[193]['arrival']['official_journey']
    assert full['outcome']['trace'][-1]['clock']==meta[193]['arrival']['time_seconds']
    root4=dict(source='experiments/reconstruction_v50/menus/004.json.gz + V51 roots/004_cheap/full128.json.gz',
        root_index=4,first_onboard_cid=111,station=graph.stops[platform.stop].name,
        scheduled_ready=menu['hubs'][0]['scheduled_ready'],candidate145_scheduled_departure=graph.connections[145].dep_time,
        static_filter_reason='c.dep_time < ready in reconstruction_v32/screen.py:departures; V50 wide_root retains this filter',
        action_menu=list(platform.rides),local_observation_menu=list(contract.cids_at(platform.stop)),
        actual_entry=cheap['states'][0]['clock'],actual_ready=ready,
        missing_candidate=dict(cid=145,route_name=meta[145]['route_name'],trip=graph.connections[145].trip_id,
            active=meta[145]['departure']['status']=='REAL',actual_departure=meta[145]['departure']['time_seconds'],
            actual_feasibility_margin=meta[145]['departure']['time_seconds']-ready),
        same_physical_trip_path=[145,182,193],same_trip_destination_arrival=meta[193]['arrival']['time_seconds'],
        saved_rollout_destination_arrival=full['outcome']['trace'][-1]['clock'],
        scope='Counterfactual physical opportunity and domain omission, not proof that an expanded-menu cheap policy would select it or improve overall performance. No menu mutation or rerun performed.')
    a=read(REPO/'experiments/reconstruction_v51/roots/062_full128.json.gz')
    b=read(REPO/'experiments/reconstruction_v51/roots/062_snapshot_csa.json.gz')
    c=read(REPO/'experiments/reconstruction_v51/roots/062_cheap.json.gz')
    root62=dict(root_index=62,cheap_loss=c['outcome']['loss'],rollout_loss=a['outcome']['loss'],
        snapshot_csa_loss=b['outcome']['loss'],all_actions_identical=a['actions']==b['actions'],
        scope='Per-root comparison only; snapshot CSA does not dominate cheap over all72 roots.')
    (HERE/'positive_root_coverage.json').write_text(json.dumps(dict(root4=root4,root62=root62),ensure_ascii=False,indent=2)+'\n')
    panel=read(REPO/'experiments/reconstruction_v52/panel.json.gz')
    allstates=read(REPO/'experiments/reconstruction_v51/states.json.gz')['states']
    changed=json.loads((HERE/'existing_bank_signals.json').read_text())['changed'];rows=[]
    fields=('kind','root_index','cid','clock','public_history','node')
    for r in changed:
        j=read(REPO/f"experiments/reconstruction_v51/roots/{r['root_index']:03}_full128.json.gz")
        s=j['states'][r['state_index']]
        match=[q for q in panel['rows'] if all(q['state'].get(k)==s.get(k) for k in fields)]
        original=[q for q in allstates if all(q['state'].get(k)==s.get(k) for k in fields)]
        assert len(original)==1
        rows.append(dict(root_index=r['root_index'],cid=r['cid'],clock=r['clock'],gate_index=r['gate_index'],
            query_indices=[q['query_index'] for q in match],source_state_index=original[0]['index'],
            original_selected_actions=original[0]['selected_actions'],original_occurrences=original[0]['occurrences'],
            matched=len(match)==1))
    result=dict(scope='Exact comparison of kind/root_index/cid/clock/node/public_history between saved V51 changed full128 gates and frozen V52 panel.',
        changed_queries=len(rows),matched_queries=sum(r['matched'] for r in rows),
        unchanged_original_panel_size=len(panel['rows']),suggested_union_size=len(panel['rows'])+sum(not r['matched'] for r in rows),rows=rows)
    (HERE/'changed_gate_panel_mapping.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(root4=root4,root62=root62,panel_mapping={k:v for k,v in result.items() if k!='rows'}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
