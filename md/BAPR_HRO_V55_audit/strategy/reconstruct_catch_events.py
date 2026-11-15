"""Descriptive partition of the ORIGINAL V51 random tape, not fresh evaluation.

Regenerates the 128 original physical worlds from the saved public state and
seed, validates every original arm loss with the unchanged executor, and logs
actual simulated platform visitation. No new policy, tuning or random seed.
A catch event is feasibility of a named through vehicle, not proof that the
policy actually boards that vehicle.
"""
import json
import os
import sys
from pathlib import Path
from time import process_time,perf_counter
import numpy as np

REPO = Path(os.environ['BAPR_HRO_REPO']).resolve()
sys.path.insert(0,str(REPO))
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context,read
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v41.physical_posterior import restore_physical_snapshot,CheckingWorld,PhysicalDecisionEngine
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v50.execute import ObservationContract,SnapshotExecutor
from experiments.reconstruction_v51.rollout import StableTripWorlds

HERE = Path(__file__).resolve().parent
SPEC = {6:(145,134,182),18:(145,134,182),32:(169,146,209)}


class VisitExecutor(SnapshotExecutor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.platform_visits=[]

    def platform(self,platform,entry,history,world,**kwargs):
        self.platform_visits.append(dict(stop=platform.stop,entry=entry,
            ready=entry+self.graph.stops[platform.stop].min_transfer_time))
        return super().platform(platform,entry,history,world,**kwargs)


def main():
    data = read(REPO/'experiments/reconstruction_v34/data.json.gz')
    fit = read(REPO/'experiments/reconstruction_v34/model_fit.json')
    emission = read(REPO/'experiments/reconstruction_v35/fitted_emission.json')
    output=[]
    for root,(up,source_next,hub_dep) in SPEC.items():
        j=read(REPO/f'experiments/reconstruction_v51/roots/{root:03}_full128.json.gz')
        menu=read(REPO/f'experiments/reconstruction_v50/menus/{root:03}.json.gz')['wide_root']
        graph,_,_,_,metadata,_=context(data,j['day'])
        model,params=setup(graph,fit)
        node=root_ride(graph,menu)
        state=j['states'][0];gate=j['gates'][0]
        h=restore_physical_snapshot(model,params,emission,state['public_history'],state['clock'])
        assert h.public_state()==state['public_history']
        contract=ObservationContract.from_menu(node)
        paths=[*node.paths(),*contract.paths()]
        bank=StableTripWorlds(h.as_belief(),params,paths,128,gate['seed'],root,model._trips)
        assert bank.used_trips==gate['common_tape']['trips']
        assert len(bank.trip_ranks)==gate['common_tape']['rank_universe_size']
        up_times,up_cancel=bank.events[up,'departure']
        source_times,source_cancel=bank.events[source_next,'arrival']
        hub_times,hub_cancel=bank.events[hub_dep,'departure']
        up_stop=graph.connections[up].dep_stop
        hub_stop=graph.connections[hub_dep].dep_stop
        ready=state['clock']+graph.stops[up_stop].min_transfer_time
        up_catch=(~up_cancel)&(up_times>=ready)
        source_departures,source_departure_cancel=bank.events[source_next,'departure']
        source_arrival_feasible=(~source_cancel)&(~source_departure_cancel)&(source_times<menu['deadline'])
        up_catch=up_catch&(up_times<menu['deadline'])
        hub_active=(~hub_cancel)&(hub_times<menu['deadline'])
        hub_catch=hub_active&source_arrival_feasible&(hub_times>=source_times+graph.stops[hub_stop].min_transfer_time)
        ds=np.array([a['loss']-c['loss'] for a,c in zip(gate['rows']['alight'],gate['rows']['continue'])])
        started, cpu = perf_counter(),process_time()
        planner=MatchedContinuationPlanner(graph,model,params,menu['deadline'])
        engine=PhysicalDecisionEngine(planner,variant='frontier')
        policy_rows=[]
        for wi in range(128):
            arm_rows={}
            for arm in ('alight','continue'):
                executor=VisitExecutor(graph,planner,menu['deadline'],engine=engine,
                    method='rolling_fixed',waiting='combined',observation_contract=contract,trace=True)
                outcome=executor.forced(node,dict(kind='onboard',cid=state['cid'],clock=state['clock']),
                    arm,h.clone(),CheckingWorld(bank,wi,state['public_history']))
                saved=gate['rows'][arm][wi]
                assert outcome['loss']==saved['loss'], (root,wi,arm,outcome['loss'],saved['loss'])
                assert outcome['status']==saved['status'], (root,wi,arm,outcome['status'],saved['status'])
                boards=[dict(stop=t['stop'],clock=t['clock'],cid=t['choice']) for t in outcome['trace']
                        if t['kind']=='platform' and t['choice'] is not None]
                arm_rows[arm]=dict(loss=outcome['loss'],status=outcome['status'],platform_visits=executor.platform_visits,
                    boards=boards,boarded_named_upstream=any(b['cid']==up for b in boards),
                    visited_hub_as_platform=any(v['stop']==hub_stop for v in executor.platform_visits),
                    boarded_named_at_hub=any(b['cid']==hub_dep for b in boards))
            policy_rows.append(dict(world_index=wi,D_A_minus_C=int(ds[wi]),upstream_catch=bool(up_catch[wi]),
                hub_catch=bool(hub_catch[wi]),source_arrival_feasible=bool(source_arrival_feasible[wi]),
                up_canceled=bool(up_cancel[wi]),hub_canceled=bool(hub_cancel[wi]),
                up_departure=float(up_times[wi]),source_next_departure=float(source_departures[wi]),
                source_next_arrival=float(source_times[wi]),hub_departure=float(hub_times[wi]),
                source_hub_ready=float(source_times[wi]+graph.stops[hub_stop].min_transfer_time),arms=arm_rows))
        validation_cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-started)
        partitions=[]
        for uc in (False,True):
            for hc in (False,True):
                mask=(up_catch==uc)&(hub_catch==hc)
                count=int(mask.sum())
                partitions.append(dict(upstream_catch=uc,hub_catch=hc,count=count,
                    contribution_to_mean_A_minus_C=float(ds[mask].sum()/128),
                    conditional_mean_A_minus_C=float(ds[mask].mean()) if count else None,
                    actual_policy_usage={arm:{key:sum(r['arms'][arm][key] for r,m in zip(policy_rows,mask) if m)
                        for key in ('boarded_named_upstream','visited_hub_as_platform','boarded_named_at_hub')}
                        for arm in ('alight','continue')}))
        actual_ready=metadata[source_next]['arrival']['time_seconds']+graph.stops[hub_stop].min_transfer_time
        out=dict(root_index=root,source=f'experiments/reconstruction_v51/roots/{root:03}_full128.json.gz#gates/0',
            seed=gate['seed'],count=128,upstream=dict(cid=up,station=graph.stops[up_stop].name,ready=ready,
                actual_departure=metadata[up]['departure']['time_seconds'],
                actual_margin=metadata[up]['departure']['time_seconds']-ready,
                original_tape_catch_count=int(up_catch.sum()),
                marginal_catch_probability=float((h.law(model._positions[up,'departure'][0]).marginal(model._positions[up,'departure'][1]) *
                    ((~h.law(model._positions[up,'departure'][0]).canceled[model._positions[up,'departure'][1]]) &
                     (h.law(model._positions[up,'departure'][0]).times[model._positions[up,'departure'][1]]>=ready)&
                     (h.law(model._positions[up,'departure'][0]).times[model._positions[up,'departure'][1]]<menu['deadline']))).sum())),
            hub=dict(cid=hub_dep,station=graph.stops[hub_stop].name,source_arrival_cid=source_next,
                mct=graph.stops[hub_stop].min_transfer_time,actual_ready=actual_ready,
                actual_departure=metadata[hub_dep]['departure']['time_seconds'],
                actual_margin=metadata[hub_dep]['departure']['time_seconds']-actual_ready,
                original_tape_catch_count=int(hub_catch.sum())),
            partitions=partitions,overall_mean_A_minus_C=float(ds.mean()),
            cancellation_and_reachability=dict(upstream_canceled=int(up_cancel.sum()),
                target_hub_departure_canceled=int(hub_cancel.sum()),source_arrival_infeasible=int((~source_arrival_feasible).sum())),
            exact_saved_loss_status_matches=256,validation_cost=validation_cost,
            validation_scope='Full public-state equality, original seed/trip ranks/trip set, all 256 indexed arm losses and statuses. Original receipts contain no raw generated endpoint vectors to compare directly.',
            marginal_scope='Numerical sum of full root P0 finite-chain conditional marginal, including hard records and selected soft emission, excluding cancellation and departure>=deadline. No future message or actual endpoint used in this probability.',
            initial_relevant_forecasts=[r for r in state['public_history'] if r.get('cid') in (up,source_next,hub_dep)
                and r['evidence'] in ('forecast','observed')],rows=policy_rows)
        output.append(out)
        print(json.dumps({k:v for k,v in out.items() if k not in ('rows','initial_relevant_forecasts')},ensure_ascii=False),flush=True)
    result=dict(commit='a2b82be9c3fc6b61e2a5826d7cd8691d39ff4491',
        scope='Original V51 random tape reconstructed from public snapshots, not independent evaluation. Catch means named vehicle departure is legal after transfer ready, not actual policy boarding. Grouped values use original saved paired losses; population conditional values are not certified.',
        selection_caveat='These three states were selected retrospectively because the actual rollout lost. Rare realized events among them are not evidence of probability miscalibration without an independent selection-free denominator.',
        operational_caveat='Actual outcomes mean archived endpoint replay under the existing station transfer-time and ideal logical execution contracts, not observed passenger trajectories.',
        rows=output)
    (HERE/'catch_event_partitions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
