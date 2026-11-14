"""V41 frozen physical-model and continuation-policy paired experiment."""
import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v33.run import jsonable
from experiments.reconstruction_v34.conditional_reference import AnchoredWorld
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.execute import ArchivedWorld, Executor
from experiments.reconstruction_v34.run import inputs, relevant_frames, setup
from experiments.reconstruction_v36.forecast import SingleMessageHistory
from experiments.reconstruction_v37.controller import has_legal_continue, node_structure, restore_single_snapshot
from experiments.reconstruction_v38.planner import FastPlanner
from experiments.reconstruction_v38.run import ordered, physical_outcome, restore_node, save
from experiments.reconstruction_v39.run import source_bank
from experiments.reconstruction_v39.state import DecisionEngine
from experiments.reconstruction_v41.action_publisher import ActionPublisher, ActionToken
from experiments.reconstruction_v41.continuation import evaluate, choose, science
from experiments.reconstruction_v41.matched_continuation import MatchedContinuationPlanner
from experiments.reconstruction_v41.physical_posterior import (
    PhysicalHistory, PhysicalPlanner, PhysicalDecisionEngine, CheckingWorld, restore_physical_snapshot,
)

HERE = Path(__file__).resolve().parent
V38, V40 = HERE.parent/'reconstruction_v38', HERE.parent/'reconstruction_v40'
ARMS = ('legacy', 'physical', 'shallow', 'physical_meeting', 'shallow_meeting')
PLANNERS = dict(legacy=FastPlanner, physical=PhysicalPlanner, shallow=MatchedContinuationPlanner)


def normalized(value):
    return json.loads(json.dumps(value, default=jsonable, allow_nan=False))


def engine(planner, family):
    return (DecisionEngine if family == 'legacy' else PhysicalDecisionEngine)(planner, variant='frontier')


def restored(family, model, parameters, fit, state):
    return (restore_single_snapshot if family == 'legacy' else restore_physical_snapshot)(
        model, parameters, fit, state['public_history'], state['clock'])


def score(eng, node, cid, clock, history):
    action, receipt = eng.onboard(node, cid, clock, history, method='rolling_fixed')
    return ('alight' if action else 'continue'), receipt


def checked_evaluation(graph, model, parameters, node, history, state, family, count, seed, meeting=False):
    world = AnchoredWorld if family == 'legacy' else CheckingWorld
    public = history.public_state()
    return evaluate(graph, model, parameters, state['deadline'], node,
        {k:state[k] for k in ('kind','cid','clock')}, history,
        planner_type=PLANNERS[family], count=count, seed=seed, meeting=meeting,
        world_type=lambda bank, index: world(bank, index, public))


def old_rows_equal(result, old):
    rows = {old['actions'][label]: values for label, values in old['rows'].items()}
    fields = ('loss','status','trace','decisions','world_index')
    return normalized({a:[{k:r[k] for k in fields} for r in values] for a,values in result['rows'].items()}) == normalized(
        {a:[{k:r[k] for k in fields} for r in values] for a,values in rows.items()})


def references(manifest, data, fit, emission):
    checks = []
    for index, entry in enumerate(manifest):
        state = entry['state']
        graph = context(data, state['day'])[0]
        measured = {}
        for arm in ordered(ARMS,index):
            path = HERE/'banks'/f'{index:03}_{arm}.json.gz'
            if path.exists():
                result = read(path)
            else:
                tick, cpu = perf_counter(), process_time()
                family = arm.removesuffix('_meeting')
                model, params = setup(graph, fit)
                node = restore_node(state['node'])
                history = restored(family, model, params, emission, state)
                planner = PLANNERS[family](graph, model, params, state['deadline'])
                base, receipt = score(engine(planner,family), node, state['cid'],state['clock'],history)
                result = checked_evaluation(graph,model,params,node,history,state,family,128,entry['seed'],
                                            meeting=arm.endswith('_meeting'))
                result.update(index=index, source=entry['source'], root_index=state['root_index'], family=family,
                    state=state, base_action=base, base_receipt=receipt, selected_action=choose(result,base),
                    complete_cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick))
                if family == 'legacy':
                    result['legacy_worlds_exact'] = old_rows_equal(result,read(source_bank(entry['source'])))
                save(path,result)
                print('bank',index,arm,'unresolved',result['paired']['unresolved_pairs'],
                    'action',result['base_action'],result['selected_action'],
                    'CPU',round(result['complete_cost']['process_cpu'],4),flush=True)
            measured[arm]=result
        check = dict(index=index,source=entry['source'],legacy_worlds_exact=measured['legacy']['legacy_worlds_exact'],
            physical_meeting_exact=normalized(science(measured['physical']))==normalized(science(measured['physical_meeting'])),
            shallow_meeting_exact=normalized(science(measured['shallow']))==normalized(science(measured['shallow_meeting'])))
        if not all(check[k] for k in ('legacy_worlds_exact','physical_meeting_exact','shallow_meeting_exact')):
            save(HERE/'reference_failure.json',check)
            raise AssertionError(check)
        checks.append(check)
        path=HERE/'tails'/f'{index:03}.json'
        if not path.exists():
            tick,cpu=perf_counter(),process_time()
            model,params=setup(graph,fit)
            history=restored('physical',model,params,emission,state)
            node=restore_node(state['node'])
            planner=MatchedContinuationPlanner(graph,model,params,state['deadline'])
            platform=node.exits[state['cid']]
            ready=state['clock']+graph.stops[platform.stop].min_transfer_time
            envelope=planner.selective_waiting_components(platform,history,np.array([ready]))
            exact=planner.frozen_combined_reference(platform,history,state['clock'],ready)
            save(path,dict(index=index,source=entry['source'],root_index=state['root_index'],clock=state['clock'],
                candidate_count=len(platform.rides),first=float(envelope['first_offer'][0])-state['clock'],
                envelope=float(envelope['value'][0])-state['clock'],
                frozen_combined=exact.value-state['clock'],counters=exact.counters,
                cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick),
                scope='Exact fixed-Combined with frozen child marks; not full G_pi or future-message optimum'))
        save(HERE/'reference_validation.json',dict(completed=len(checks),expected=148,states=checks))


def public_seed(root_index, node, cid, clock, history, manifest):
    public, structure=normalized(history.public_state()),node_structure(node)
    match=next((r for r in manifest if r['state']['cid']==cid and r['state']['clock']==clock
        and r['state']['node']==structure and r['state']['public_history']==public),None)
    if match is not None:
        return match['seed'],match['source']
    return int(np.random.SeedSequence([41050,root_index,cid]).generate_state(1)[0]),None


class OnlineExecutor(Executor):
    def __init__(self,*args,family,mode,root_index,manifest,cpu_start,**kwargs):
        super().__init__(*args,method='rolling_fixed',waiting='combined',**kwargs)
        self.family,self.mode,self.root_index=family,mode,root_index
        self.manifest=manifest
        self.engine=engine(self.planner,family)
        self.gates,self.publications,self.decision_costs=[],[],[]
        self.pending_cpu=cpu_start

    def publish(self,clock,legal,fallback,compute=None):
        # Frozen ideal public-event contract: logical time does not advance
        # with computing. Real wall-window behavior is tested separately.
        publisher=ActionPublisher(wall_clock=lambda: float(clock))
        token=ActionToken(str(len(self.publications)),tuple(legal),clock,clock+1,
            'ideal logical event; no wall deadline or inferred door window')
        request=publisher.request(token,fallback)
        if compute is not None:
            request.submit(compute())
        consumed=request.take()
        self.publications.append(request.finish())
        if consumed is None:
            raise AssertionError('Ideal event failed to deliver its legal action')
        return consumed['action']

    def done(self,kind,clock):
        end=process_time()
        self.decision_costs.append(dict(kind=kind,clock=clock,process_cpu=end-self.pending_cpu))
        self.pending_cpu=end

    def choose_onboard(self,node,cid,clock,history):
        base,receipt=score(self.engine,node,cid,clock,history)
        compute=None
        if self.mode!='cheap' and has_legal_continue(node,cid):
            def compute():
                count=16 if self.mode.endswith('16') else 128
                seed,source=public_seed(self.root_index,node,cid,clock,history,self.manifest)
                state=dict(kind='onboard',cid=cid,clock=clock,deadline=self.deadline)
                gate=checked_evaluation(self.graph,self.planner.model,self.planner.parameters,node,history,state,
                    self.family,count,seed,meeting=self.mode.startswith('meeting'))
                gate.update(source=source,cid=cid,clock=clock,base_action=base,base_receipt=receipt,
                    selected_action=choose(gate,base))
                self.gates.append(gate)
                return gate['selected_action']
        legal=('alight','continue')
        chosen=self.publish(clock,legal,base,compute)
        self.done('onboard',clock)
        return chosen=='alight',dict(base=base,selected=chosen,cheap=receipt)

    def choose_platform(self,platform,entry,ready,history,visible,observed,clock):
        choice,receipt=self.engine.platform(platform,entry,ready,history,visible,observed,clock,self.choose,
            method=self.method)
        choice=self.publish(clock,(*visible,None),choice)
        self.done('platform',clock)
        return choice,receipt


def online(manifest,data,messages,fit,emission,selection,protocol):
    by_root={i:[e for e in manifest if e['state']['root_index']==i] for i in range(72)}
    for index,root in enumerate(selection['selected']):
        tick,cpu=perf_counter(),process_time()
        graph,_,_,mapping,metadata,_=context(data,root['day'])
        preparation=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
        frames=messages['dates'][root['day']]['frames']
        for label in ordered(protocol['online_configurations'],index):
            path=HERE/'roots'/f'{index:03}_{label}.json.gz'
            if path.exists():
                continue
            family,mode=label.split('_',1)
            tick,cpu=perf_counter(),process_time()
            model,params=setup(graph,fit)
            node=root_ride(graph,root)
            planner=PLANNERS[family](graph,model,params,root['deadline'])
            history_type=SingleMessageHistory if family=='legacy' else PhysicalHistory
            history=history_type(model,params,fit=emission,
                frames=relevant_frames(frames,node.trips(model),mapping),source_to_local=mapping)
            cold=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
            executor=OnlineExecutor(graph,planner,root['deadline'],family=family,mode=mode,
                root_index=index,manifest=by_root[index],cpu_start=cpu)
            outcome=executor.run(node,root['start'],history,ArchivedWorld(metadata))
            end_cpu,end_wall=process_time(),perf_counter()
            result=dict(root_index=index,root_id=root['root_id'],day=root['day'],origin=root['origin_name'],
                goal=root['goal_name'],label=label,loss_cap=root['deadline']-root['start'],
                outcome=outcome,gates=executor.gates,publications=executor.publications,
                decision_costs=executor.decision_costs,preparation=preparation,
                cost=dict(total_process_cpu=end_cpu-cpu,total_wall=end_wall-tick,cold=cold,
                    decision_cpu_sum=sum(r['process_cpu'] for r in executor.decision_costs),
                    final_execution_cpu=end_cpu-executor.pending_cpu,planner_stats=dict(planner.stats)),
                saved_answers_reused=0)
            if family=='legacy':
                name='frontier__combined_rolling' if mode=='cheap' else 'frontier__pairs128'
                old=read(V40/'roots'/f'{index:03}_{"cheap_" if mode=="cheap" else ""}{name}.json.gz')
                result['legacy_physical_exact']=physical_outcome(outcome)==physical_outcome(old['outcome'])
            save(path,result)
            print('root',index,label,outcome['status'],'loss',outcome['loss'],
                'CPU',round(result['cost']['total_process_cpu'],4),'gates',len(executor.gates),flush=True)
            if result.get('legacy_physical_exact') is False:
                raise AssertionError(f'Preserved legacy physical mismatch: {path}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('banks','online'))
    args=parser.parse_args()
    tick,cpu=perf_counter(),process_time()
    protocol=read(HERE/'protocol.json')
    manifest=read(V38/'seed_manifest.json')
    _,data,messages,fit,selection=inputs()
    emission=read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    preparation=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick,manifest_states=len(manifest),
        roots=len(selection['selected']),scope='Resident common archive, manifest and parameter inputs; once per process')
    save(HERE/f'{args.stage}_preparation.json',preparation)
    if args.stage=='banks':
        references(manifest,data,fit,emission)
    else:
        if read(HERE/'reference_validation.json')['completed']!=148:
            raise ValueError('Complete every reference state before online comparisons')
        online(manifest,data,messages,fit,emission,selection,protocol)


if __name__=='__main__':
    main()
