"""Cumulatively refined pass on the identical frozen causal reference panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import process_time,perf_counter

from experiments.reconstruction_v34.discover import read,context
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node,save
from experiments.reconstruction_v41.run import restored
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v54.reference import RefiningReference

HERE=Path(__file__).resolve().parent
RESIDENT={}


def initialize():
    tick,cpu=perf_counter(),process_time()
    RESIDENT.update(data=read(HERE.parent/'reconstruction_v34/data.json.gz'),
        fit=read(HERE.parent/'reconstruction_v34/model_fit.json'),
        emission=read(HERE.parent/'reconstruction_v35/fitted_emission.json'),graphs={})
    RESIDENT['preparation']=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)


def execute(job):
    item,mode,budget,workers=job
    query=item['query_index'];level=budget['name']
    path=HERE/'natural'/f'{query:03}_{level}_{mode}.json.gz'
    if path.exists():
        return dict(query=query,budget=level,mode=mode,reused=True)
    state=item['state']
    tick,cpu=perf_counter(),process_time()
    if state['day'] not in RESIDENT['graphs']:
        RESIDENT['graphs'][state['day']]=context(RESIDENT['data'],state['day'])[0]
    graph=RESIDENT['graphs'][state['day']]
    graph_preparation=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
    tick,cpu=perf_counter(),process_time()
    model,parameters=setup(graph,RESIDENT['fit'])
    history=restored('shallow',model,parameters,RESIDENT['emission'],state)
    node=restore_node(state['node'])
    description=item['context']['observation_contract']
    observations=ObservationContract({int(r['stop']):tuple(r['cids']) for r in description['station_to_cids']})
    prefix=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
    allowance=max(0.,budget['process_cpu_seconds']-prefix['process_cpu'])
    reference=RefiningReference(graph,model,parameters,state['deadline'],history,observations,
        max_states=budget['states'],max_process_cpu=allowance)
    result=reference.solve(node,state,mode=mode)
    result.update(query_index=query,source_state_index=item['source_state_index'],root_index=state['root_index'],
        day=state['day'],selection_reasons=item['selection_reasons'],
        observed_baseline_actions=item['selected_actions'],budget=budget,
        source=f'../reconstruction_v52/panel.json.gz#/rows/{query}',
        prefix_cost=prefix,complete_cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick),
        graph_preparation=graph_preparation,resident_preparation=RESIDENT['preparation'],screening_workers=workers)
    save(path,result)
    return dict(query=query,root=state['root_index'],budget=level,mode=mode,
        interval=[result['value']['lower'],result['value']['upper']],complete=result['value']['complete'],
        certified=result['certified_action'],states=result['counters']['states'],
        cache_hits=result['counters']['cache_hits'],whole_tails=result['counters']['whole_tail_integrations'],
        impact_pruned=result['counters']['impact_pruned_branches'],
        reflows=result['root_refinement']['reflow_calls'],
        refinements=result['root_refinement']['threshold_refinements'],
        stop=result['root_refinement']['final_stop'],
        cpu=result['complete_cost']['process_cpu'])


def jobs():
    contract=read(HERE/'contract.json')
    panel=read(HERE.parent/'reconstruction_v52/panel.json.gz')
    if panel['queries']!=88 or panel['represented_roots']!=72:
        raise ValueError('Frozen V52 panel changed')
    rows=[]
    for item in panel['rows']:
        q=item['query_index'];modes=contract['modes'];budgets=contract['budgets']
        modes=modes[q%len(modes):]+modes[:q%len(modes)]
        budgets=budgets[q%len(budgets):]+budgets[:q%len(budgets)]
        for budget in budgets:
            rows.extend((item,mode,budget) for mode in modes)
    return rows


def run(workers,queries=None):
    selected=jobs()
    if queries is not None:
        selected=[job for job in selected if job[0]['query_index'] in queries]
    selected=[(*job,workers) for job in selected]
    if workers==1:
        initialize()
        for job in selected:
            print(execute(job),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
            for result in pool.map(execute,selected,chunksize=1):
                print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--queries',type=int,nargs='+')
    args=parser.parse_args()
    run(args.workers,args.queries)
