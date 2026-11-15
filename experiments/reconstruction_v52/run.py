"""Input-only V52 panel and bounded causal-policy screening."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v32.conditional import UnsupportedHistory
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node, save
from experiments.reconstruction_v41.run import restored
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v52.reference import CausalReference

HERE = Path(__file__).resolve().parent
RESIDENT = {}


def _action(action):
    return 'wait' if action is None else str(action)


def prepare():
    """Select without reading a V52 bound, modelvalue, or actual future loss."""
    source = read(HERE.parent/'reconstruction_v51/states.json.gz')
    states = source['states']
    contexts = {int(row['root_index']):row for row in source['contexts']}
    rows, missing, selected = [], [], set()

    def ordered(row):
        return (row['state']['clock'],
                min((o['state_index'] for o in row['occurrences']),default=0),row['index'])

    def add(row, reason):
        index = row['index']
        if index in selected:
            next(r for r in rows if r['source_state_index']==index)['selection_reasons'].append(reason)
            return
        selected.add(index)
        rows.append(dict(query_index=len(rows),source_state_index=index,state=row['state'],
            occurrences=row['occurrences'],selected_actions=row['selected_actions'],
            selection_reasons=[reason],context=contexts[int(row['state']['root_index'])]))

    all_disagreements = []
    for root_index in range(72):
        available = sorted((row for row in states if row['state']['root_index']==root_index
                            and len(row['state']['legal_actions'])>=2
                            and (row['state']['kind']!='onboard'
                                 or row['state']['has_continuation'])),key=ordered)
        if not available:
            missing.append(root_index)
            continue
        add(available[0],'first_nontrivial_public_decision')
        divergent = [row for row in available if len({_action(a) for a in row['selected_actions']})>1]
        all_disagreements.extend(dict(root_index=root_index,source_state_index=row['index'],
            clock=row['state']['clock'],actions=row['selected_actions']) for row in divergent)
        for row in divergent:
            add(row,'same_history_policy_disagreement')
    result = dict(stage='V52',source='experiments/reconstruction_v51/states.json.gz',
        all_state_count=len(states),original_root_denominator=72,
        represented_roots=len({row['state']['root_index'] for row in rows}),
        missing_first_nontrivial_roots=missing,queries=len(rows),
        same_history_disagreement_states=len(all_disagreements),all_disagreements=all_disagreements,
        input_only_selection=True,rows=rows)
    path=HERE/'panel.json.gz'
    if path.exists():
        old=read(path)
        if old!=result:
            raise ValueError('V51 input panel differs from the already frozen V52 panel')
    else:
        save(path,result)
    print(dict(states=len(states),queries=len(rows),represented_roots=result['represented_roots'],
               missing=missing,disagreements=len(all_disagreements)),flush=True)
    return result


def initialize():
    tick,cpu=perf_counter(),process_time()
    RESIDENT.update(data=read(HERE.parent/'reconstruction_v34/data.json.gz'),
        fit=read(HERE.parent/'reconstruction_v34/model_fit.json'),
        emission=read(HERE.parent/'reconstruction_v35/fitted_emission.json'),graphs={})
    RESIDENT['preparation']=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)


def execute(job):
    item,mode,workers,implementation=job
    query_index=item['query_index']
    folder='natural' if implementation=='chronological' else 'natural_optimized'
    path=HERE/folder/f'{query_index:03}_{mode}.json.gz'
    if path.exists():
        return dict(query_index=query_index,mode=mode,reused=True)
    state=item['state']
    tick,cpu=perf_counter(),process_time()
    if state['day'] not in RESIDENT['graphs']:
        RESIDENT['graphs'][state['day']]=context(RESIDENT['data'],state['day'])[0]
    graph=RESIDENT['graphs'][state['day']]
    graph_preparation=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
    tick,cpu=perf_counter(),process_time()
    model,params=setup(graph,RESIDENT['fit'])
    history=restored('shallow',model,params,RESIDENT['emission'],state)
    node=restore_node(state['node'])
    description=item['context']['observation_contract']
    contract=ObservationContract({int(row['stop']):tuple(row['cids'])
                                 for row in description['station_to_cids']})
    prefix=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)
    limits=read(HERE/'contract.json')['limits']
    allowance=max(0.,limits['process_cpu_seconds_per_query']-prefix['process_cpu'])
    oracle=CausalReference(graph,model,params,state['deadline'],history,contract,
                          max_states=limits['states_per_query'],max_process_cpu=allowance,
                          implementation=implementation)
    try:
        result=oracle.solve(node,state,mode=mode)
    except UnsupportedHistory as error:
        cap=state['deadline']-state['clock']
        result=dict(mode=mode,status='unsupported_root_history',error=str(error),
            value=dict(lower=0.,upper=cap,mean=None,complete=False),actions={},
            selected_action=None,certified_action=None,counters=dict(oracle.stats),
            reasons={'unsupported_root_history':1},clock=state['clock'],deadline=state['deadline'],
            future_public_frames=0,population_scope=True,public_history=history.public_state())
    result.update(query_index=query_index,source_state_index=item['source_state_index'],
        root_index=state['root_index'],day=state['day'],selection_reasons=item['selection_reasons'],
        source=f'panel.json.gz#/rows/{query_index}',observed_baseline_actions=item['selected_actions'],
        requested_limits=limits,prefix_cost=prefix,solver_cpu_allowance=allowance,
        complete_cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick),
        resident_preparation=RESIDENT['preparation'],graph_preparation=graph_preparation,
        screening_workers=workers,implementation=implementation)
    save(path,result)
    return dict(query_index=query_index,root_index=state['root_index'],mode=mode,
        complete=result['value']['complete'],interval=[result['value']['lower'],result['value']['upper']],
        certified=result['certified_action'],states=result['counters']['states'],
        cpu=result['complete_cost']['process_cpu'])


def run(workers=1,implementation='chronological'):
    if workers not in (1,2):
        raise ValueError('The frozen screening protocol allows one or two workers')
    panel=prepare()
    modes=read(HERE/'contract.json')['modes']
    jobs=[]
    for item in panel['rows']:
        shift=item['query_index']%len(modes)
        order=modes[shift:]+modes[:shift]
        jobs.extend((item,mode,workers,implementation) for mode in order)
    if workers==1:
        initialize()
        for job in jobs:
            print(execute(job),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
            for result in pool.map(execute,jobs,chunksize=1):
                print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--implementation',choices=('chronological','mass_terminal'),default='chronological')
    args=parser.parse_args()
    prepare() if args.prepare else run(args.workers,args.implementation)
