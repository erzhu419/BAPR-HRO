"""Read-only V55 profile. Instrumentation changes work completed under the cap.

This diagnoses which functions consume CPU; it is not a timing benchmark or a
new paired outcome. Archived experiment receipts are never overwritten.
"""
import cProfile
import json
import os
import sys
from pathlib import Path
import pstats
from time import process_time, perf_counter

REPO = Path(os.environ.get('BAPR_HRO_REPO', '/workspace/scratch/fde8f7d289c2/repos/BAPR-HRO-a2b82be9'))
sys.path.insert(0, str(REPO))

from experiments.reconstruction_v34.discover import read, context
from experiments.reconstruction_v34.run import setup
from experiments.reconstruction_v38.run import restore_node
from experiments.reconstruction_v41.run import restored
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v55.priority import ImpactStructuralReference

BASE = REPO / 'experiments'
OUT = Path(__file__).resolve().parent
data = read(BASE/'reconstruction_v34/data.json.gz')
fit = read(BASE/'reconstruction_v34/model_fit.json')
emission = read(BASE/'reconstruction_v35/fitted_emission.json')
panel = read(BASE/'reconstruction_v52/panel.json.gz')
graphs = {}

for query in (8, 39):
    item = next(row for row in panel['rows'] if row['query_index'] == query)
    state = item['state']
    if state['day'] not in graphs:
        graphs[state['day']] = context(data, state['day'])[0]
    graph = graphs[state['day']]
    tick, cpu = perf_counter(), process_time()
    model, parameters = setup(graph, fit)
    history = restored('shallow', model, parameters, emission, state)
    node = restore_node(state['node'])
    description = item['context']['observation_contract']
    observations = ObservationContract({int(row['stop']): tuple(row['cids'])
        for row in description['station_to_cids']})
    prefix = {'process_cpu': process_time()-cpu, 'wall': perf_counter()-tick}
    allowance = max(0., 10.-prefix['process_cpu'])
    profiler = cProfile.Profile()
    profiler.enable()
    reference = ImpactStructuralReference(graph, model, parameters,
        state['deadline'], history, observations, max_states=20000,
        max_process_cpu=allowance)
    result = reference.solve(node, state, mode='cheap_working')
    profiler.disable()
    result.update(query_index=query, root_index=state['root_index'],
        prefix_cost=prefix, complete_cost={'process_cpu': process_time()-cpu,
        'wall': perf_counter()-tick},
        audit_scope='cProfile overhead is charged inside original 10 CPU-second cap. Diagnosis only; do not compare performance or bound quality to uninstrumented receipts.')
    stem = OUT/f'profile_root{state["root_index"]}_cheap_working_b10'
    stem.with_suffix('.json').write_text(json.dumps(result, indent=2))
    profiler.dump_stats(str(stem.with_suffix('.pstats')))
    with stem.with_suffix('.txt').open('w') as handle:
        stats = pstats.Stats(profiler, stream=handle)
        stats.sort_stats('cumulative').print_stats(55)
        stats.sort_stats('tottime').print_stats(35)
    print(json.dumps({'root':state['root_index'], 'query':query,
        'cpu':result['complete_cost']['process_cpu'], 'value':result['value'],
        'counters':result['counters']}), flush=True)
