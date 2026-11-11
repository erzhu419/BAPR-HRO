"""One frozen-history diagnostic; run alone with a 30-second external timeout."""
import json
import os
import platform
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.reconstruction_v4.run_city_replay import load_city
from experiments.reconstruction_v4.run_dated_replay import dated_graph, fit_training
from experiments.reconstruction_v5.run_sparse_certificates import context
import src.temporal_scan as scan
from src.evolving_trip_belief import logsumexp

def old_draw(law, uniforms):
    n, slots = uniforms.shape
    result = np.empty((n, slots), dtype=np.intp)
    for index in range(slots-1,-1,-1):
        forward = law._full_forward(index)
        if index == slots-1:
            weights, inverse = forward[None,:], np.zeros(n,dtype=np.intp)
        else:
            next_state,inverse = np.unique(result[:,index+1],return_inverse=True)
            weights = forward + law._transition(index)[:,next_state].T
        probabilities = np.exp(weights-logsumexp(weights,axis=1,keepdims=True))
        cumulative = np.cumsum(probabilities,axis=1)
        cumulative /= cumulative[:,-1:]
        result[:,index] = np.sum(uniforms[:,index,None]>=cumulative[inverse],axis=1)
    return result

data,_=load_city(ROOT / 'experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz')
fit=fit_training(data,retain_no_pickup=True)
r=json.loads((ROOT / 'experiments/reconstruction_v9/sample_results.json').read_text())['rows'][0]
journey=next(j for j in json.loads((ROOT / 'experiments/reconstruction_v4/city_replay_results.json').read_text())['rows'] if j['method']=='evolving_rollout' and (j['date'],j['query'])==(r['date'],r['query']))
graph,_,stops,_,metadata,_=dated_graph(data,r['date'],'gtfs_station_max',True)
parameters={c.route:fit['route_priors'].get(c.route,fit['route_priors']['pooled']) for c in graph.connections}
new_draw=scan.draw_chain_batch
result = dict(
    protocol='Single frozen history, old then native, one cold and one warm call per sample count. '
             'Cold includes law and forward-message construction. Window/template setup is excluded. '
             'Each arm receives an independently reconstructed belief and the identical RNG seed. '
             'Warm reuses that arm\'s law caches. Compilation is common setup, reported separately. '
             'Scalar128 uses the longest retained trip with warm forward messages; its cold label '
             'means empty original scalar CDF cache. Single-run timings are diagnostic, not a final '
             'balanced performance experiment. No CPU pinning was applied.',
    command='timeout 30s python3 experiments/reconstruction_v11/sampler_run.py',
    hardware=dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                  cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                 if line.startswith('model name')), affinity=sorted(os.sched_getaffinity(0))),
    history=dict(date=r['date'], query=r['query'], decision=r['decision'],
                 seed=[*r['random_seed'],10,202611], deadline=journey['od']['deadline_seconds']), rows=[])
with TemporaryDirectory(prefix='bapr_ffbs_') as tmp:
    start=perf_counter(); kernel=scan.build_temporal_kernel(Path(tmp)/'scan.so'); compile_seconds=perf_counter()-start
    result['compile_seconds'] = compile_seconds
    print(json.dumps(dict(compile_seconds=compile_seconds,date=r['date'],query=r['query'],decision=r['decision'])),flush=True)
    for n in [128,512]:
        output={}; expected=None
        for name,primitive in [('old',old_draw),('native',new_draw)]:
            belief,state=context(graph,metadata,journey,r['decision'],fit)
            trips=scan.window_trips(belief,parameters,state.time,journey['od']['deadline_seconds'])
            templates=scan.template_trip_map(belief,parameters)
            scan.draw_chain_batch=primitive
            output[name]={}
            for phase in ['cold','warm']:
                rng=np.random.default_rng(np.random.SeedSequence([*r['random_seed'],10,202611]))
                start=perf_counter(); values=scan.sample_clocks(belief,parameters,rng,n,trips=trips,templates=templates)
                output[name][phase]=perf_counter()-start
            if name=='old': expected=values
            else:
                assert all(np.array_equal(a,b) for a,b in zip(expected,values))
        row = dict(n=n,trips=len(trips),seconds=output,exact_old_native=True)
        result['rows'].append(row)
        print(json.dumps(row),flush=True)
    law=belief._law(max(trips,key=lambda t:len(belief._trips[t])),parameters)
    tape=np.random.default_rng(419).random((128,len(law.times)))[:,::-1]
    output={}
    for label,draw in [('old',law._draw_uniforms_python),('native',law.draw_uniforms)]:
        for phase in ['cold','warm']:
            start=perf_counter(); values=np.stack([draw(row) for row in tape]); output[label+'_'+phase]=perf_counter()-start
        if label=='old': expected=values
        else: assert np.array_equal(expected,values)
    print(json.dumps(dict(scalar128_seconds=output,slots=len(law.times),exact_old_native=True)),flush=True)
    result['scalar128'] = dict(seconds=output, slots=len(law.times), exact_old_native=True)
(Path(__file__).with_name('sampler_results.json')).write_text(json.dumps(result, indent=2) + '\n')
