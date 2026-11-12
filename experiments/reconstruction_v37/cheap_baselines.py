"""Declared post-readout audit of four existing deterministic cheap controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
from time import perf_counter, process_time

from experiments.reconstruction_v33.run import jsonable
from experiments.reconstruction_v34.discover import context, read
from experiments.reconstruction_v34.run import inputs
from experiments.reconstruction_v36.run import one


HERE = Path(__file__).resolve().parent


def run_root(index):
    tick, cpu = perf_counter(), process_time()
    protocol = read(HERE/'cheap_baseline_protocol.json')
    _, data, messages, physical_fit, selection = inputs()
    emission_fit = read(HERE.parent/'reconstruction_v35/fitted_emission.json')
    root = selection['selected'][index]
    graph, _, _, mapping, metadata, _ = context(data, root['day'])
    target = HERE/'cheap_roots'/f'{index:03}.json.gz'
    result = (read(target) if target.exists() else dict(root_index=index, root_id=root['root_id'],
        day=root['day'], protocol='cheap_baseline_protocol.json',
        model='reconstruction_v36.SingleMessageHistory', rows=dict(public_messages={})))
    previous_cpu, previous_wall = result.get('total_process_cpu', 0.), result.get('total_wall', 0.)
    previous_preparation = result.get('preparation', dict(process_cpu=0., wall=0.))
    result['preparation'] = dict(process_cpu=previous_preparation['process_cpu']+process_time()-cpu,
                                wall=previous_preparation['wall']+perf_counter()-tick)
    for name, (method, waiting) in protocol['methods'].items():
        if name in result['rows']['public_messages']:
            continue
        row = one(index, graph, metadata, mapping, root, physical_fit, emission_fit,
                  messages['dates'][root['day']]['frames'], name, method, waiting)
        result['rows']['public_messages'][name] = row
        result.update(total_process_cpu=previous_cpu+process_time()-cpu,
                      total_wall=previous_wall+perf_counter()-tick)
        # Persist each completed deterministic control, so resuming never
        # discards an already paid result from a partly completed root.
        with gzip.open(target, 'wt') as handle:
            json.dump(result, handle, default=jsonable, allow_nan=False)
        print(index, name, row['outcome']['status'], row['outcome']['loss'],
              'CPU', round(row['cost']['total_process_cpu'], 4), flush=True)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    methods = read(HERE/'cheap_baseline_protocol.json')['methods']
    selection = read(HERE.parent/'reconstruction_v34/selection.json')['selected']
    (HERE/'cheap_roots').mkdir(exist_ok=True)
    pending = []
    for index in range(len(selection)):
        path = HERE/'cheap_roots'/f'{index:03}.json.gz'
        if not path.exists() or any(name not in read(path)['rows']['public_messages'] for name in methods):
            pending.append(index)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_root, index) for index in pending]
        for job in as_completed(jobs):
            job.result()


if __name__ == '__main__':
    main()
