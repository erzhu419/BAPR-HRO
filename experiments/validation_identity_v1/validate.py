"""Run the original V35 prediction gate against the frozen repaired inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path

from experiments.reconstruction_v35 import validate as original

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = ROOT/'experiments/reconstruction_v35'


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as handle:
        return json.load(handle)


def check_inputs(directory=HERE):
    directory = Path(directory)
    protocol, fit = read(directory/'protocol.json'), read(directory/'fitted_emission.json')
    reference = read(ORIGINAL/'protocol.json')
    if fit != read(ROOT/protocol['candidate_source']):
        raise ValueError('The validation candidate differs from the frozen repaired training fit')
    if fit['training_dates'] != reference['training_dates']:
        raise ValueError('The candidate training dates changed')
    if protocol['validation_dates'] != reference['validation_dates']:
        raise ValueError('The original nine validation dates changed')
    for name, value in protocol['validation'].items():
        if value != reference['validation'][name]:
            raise ValueError(f'The original validation rule changed: {name}')
    events = read(directory/'events.json.gz')['rows']
    train = [r for r in events if r['split']=='training']
    frozen_train = read(ROOT/'experiments/training_identity_v1/corrected_training_events.json.gz')['rows']
    if train != frozen_train:
        raise ValueError('Training rows changed during validation preparation')
    old_events = [r for r in read(ORIGINAL/'events.json.gz')['rows'] if r['split']=='validation']
    new_events = [r for r in events if r['split']=='validation']
    identity = lambda r:(r['date'],r['source_connection_id'],r['kind'])
    before, after = {identity(r):r for r in old_events}, {identity(r):r for r in new_events}
    if before.keys() != after.keys() or len(new_events) != len(old_events):
        raise ValueError('Original validation queries were dropped or duplicated')
    fixed = ('trip_id','route_id','stop_id','stop_sequence','scheduled','prediction_clock',
             'actual_status','actual_seconds','official_journey')
    if any(before[k][field] != after[k][field] for k in before for field in fixed):
        raise ValueError('An original query, cutoff or actual label changed')
    for day in protocol['validation_dates']:
        read(directory/'message_days'/f'{day}.json.gz')
    return protocol


def score_day(day, directory=HERE):
    """Only redirect input/output; preserve the original three model classes."""
    saved = original.HERE
    try:
        original.HERE = Path(directory)
        return original.run_day(day)
    finally:
        original.HERE = saved


def aggregate(directory=HERE, days=None):
    saved = original.HERE
    try:
        original.HERE = Path(directory)
        original.aggregate(days or read(Path(directory)/'protocol.json')['validation_dates'])
    finally:
        original.HERE = saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    protocol = check_inputs()
    (HERE/'validation_days').mkdir(exist_ok=True)
    pending = [d for d in protocol['validation_dates'] if not (HERE/'validation_days'/f'{d}.json.gz').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(score_day, d) for d in pending]):
            future.result()
    aggregate()


if __name__ == '__main__':
    main()
