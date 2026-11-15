"""Evaluate frozen dates as both completed source-subset receipts arrive.

This entry runs daily evaluations only. The full fourteen-date bootstrap is a
separate operation after every daily result is complete.
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
import traceback

from experiments.decision_gap_v1.panel import read, write
from experiments.decision_calibration_v2.run_calibration import evaluate_day

HERE = Path(__file__).resolve().parent
WORKERS = 4


def now():
    return datetime.now(timezone.utc).isoformat()


def inputs_ready(directory, day):
    # Each producer closes its compressed subset before creating this receipt.
    return ((directory/'actual'/f'{day}.csv.audit.json').exists()
            and (directory/'messages'/f'{day}.json.audit.json').exists())


def result_receipt(directory, day):
    result = read(directory/'results'/f'{day}.json.gz')
    if result['day'] != day:
        raise ValueError('Existing daily result belongs to a different date')
    return dict(completed_at_utc=now(), roots=len(result['roots']), events=len(result['events']),
        initial_states=dict(Counter(row['initial_state']['status'] for row in result['roots'])),
        model_statuses=dict(Counter(row['model']['status'] for row in result['events'])),
        actual_statuses=dict(Counter(row['actual']['status'] for row in result['events'])),
        cost=result['cost'])


def run(directory):
    directory = Path(directory)
    manifest = read(directory/'date_manifest.json')
    selection = read(directory/'selection.json')
    if not manifest['dates_frozen'] or len(manifest['selected_dates']) != 14:
        raise ValueError('This stage requires the frozen fourteen-date panel')
    audit = dict(started_at_utc=now(), date_freeze_utc=manifest['frozen_at_utc'],
        selection_freeze_utc=selection['frozen_at_utc'], max_workers=WORKERS,
        blas_threads=1, days={}, errors={}, bootstrap_summary_run=False)
    remaining = list(manifest['selected_dates'])
    for day in tuple(remaining):
        if (directory/'results'/f'{day}.json.gz').exists():
            audit['days'][day] = dict(result_receipt(directory, day), reused_complete_result=True)
            remaining.remove(day)
    jobs = {}

    def save():
        audit.update(updated_at_utc=now(), waiting_for_sources=list(remaining),
                     evaluating_dates=sorted(jobs.values()))
        write(directory/'evaluation_run.json', audit)

    save()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        while remaining or jobs:
            changed = False
            for day in tuple(remaining):
                if len(jobs) == WORKERS:
                    break
                if inputs_ready(directory, day):
                    jobs[pool.submit(evaluate_day, directory, day)] = day
                    remaining.remove(day)
                    changed = True
                    print(day, 'evaluating_complete_source_subsets', flush=True)
            done = wait(jobs, timeout=5, return_when=FIRST_COMPLETED).done if jobs else ()
            for future in done:
                day = jobs.pop(future)
                try:
                    future.result()
                    audit['days'][day] = result_receipt(directory, day)
                    print(day, 'complete', audit['days'][day], flush=True)
                except Exception as error:
                    audit['errors'][day] = dict(error=repr(error), failed_at_utc=now())
                    traceback.print_exc()
                    print(day, 'evaluation_failed', repr(error), flush=True)
                changed = True
            if changed:
                save()
            if remaining and not jobs:
                sleep(5)
    audit.update(completed_at_utc=now(), status='failed' if audit['errors'] else 'completed')
    save()
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=HERE)
    result = run(parser.parse_args().directory)
    if result['errors']:
        raise SystemExit(1)
