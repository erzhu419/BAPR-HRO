"""Re-run only the failed VRP seeds for wave 1 (after syntax fix)."""

import os, sys, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE, 'experiments_log', 'multiseed')
os.makedirs(LOG_DIR, exist_ok=True)

JOBS = []
for seed in range(6):
    JOBS.append((f'vrp_s{seed}',
                 [sys.executable, '-u',
                  'VRP/experiments/run_vrp_comparison.py',
                  '--seed-offset', str(seed),
                  '--out', os.path.join(LOG_DIR, f'vrp_seed{seed}.json')]))


def run_job(job):
    name, cmd = job
    t0 = time.time()
    log_path = os.path.join(LOG_DIR, f'{name}.log')
    with open(log_path, 'w') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=BASE)
    return f'[{name}] rc={proc.returncode} {time.time()-t0:.0f}s'


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=6) as exe:
        futures = [exe.submit(run_job, j) for j in JOBS]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)
