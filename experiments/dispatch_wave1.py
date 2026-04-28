"""Wave 1 of post-multi-day parallel batch:

18 jobs = cross-domain × 6 seeds × 3 domains.

Dispatches subprocess.run for each (domain, seed). With 18 workers,
all jobs complete in roughly 1 wave (3-5 minutes for cross-domain).
"""

import os, sys, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE, 'experiments_log', 'multiseed')
os.makedirs(LOG_DIR, exist_ok=True)

JOBS = []
for seed in range(6):  # 6 seeds: 0,1,2,3,4,5
    JOBS.append(('vrp', seed,
                 [sys.executable, '-u', 'VRP/experiments/run_vrp_comparison.py',
                  '--seed-offset', str(seed),
                  '--out', os.path.join(LOG_DIR, f'vrp_seed{seed}.json')]))
    JOBS.append(('uc', seed,
                 [sys.executable, '-u',
                  'power_dispatch/experiments/run_uc_comparison.py',
                  '--seed-offset', str(seed),
                  '--out', os.path.join(LOG_DIR, f'uc_seed{seed}.json')]))
    JOBS.append(('sdn', seed,
                 [sys.executable, '-u',
                  'sdn_routing/experiments/run_sdn_comparison.py',
                  '--seed-offset', str(seed),
                  '--out', os.path.join(LOG_DIR, f'sdn_seed{seed}.json')]))

# Total jobs = 18 (one full wave on 18 workers)


def run_job(job):
    name, seed, cmd = job
    t0 = time.time()
    log_path = os.path.join(LOG_DIR, f'{name}_seed{seed}.log')
    with open(log_path, 'w') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=BASE)
    elapsed = time.time() - t0
    return f'[{name}/{seed}] rc={proc.returncode} {elapsed:.0f}s'


def main():
    print(f'Wave 1: {len(JOBS)} jobs on 18 workers')
    print(f'Logs in {LOG_DIR}')
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=18) as exe:
        futures = [exe.submit(run_job, j) for j in JOBS]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)
    print(f'\nWave 1 done: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
