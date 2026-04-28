"""Wave 2 of post-multi-day parallel batch:

18 jobs = 12 BAMCP rollout-sweep cells + 6 extra cross-domain seeds.

BAMCP sweep:
  3 budgets (60, 120, 240) × 2 networks (small, large) × 2 scenarios
  (normal, disrupted) = 12 cells. Runs Static + BAMCP-{N} per cell.

Extra cross-domain seeds (to fill the wave):
  VRP/UC/SDN × seeds 6, 7 = 6 jobs.
"""

import os, sys, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAMCP_DIR = os.path.join(BASE, 'experiments_log', 'bamcp_sweep')
MULTISEED_DIR = os.path.join(BASE, 'experiments_log', 'multiseed')
os.makedirs(BAMCP_DIR, exist_ok=True)
os.makedirs(MULTISEED_DIR, exist_ok=True)

JOBS = []

# 12 BAMCP rollout-sweep cells. We run each cell with 3 BAMCP variants
# in one subprocess (run_full_comparison's run_experiment loops methods),
# so 12 cells = 12 jobs not 36.
for budget in [60, 120, 240]:
    for net in ['small', 'large']:
        for scen in ['normal', 'disrupted']:
            tag = f'{net}_{scen}_b{budget}'
            JOBS.append((f'bamcp_{tag}',
                         [sys.executable, '-u',
                          'experiments/run_bamcp_sweep.py',
                          '--net', net, '--scenario', scen,
                          '--methods', f'Static,BAMCP-{budget}',
                          '--n', '50',
                          '--out', os.path.join(BAMCP_DIR, f'{tag}.json')]))

# 6 extra cross-domain seeds (6, 7) × 3 domains
for seed in [6, 7]:
    JOBS.append((f'vrp_s{seed}',
                 [sys.executable, '-u',
                  'VRP/experiments/run_vrp_comparison.py',
                  '--seed-offset', str(seed),
                  '--out', os.path.join(MULTISEED_DIR, f'vrp_seed{seed}.json')]))
    JOBS.append((f'uc_s{seed}',
                 [sys.executable, '-u',
                  'power_dispatch/experiments/run_uc_comparison.py',
                  '--seed-offset', str(seed),
                  '--out', os.path.join(MULTISEED_DIR, f'uc_seed{seed}.json')]))
    JOBS.append((f'sdn_s{seed}',
                 [sys.executable, '-u',
                  'sdn_routing/experiments/run_sdn_comparison.py',
                  '--seed-offset', str(seed),
                  '--out', os.path.join(MULTISEED_DIR, f'sdn_seed{seed}.json')]))

# Total = 12 + 6 = 18 jobs


def run_job(job):
    name, cmd = job
    t0 = time.time()
    log_path = os.path.join(BASE, 'experiments_log',
                            'wave2_logs', f'{name}.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=BASE)
    elapsed = time.time() - t0
    return f'[{name}] rc={proc.returncode} {elapsed:.0f}s'


def main():
    print(f'Wave 2: {len(JOBS)} jobs on 18 workers')
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=18) as exe:
        futures = [exe.submit(run_job, j) for j in JOBS]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)
    print(f'\nWave 2 done: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
