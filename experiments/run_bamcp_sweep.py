"""BAMCP rollout-budget sweep: BAMCP-60 / BAMCP-120 / BAMCP-240.

Reviewer R-final concern #4.4: "BAMCP-60 with 60 rollouts may be
compute-starved. If so, it's a resource-restricted baseline, not a
methodologically inferior method."

Run BAMCP at 60, 120, 240 rollouts on synthetic small + large networks
× normal + disrupted scenarios. Compare mean travel time. If BAMCP-240
≈ BAMCP-60, then 60 was sufficient. If BAMCP-240 substantially
outperforms BAMCP-60, then the paper should add BAMCP-240 to the
main comparison.

Usage:
    python3 experiments/run_bamcp_sweep.py \\
        --net small --scenario disrupted --methods BAMCP-60,BAMCP-120,BAMCP-240,Static \\
        --out experiments_log/bamcp_sweep/small_disrupted.json
"""

import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_full_comparison import run_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--net', choices=['small', 'large'], required=True)
    p.add_argument('--scenario', required=True,
                   help='Normal/disrupted (e.g. normal, disrupted, '
                        'disrupted_402, no_disruption, etc.)')
    p.add_argument('--methods', default='Static,BAMCP-60,BAMCP-120,BAMCP-240')
    p.add_argument('--n', type=int, default=50,
                   help='journeys per cell')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(',')]
    print(f'BAMCP sweep: net={args.net} scenario={args.scenario} '
          f'methods={methods} n={args.n} seed={args.seed}')
    t0 = time.time()
    res = run_experiment(args.net, args.scenario, methods, args.n, args.seed)
    elapsed = time.time() - t0

    out = {'net': args.net, 'scenario': args.scenario, 'seed': args.seed,
           'n_journeys': args.n, 'elapsed_s': elapsed,
           'results': {m: {k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in r.items() if k not in ('trials',)}
                       for m, r in res.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved {args.out} ({elapsed:.0f}s)')


if __name__ == '__main__':
    main()
