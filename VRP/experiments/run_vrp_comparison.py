"""BAPR-HRO vs baselines on SVRPBench-style VRP instances.

Architecture:
  1. Pre-compute K=10 candidate routes (noisy NN perturbations)
  2. Each method picks one route per episode using its selection policy
  3. Execute the route under stochastic travel times + congestion zones
  4. Learning methods update beliefs from observed delays
  5. Repeat for N episodes (same instance, beliefs carry over)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time as _t

from vrp_env import VRPInstance, generate_instance
from lcb_vrp import (
    generate_candidate_routes,
    StaticNNRouter, LCBRouterV1, LCBRouterV2, TSRouter,
    AdaptiveBetaRouter, HybridRouter, FlowLCBRouter,
    run_episode,
)


def run_experiment(
    n_customers: int = 20,
    n_instances: int = 10,
    n_episodes: int = 15,
    n_candidates: int = 10,
    methods: list[str] | None = None,
    seed_offset: int = 0,
    out_path: str | None = None,
):
    if methods is None:
        methods = ["Static-NN", "TS", "V1-LCB", "V2-LCB", "Adapt-β", "Hybrid", "Flow-LCB"]
    results: dict[str, dict] = {}

    print(f"\n{'='*75}")
    print(f"  VRP Experiment: {n_customers} cust, {n_instances} inst, "
          f"{n_episodes} ep, {n_candidates} candidate routes")
    print(f"{'='*75}")

    header = f"{'Method':<12}  {'Ep1':>6} {'Ep5':>6} {'Ep10':>6} {'Ep15':>6}"
    header += f"  {'Mean':>6} {'Δ%':>6} {'Last5':>6}"
    print(f"\n{header}")
    print("-" * 70)

    static_mean = None

    for method in methods:
        t0 = _t.time()
        all_eps = {i: [] for i in range(n_episodes)}

        for _inst_idx in range(n_instances):
            inst_seed = seed_offset * 1000 + _inst_idx
            instance = generate_instance(
                n_customers=n_customers, seed=inst_seed, n_congestion_zones=2)
            candidates = generate_candidate_routes(
                instance, k=n_candidates, seed=inst_seed)

            # Create router
            if method == "Static-NN":
                router = StaticNNRouter(instance, candidates)
            elif method == "TS":
                router = TSRouter(instance, candidates, seed=inst_seed)
            elif method == "V1-LCB":
                router = LCBRouterV1(instance, candidates, beta0=1.5)
            elif method == "V2-LCB":
                router = LCBRouterV2(instance, candidates,
                                     beta_base=1.0, beta_ood=1.0, seed=inst_seed)
            elif method == "Adapt-β":
                router = AdaptiveBetaRouter(instance, candidates, seed=inst_seed)
            elif method == "Hybrid":
                router = HybridRouter(instance, candidates, beta0=2.0, switch_ep=5)
            elif method == "Flow-LCB":
                router = FlowLCBRouter(instance, candidates, beta0=2.0, commit_duration=5)
            else:
                raise ValueError(method)

            for ep in range(n_episodes):
                metrics = run_episode(instance, router,
                                     start_time=360.0, seed=inst_seed * 100 + ep)
                all_eps[ep].append(metrics["total_time"])

        ep_means = [np.mean(all_eps[i]) for i in range(n_episodes)]
        mean_all = np.mean(ep_means)
        last5 = float(np.mean(ep_means[-5:])) if n_episodes >= 5 else float(mean_all)
        results[method] = {'mean': float(mean_all), 'last5': last5,
                           'ep_means': [float(x) for x in ep_means]}

        if static_mean is None:
            static_mean = mean_all
        delta = (mean_all - static_mean) / static_mean * 100

        last5 = np.mean(ep_means[-5:]) if n_episodes >= 5 else mean_all
        elapsed = _t.time() - t0

        ep1 = ep_means[0]
        ep5 = ep_means[4] if n_episodes > 4 else 0
        ep10 = ep_means[9] if n_episodes > 9 else 0
        ep15 = ep_means[14] if n_episodes > 14 else 0

        print(f"{method:<12}  {ep1:>6.0f} {ep5:>6.0f} {ep10:>6.0f} {ep15:>6.0f}"
              f"  {mean_all:>6.0f} {delta:>+5.1f}% {last5:>6.0f}  ({elapsed:.1f}s)")

    # Learning curve
    print(f"\n--- Learning Curve (avg over {n_instances} instances) ---")
    print(f"{'Ep':<4}", end="")
    for method in methods:
        print(f" {method:>10}", end="")
    print()

    for ep in range(n_episodes):
        print(f"  {ep+1:<2}", end="")
        for method in methods:
            ep_times = []
            for _inst_idx in range(n_instances):
                inst_seed = seed_offset * 1000 + _inst_idx
                instance = generate_instance(
                    n_customers=n_customers, seed=inst_seed, n_congestion_zones=2)
                candidates = generate_candidate_routes(
                    instance, k=n_candidates, seed=inst_seed)

                if method == "Static-NN":
                    router = StaticNNRouter(instance, candidates)
                elif method == "TS":
                    router = TSRouter(instance, candidates, seed=inst_seed)
                elif method == "V1-LCB":
                    router = LCBRouterV1(instance, candidates, beta0=1.5)
                elif method == "V2-LCB":
                    router = LCBRouterV2(instance, candidates,
                                         beta_base=1.0, beta_ood=1.0, seed=inst_seed)
                elif method == "Adapt-β":
                    router = AdaptiveBetaRouter(instance, candidates, seed=inst_seed)
                elif method == "Hybrid":
                    router = HybridRouter(instance, candidates, beta0=2.0, switch_ep=5)
                elif method == "Flow-LCB":
                    router = FlowLCBRouter(instance, candidates, beta0=2.0, commit_duration=5)
                else:
                    raise ValueError(method)

                # Run all episodes up to current
                for prev in range(ep + 1):
                    m = run_episode(instance, router, start_time=360.0,
                                   seed=inst_seed * 100 + prev)
                ep_times.append(m["total_time"])
            print(f" {np.mean(ep_times):>10.1f}", end="")
        print()

    return results


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--out', default=None)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        run_experiment(n_customers=15, n_instances=5, n_episodes=10,
                       n_candidates=10, seed_offset=args.seed_offset)
        print()
    print(f"\n=== Main (20 cust, 10 inst, 15 ep, 15 routes), seed={args.seed_offset} ===")
    res = run_experiment(n_customers=20, n_instances=10, n_episodes=15,
                          n_candidates=15, seed_offset=args.seed_offset,
                          out_path=args.out)
    if args.out and isinstance(res, dict):
        with open(args.out, 'w') as f:
            json.dump({'seed_offset': args.seed_offset, 'results': res},
                      f, indent=2)
        print(f"Saved {args.out}")
