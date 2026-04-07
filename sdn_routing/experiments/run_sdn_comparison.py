"""BAPR-HRO vs React-UCB on SDN routing (NSFNet/GEANT2).

Each episode: N random (src, dst) demands arrive. For each demand,
the router picks one of K=4 candidate paths. The actual delay is
sampled from the environment (stochastic + regime shifts).
Over episodes, routers learn link delays and adapt path selection.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time as _t
from sdn_env import (SDNEnv, StaticRouter, LCBRouter, LCBRouterV2,
                      ReactUCBRouter, TSRouter)


def run_experiment(
    topology: str = "nsfnet",
    n_episodes: int = 100,
    demands_per_episode: int = 20,
    n_regime_shifts: int = 2,
    methods: list[str] | None = None,
    seed: int = 42,
):
    if methods is None:
        methods = ["Static", "React-UCB", "TS", "V1-LCB", "V2-LCB"]

    print(f"\n{'='*75}")
    print(f"  SDN Routing: {topology}, {n_episodes} episodes, "
          f"{demands_per_episode} demands/ep, {n_regime_shifts} regime shifts")
    print(f"{'='*75}")

    # Show regime shifts
    env_probe = SDNEnv(topology=topology, seed=seed,
                       n_regime_shifts=n_regime_shifts,
                       total_episodes=n_episodes)
    for rs in env_probe.regime_shifts:
        print(f"  Regime: {rs.shift_type} on {rs.affected_links} "
              f"ep {rs.time_start}-{rs.time_end} sev={rs.severity:.1f}")

    print(f"\n{'Method':<12} {'Ep1-10':>8} {'Ep50':>8} {'Ep90-100':>8}"
          f"  {'Mean':>8} {'Δ%':>6} {'During':>8} {'After':>8}")
    print("-" * 78)

    static_mean = None

    # Find regime shift periods for analysis
    shifts = env_probe.regime_shifts
    shift_eps = set()
    for rs in shifts:
        for e in range(rs.time_start, rs.time_end):
            shift_eps.add(e)

    for method in methods:
        t0 = _t.time()

        env = SDNEnv(topology=topology, seed=seed,
                     n_regime_shifts=n_regime_shifts,
                     total_episodes=n_episodes)

        if method == "Static":
            router = StaticRouter()
        elif method == "React-UCB":
            router = ReactUCBRouter(c=1.0, gamma=0.9)
        elif method == "TS":
            router = TSRouter(seed=seed)
        elif method == "V1-LCB":
            router = LCBRouter(beta=1.0)
        elif method == "V2-LCB":
            router = LCBRouterV2(beta_base=0.8, beta_ood=0.8, seed=seed)

        ep_delays = []
        during_shift_delays = []
        after_shift_delays = []

        # Generate fixed (src, dst) pairs for fair comparison
        pair_rng = np.random.default_rng(seed + 1000)
        all_pairs = []
        for ep in range(n_episodes):
            pairs = []
            for _ in range(demands_per_episode):
                src = int(pair_rng.integers(0, env.n_nodes))
                dst = int(pair_rng.integers(0, env.n_nodes - 1))
                if dst >= src:
                    dst += 1
                pairs.append((src, dst))
            all_pairs.append(pairs)

        for ep in range(n_episodes):
            ep_total_delay = 0.0
            for src, dst in all_pairs[ep]:
                paths = env.get_paths(src, dst)
                if not paths:
                    continue

                path_idx = router.select_path(paths, src=src, dst=dst)
                path_idx = min(path_idx, len(paths) - 1)
                delay = env.sample_path_delay(paths[path_idx], ep)
                router.observe(path_idx, delay, src=src, dst=dst, paths=paths)
                ep_total_delay += delay

            avg_delay = ep_total_delay / max(demands_per_episode, 1)
            ep_delays.append(avg_delay)

            if ep in shift_eps:
                during_shift_delays.append(avg_delay)
            elif ep > max((rs.time_end for rs in shifts), default=0):
                after_shift_delays.append(avg_delay)

            env.step_episode()

        mean_all = np.mean(ep_delays)
        if static_mean is None:
            static_mean = mean_all
        delta = (mean_all - static_mean) / static_mean * 100

        early = np.mean(ep_delays[:10])
        mid = ep_delays[49] if n_episodes > 49 else 0
        late = np.mean(ep_delays[-10:])
        during = np.mean(during_shift_delays) if during_shift_delays else 0
        after = np.mean(after_shift_delays) if after_shift_delays else 0
        elapsed = _t.time() - t0

        print(f"{method:<12} {early:>8.2f} {mid:>8.2f} {late:>8.2f}"
              f"  {mean_all:>8.2f} {delta:>+5.1f}% {during:>8.2f} {after:>8.2f}"
              f"  ({elapsed:.1f}s)")

    # Learning curve
    print(f"\n--- Learning curve (avg delay per episode, first 5 methods) ---")
    milestones = [1, 5, 10, 20, 30, 50, 70, 90, 100]
    milestones = [m for m in milestones if m <= n_episodes]
    print(f"{'Ep':<4}", end="")
    for m in methods:
        print(f" {m:>10}", end="")
    print()
    for ms in milestones:
        print(f"  {ms:<2}", end="")
        for method in methods:
            env = SDNEnv(topology=topology, seed=seed,
                         n_regime_shifts=n_regime_shifts,
                         total_episodes=n_episodes)
            if method == "Static":
                router = StaticRouter()
            elif method == "React-UCB":
                router = ReactUCBRouter(c=1.0, gamma=0.9)
            elif method == "TS":
                router = TSRouter(seed=seed)
            elif method == "V1-LCB":
                router = LCBRouter(beta=1.0)
            elif method == "V2-LCB":
                router = LCBRouterV2(beta_base=0.8, beta_ood=0.8, seed=seed)

            for ep in range(ms):
                for src, dst in all_pairs[ep]:
                    paths = env.get_paths(src, dst)
                    if not paths:
                        continue
                    pi = router.select_path(paths, src=src, dst=dst)
                    pi = min(pi, len(paths) - 1)
                    d = env.sample_path_delay(paths[pi], ep)
                    router.observe(pi, d, src=src, dst=dst, paths=paths)
                env.step_episode()

            # Delay at milestone episode
            last_delay = 0
            env2 = SDNEnv(topology=topology, seed=seed,
                          n_regime_shifts=n_regime_shifts,
                          total_episodes=n_episodes)
            for src, dst in all_pairs[ms - 1]:
                paths = env2.get_paths(src, dst)
                if not paths:
                    continue
                pi = router.select_path(paths, src=src, dst=dst)
                pi = min(pi, len(paths) - 1)
                d = env2.sample_path_delay(paths[pi], ms - 1)
                router.observe(pi, d, src=src, dst=dst, paths=paths)
                last_delay += d
            last_delay /= max(demands_per_episode, 1)
            print(f" {last_delay:>10.2f}", end="")
        print()


if __name__ == "__main__":
    run_experiment(topology="nsfnet", n_episodes=100,
                   demands_per_episode=20, n_regime_shifts=3, seed=42)
