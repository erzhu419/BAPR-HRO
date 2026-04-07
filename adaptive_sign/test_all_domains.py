"""Test Adaptive-Sign BAPR on all 3 domains with multiple seeds."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from adaptive_lcb import AdaptiveSignRouter, AdaptiveSignLinkRouter


def test_power_dispatch(seed=0):
    sys.path.insert(0, '../power_dispatch')
    from uc_env import generate_candidate_schedules, execute_schedule, _load_rl4uc_env

    env = _load_rl4uc_env(num_gen=10, voll=500)
    fd = env.profiles_df[env.profiles_df['date']==env.profiles_df['date'].unique()[0]]['demand'].values[:48]
    fw = env.profiles_df[env.profiles_df['date']==env.profiles_df['date'].unique()[0]]['wind'].values[:48]
    scheds = generate_candidate_schedules(num_gen=10, n_candidates=6,
                                          forecast_demand=fd, forecast_wind=fw,
                                          gen_max=env.max_output)

    warm_costs = [execute_schedule(s, num_gen=10, seed=9999, voll=500).total_cost for s in scheds]
    wind_regimes = ['normal']*15 + ['low_wind']*10 + ['normal']*15

    router = AdaptiveSignRouter(n_arms=6, beta0=2.0, warmup=20, warm_costs=warm_costs)
    costs = []
    for day in range(40):
        idx = router.select()
        r = execute_schedule(scheds[idx], num_gen=10, seed=day+seed*100, voll=500,
                             wind_regime=wind_regimes[day])
        router.observe(idx, r.total_cost)
        costs.append(r.total_cost)
    return np.mean(costs), router.sign_factor, router.mode


def test_sdn(seed=0):
    sys.path.insert(0, '../sdn_routing')
    from sdn_env import SDNEnv

    env = SDNEnv(topology='nsfnet', seed=seed, n_regime_shifts=3, total_episodes=100)
    pair_rng = np.random.default_rng(seed+1000)
    all_pairs = []
    for ep in range(100):
        pairs = [(int(pair_rng.integers(0,14)),
                  (lambda s: (s, int(pair_rng.integers(0,13))))(int(pair_rng.integers(0,14))))
                 for _ in range(20)]
        pairs = [(s, d+1 if d>=s else d) for s,(_, d) in pairs]
        all_pairs.append(pairs)

    router = AdaptiveSignLinkRouter(beta0=2.0, warmup=40)
    delays = []
    for ep in range(100):
        total = 0
        for src, dst in all_pairs[ep]:
            paths = env.get_paths(src, dst)
            if not paths: continue
            pi = min(router.select_path(paths, src=src, dst=dst), len(paths)-1)
            d = env.sample_path_delay(paths[pi], ep)
            router.observe(pi, d, paths=paths, src=src, dst=dst)
            total += d
        delays.append(total / 20)
        env.step_episode()
    return np.mean(delays), router.sign_factor, router.mode


def test_vrp(seed=0):
    sys.path.insert(0, '../VRP')
    from vrp_env import generate_instance
    from lcb_vrp import generate_candidate_routes, _execute_route

    inst = generate_instance(n_customers=25, seed=seed, n_congestion_zones=3)
    cands = generate_candidate_routes(inst, k=8, seed=seed)
    router = AdaptiveSignRouter(n_arms=len(cands), beta0=2.0, warmup=10)

    costs = []
    for ep in range(20):
        idx = router.select()
        if idx < len(cands):
            metrics, steps = _execute_route(inst, cands[idx], start_time=360, seed=seed*100+ep)
            router.observe(idx, metrics['total_time'])
            costs.append(metrics['total_time'])
    return np.mean(costs) if costs else 0, router.sign_factor, router.mode


if __name__ == "__main__":
    print("="*70)
    print("  Adaptive-Sign BAPR: auto-detect LCB vs UCB (multi-seed)")
    print("="*70)

    for domain, test_fn, n_seeds, expected in [
        ("Power Dispatch", test_power_dispatch, 5, "LCB"),
        ("SDN Routing", test_sdn, 5, "UCB"),
        ("VRP", test_vrp, 10, "LCB"),
    ]:
        sfs = []
        modes = []
        costs = []
        for s in range(n_seeds):
            c, sf, mode = test_fn(seed=s)
            sfs.append(sf)
            modes.append(mode)
            costs.append(c)

        avg_sf = np.mean(sfs)
        lcb_count = sum(1 for m in modes if m == "LCB")
        ucb_count = sum(1 for m in modes if m == "UCB")
        detected = "LCB" if lcb_count > ucb_count else ("UCB" if ucb_count > lcb_count else "~0")

        print(f"\n{domain}:")
        print(f"  sign_factors: [{', '.join(f'{s:.3f}' for s in sfs)}]")
        print(f"  avg sign_factor: {avg_sf:+.3f}")
        print(f"  detected modes: LCB={lcb_count}/{n_seeds} UCB={ucb_count}/{n_seeds}")
        print(f"  → Detected: {detected}  (expected: {expected})  {'✓' if detected == expected else '✗'}")
