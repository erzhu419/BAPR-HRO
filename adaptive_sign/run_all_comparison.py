"""Final comparison: Adaptive-Sign BAPR vs all baselines on 4 domains."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import time as _t
from adaptive_lcb import AdaptiveSignRouter, AdaptiveSignLinkRouter


def run_power_dispatch(n_seeds=5):
    """Power Dispatch: 10-gen UC with wind regime shifts."""
    sys.path.insert(0, '../power_dispatch')
    from uc_env import generate_candidate_schedules, execute_schedule, _load_rl4uc_env
    from lcb_uc import (StaticRouter, LCBRouter, LCBRouterV2, TSRouter, HybridRouter)

    env = _load_rl4uc_env(num_gen=10, voll=500)
    fd = env.profiles_df[env.profiles_df['date']==env.profiles_df['date'].unique()[0]]['demand'].values[:48]
    fw = env.profiles_df[env.profiles_df['date']==env.profiles_df['date'].unique()[0]]['wind'].values[:48]
    scheds = generate_candidate_schedules(num_gen=10, n_candidates=6,
                                          forecast_demand=fd, forecast_wind=fw,
                                          gen_max=env.max_output)
    warm_costs = [execute_schedule(s, num_gen=10, seed=9999, voll=500).total_cost for s in scheds]
    wind_regimes = ['normal']*15 + ['low_wind']*10 + ['normal']*15
    n_days = 40

    methods = {
        'Static':   lambda s: StaticRouter(6),
        'TS':       lambda s: TSRouter(6, seed=s, warm_costs=warm_costs),
        'V1-LCB':   lambda s: LCBRouter(6, beta0=2.0, warm_costs=warm_costs),
        'V2-LCB':   lambda s: LCBRouterV2(6, beta_base=1.5, beta_ood=0.5, seed=s, warm_costs=warm_costs),
        'Hybrid':   lambda s: HybridRouter(6, beta0=2.0, switch_ep=10, warm_costs=warm_costs),
        'Adaptive': lambda s: AdaptiveSignRouter(6, beta0=2.0, warmup=20, warm_costs=warm_costs),
    }

    results = {}
    for mname, make in methods.items():
        seed_costs = []
        for seed in range(n_seeds):
            router = make(seed)
            costs = []
            for day in range(n_days):
                if mname == 'Adaptive':
                    idx = router.select()
                else:
                    idx = router.select_schedule()
                r = execute_schedule(scheds[idx], num_gen=10, seed=day+seed*100,
                                     voll=500, wind_regime=wind_regimes[day])
                if mname == 'Adaptive':
                    router.observe(idx, r.total_cost)
                else:
                    router.observe(idx, r)
                costs.append(r.total_cost)
            seed_costs.append(np.mean(costs))
        results[mname] = np.mean(seed_costs)

    return results


def run_vrp(n_seeds=10):
    """VRP: 25 customers, 3 congestion zones."""
    sys.path.insert(0, '../VRP')
    from vrp_env import generate_instance
    from lcb_vrp import (generate_candidate_routes, StaticNNRouter, LCBRouterV1,
                         LCBRouterV2, TSRouter, run_episode)

    n_ep = 20
    n_cands = 10

    methods_arm = ['Static-NN', 'TS', 'V1-LCB', 'V2-LCB', 'Adaptive']
    results = {m: [] for m in methods_arm}

    for inst_seed in range(n_seeds):
        inst = generate_instance(n_customers=25, seed=inst_seed, n_congestion_zones=3)
        cands = generate_candidate_routes(inst, k=n_cands, seed=inst_seed)

        for mname in methods_arm:
            if mname == 'Static-NN':
                router = StaticNNRouter(inst, cands)
            elif mname == 'TS':
                router = TSRouter(inst, cands, seed=inst_seed, explore_top=4)
            elif mname == 'V1-LCB':
                router = LCBRouterV1(inst, cands, beta0=2.0, explore_top=4)
            elif mname == 'V2-LCB':
                router = LCBRouterV2(inst, cands, beta_base=0.8, beta_ood=0.8,
                                     seed=inst_seed, explore_top=4)
            elif mname == 'Adaptive':
                router = AdaptiveSignRouter(n_arms=len(cands), beta0=2.0, warmup=10)

            ep_costs = []
            for ep in range(n_ep):
                if mname == 'Adaptive':
                    idx = router.select()
                    from lcb_vrp import _execute_route
                    metrics, steps = _execute_route(inst, cands[idx], start_time=360,
                                                    seed=inst_seed*100+ep)
                    router.observe(idx, metrics['total_time'])
                    ep_costs.append(metrics['total_time'])
                else:
                    m = run_episode(inst, router, start_time=360, seed=inst_seed*100+ep)
                    ep_costs.append(m['total_time'])

            results[mname].append(np.mean(ep_costs[5:]))  # post-exploration

    return {m: np.mean(v) for m, v in results.items()}


def run_sdn(n_seeds=5):
    """SDN Routing: NSFNet, 3 regime shifts."""
    sys.path.insert(0, '../sdn_routing')
    from sdn_env import (SDNEnv, StaticRouter, LCBRouter, LCBRouterV2,
                         ReactUCBRouter, TSRouter)

    n_ep = 100
    n_dem = 20

    methods = {
        'Static':     lambda s: StaticRouter(),
        'React-UCB':  lambda s: ReactUCBRouter(c=1.0, gamma=0.95),
        'TS':         lambda s: TSRouter(seed=s),
        'V1-LCB':    lambda s: LCBRouter(beta=1.0),
        'Adaptive':   lambda s: AdaptiveSignLinkRouter(beta0=2.0, warmup=30),
    }

    # Pre-generate all (src, dst) pairs per seed for fairness
    all_seed_pairs = {}
    for seed in range(n_seeds):
        pair_rng = np.random.default_rng(seed + 1000)
        seed_pairs = []
        for ep in range(n_ep):
            ep_pairs = []
            for _ in range(n_dem):
                s = int(pair_rng.integers(0, 14))
                d = int(pair_rng.integers(0, 13))
                if d >= s: d += 1
                ep_pairs.append((s, d))
            seed_pairs.append(ep_pairs)
        all_seed_pairs[seed] = seed_pairs

    results = {}
    for mname, make in methods.items():
        seed_delays = []
        for seed in range(n_seeds):
            # FRESH env per method+seed — critical for independent RNG
            env = SDNEnv(topology='nsfnet', seed=seed, n_regime_shifts=3,
                         total_episodes=n_ep)
            router = make(seed)

            ep_delays = []
            for ep in range(n_ep):
                total = 0
                for di, (s, d) in enumerate(all_seed_pairs[seed][ep]):
                    paths = env.get_paths(s, d)
                    if not paths: continue
                    pi = min(router.select_path(paths, src=s, dst=d), len(paths)-1)
                    delay = env.sample_path_delay(paths[pi], ep, demand_idx=di)
                    router.observe(pi, delay, paths=paths, src=s, dst=d)
                    total += delay
                ep_delays.append(total / n_dem)
                env.step_episode()

            seed_delays.append(np.mean(ep_delays))
        results[mname] = np.mean(seed_delays)

    return results


def run_transit(n_seeds=5):
    """Transit Routing: Durner-style synthetic network."""
    from src.synthetic_network import create_bus_story_network
    from src.router import StaticRouter
    from src.bandit_router import BanditRouter
    from src.bandit_router_v2 import BanditRouterV2
    from src.simulate_bandit import simulate_bandit_journey
    from src.simulator import RegimeSchedule

    schedule = RegimeSchedule(shifts=[(0, 'normal'), (490, 'disrupted_402'), (540, 'normal')])
    n_journeys = 30

    methods = {
        'Static':  lambda s: StaticRouter(create_bus_story_network()),
        'V1-LCB':  lambda s: BanditRouter(create_bus_story_network()),
        'V2-LCB':  lambda s: BanditRouterV2(create_bus_story_network(), seed=s),
    }

    results = {}
    for mname, make in methods.items():
        seed_tts = []
        for seed in range(n_seeds):
            graph = create_bus_story_network()
            router = make(seed)
            tts = []
            for i in range(n_journeys):
                rng = np.random.default_rng(seed*100+i)
                jr = simulate_bandit_journey(graph, router, 0, 9, 480+i,
                                             schedule, rng)
                tts.append(jr.arrival_time - jr.departure_time)
            seed_tts.append(np.mean(tts))
        results[mname] = np.mean(seed_tts)

    return results


if __name__ == "__main__":
    print("="*80)
    print("  BAPR-HRO Final Comparison: All Methods × All Domains")
    print("="*80)

    # Transit
    print("\n--- 1. Transit Routing (disrupted_402, 30 journeys, 5 seeds) ---")
    t0 = _t.time()
    transit = run_transit()
    static = transit['Static']
    print(f"{'Method':<14} {'Mean TT':>8} {'Δ%':>7}")
    for m, v in transit.items():
        print(f"  {m:<12} {v:>8.1f} {(v-static)/static*100:>+6.1f}%")
    print(f"  ({_t.time()-t0:.0f}s)")

    # Power
    print("\n--- 2. Power Dispatch (10gen, regime shift, 40 days, 5 seeds) ---")
    t0 = _t.time()
    power = run_power_dispatch()
    static = power['Static']
    print(f"{'Method':<14} {'Mean $':>10} {'Δ%':>7}")
    for m, v in power.items():
        print(f"  {m:<12} {v:>10.0f} {(v-static)/static*100:>+6.1f}%")
    print(f"  ({_t.time()-t0:.0f}s)")

    # VRP
    print("\n--- 3. VRP (25 cust, 3 zones, 20 ep, 10 seeds) ---")
    t0 = _t.time()
    vrp = run_vrp()
    static = vrp['Static-NN']
    print(f"{'Method':<14} {'Post-expl':>10} {'Δ%':>7}")
    for m, v in vrp.items():
        print(f"  {m:<12} {v:>10.1f} {(v-static)/static*100:>+6.1f}%")
    print(f"  ({_t.time()-t0:.0f}s)")

    # SDN
    print("\n--- 4. SDN Routing (NSFNet, 100 ep, 3 shifts, 5 seeds) ---")
    t0 = _t.time()
    sdn = run_sdn()
    static = sdn['Static']
    print(f"{'Method':<14} {'Mean delay':>10} {'Δ%':>7}")
    for m, v in sdn.items():
        print(f"  {m:<12} {v:>10.2f} {(v-static)/static*100:>+6.1f}%")
    print(f"  ({_t.time()-t0:.0f}s)")

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY: Best method per domain")
    print("="*80)
    for domain, results, expected in [
        ("Transit", transit, "V1-LCB"),
        ("Power", power, "V1-LCB"),
        ("VRP", vrp, "V1-LCB"),
        ("SDN", sdn, "React-UCB"),
    ]:
        best = min(results.items(), key=lambda x: x[1])
        static_v = list(results.values())[0]
        print(f"  {domain:<10}: best={best[0]:<12} ({(best[1]-static_v)/static_v*100:+.1f}%)  "
              f"expected={expected}")
