"""Final comparison: ALL methods × ALL 4 domains.

Key question: does Adaptive-Sign match the best specialist in each domain?
  - In LCB-favoring domains (Transit, Power, VRP): should ≈ V1-LCB
  - In UCB-favoring domains (SDN): should ≈ React-UCB
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import time as _t
from adaptive_lcb import AdaptiveSignRouter, AdaptiveSignLinkRouter


# ── 1. Transit ───────────────────────────────────────────────────────────────

def run_transit(n_seeds=5):
    from src.synthetic_network import create_bus_story_network
    from src.router import StaticRouter
    from src.bandit_router import BanditRouter
    from src.bandit_router_v2 import BanditRouterV2
    from src.simulate_bandit import simulate_bandit_journey
    from src.simulator import RegimeSchedule

    schedule = RegimeSchedule(shifts=[(0, 'normal'), (490, 'disrupted_402'), (540, 'normal')])
    n_j = 30

    results = {}
    for mname in ['Static', 'V1-LCB', 'V2-LCB']:
        seed_tts = []
        for seed in range(n_seeds):
            graph = create_bus_story_network()
            if mname == 'Static':
                router = StaticRouter(graph)
            elif mname == 'V1-LCB':
                router = BanditRouter(graph)
            elif mname == 'V2-LCB':
                router = BanditRouterV2(graph, seed=seed)
            tts = []
            for i in range(n_j):
                jr = simulate_bandit_journey(graph, router, 0, 9, 480+i, schedule,
                                             np.random.default_rng(seed*100+i))
                tts.append(jr.arrival_time - jr.departure_time)
            seed_tts.append(np.mean(tts))
        results[mname] = np.mean(seed_tts)

    # Adaptive: use arm-level on route-connection choices
    # Transit doesn't map cleanly to arm-level, so we skip it here
    # (Transit's select_connection is per-stop, not per-route)
    # Instead note: V1-LCB IS the adaptive choice for transit

    return results


# ── 2. Power Dispatch ────────────────────────────────────────────────────────

def run_power(n_seeds=5):
    sys.path.insert(0, '../power_dispatch')
    from uc_env import generate_candidate_schedules, execute_schedule, _load_rl4uc_env
    from lcb_uc import StaticRouter, LCBRouter, TSRouter

    env = _load_rl4uc_env(num_gen=10, voll=500)
    fd = env.profiles_df[env.profiles_df['date']==env.profiles_df['date'].unique()[0]]['demand'].values[:48]
    fw = env.profiles_df[env.profiles_df['date']==env.profiles_df['date'].unique()[0]]['wind'].values[:48]
    scheds = generate_candidate_schedules(num_gen=10, n_candidates=6,
                                          forecast_demand=fd, forecast_wind=fw,
                                          gen_max=env.max_output)
    warm = [execute_schedule(s, num_gen=10, seed=9999, voll=500).total_cost for s in scheds]
    regimes = ['normal']*15 + ['low_wind']*10 + ['normal']*15

    def _run_method(make_router, n_seeds):
        costs = []
        for seed in range(n_seeds):
            router = make_router(seed)
            for day in range(40):
                if hasattr(router, 'select'):
                    idx = router.select()
                else:
                    idx = router.select_schedule()
                r = execute_schedule(scheds[idx], num_gen=10, seed=day+seed*100,
                                     voll=500, wind_regime=regimes[day])
                if hasattr(router, 'select'):
                    router.observe(idx, r.total_cost)
                else:
                    router.observe(idx, r)
                costs.append(r.total_cost)
        return np.mean(costs)

    return {
        'Static':   _run_method(lambda s: StaticRouter(6), n_seeds),
        'TS':       _run_method(lambda s: TSRouter(6, seed=s, warm_costs=warm), n_seeds),
        'V1-LCB':   _run_method(lambda s: LCBRouter(6, beta0=2.0, warm_costs=warm), n_seeds),
        'Adaptive': _run_method(lambda s: AdaptiveSignRouter(6, beta0=2.0, calibrate_eps=10, warm_costs=warm), n_seeds),
    }


# ── 3. VRP ───────────────────────────────────────────────────────────────────

def run_vrp(n_seeds=10):
    sys.path.insert(0, '../VRP')
    from vrp_env import generate_instance
    from lcb_vrp import (generate_candidate_routes, StaticNNRouter, LCBRouterV1,
                         TSRouter, run_episode, _execute_route)

    n_ep, n_cands = 20, 8

    def _run_method(mname, n_seeds):
        all_costs = []
        for inst_seed in range(n_seeds):
            inst = generate_instance(n_customers=25, seed=inst_seed, n_congestion_zones=3)
            cands = generate_candidate_routes(inst, k=n_cands, seed=inst_seed)

            if mname == 'Static-NN':
                router = StaticNNRouter(inst, cands)
            elif mname == 'TS':
                router = TSRouter(inst, cands, seed=inst_seed, explore_top=4)
            elif mname == 'V1-LCB':
                router = LCBRouterV1(inst, cands, beta0=2.0, explore_top=4)
            elif mname == 'Adaptive':
                router = AdaptiveSignRouter(n_arms=len(cands), beta0=2.0, calibrate_eps=6)

            for ep in range(n_ep):
                if mname == 'Adaptive':
                    idx = router.select()
                    metrics, steps = _execute_route(inst, cands[idx], start_time=360,
                                                    seed=inst_seed*100+ep)
                    router.observe(idx, metrics['total_time'])
                    if ep >= 5:  # post-exploration
                        all_costs.append(metrics['total_time'])
                else:
                    m = run_episode(inst, router, start_time=360, seed=inst_seed*100+ep)
                    if ep >= 5:
                        all_costs.append(m['total_time'])
        return np.mean(all_costs)

    return {
        'Static-NN': _run_method('Static-NN', n_seeds),
        'TS':        _run_method('TS', n_seeds),
        'V1-LCB':    _run_method('V1-LCB', n_seeds),
        'Adaptive':  _run_method('Adaptive', n_seeds),
    }


# ── 4. SDN ───────────────────────────────────────────────────────────────────

def run_sdn(n_seeds=5):
    sys.path.insert(0, '../sdn_routing')
    from sdn_env import SDNEnv, StaticRouter, LCBRouter, ReactUCBRouter, TSRouter

    n_ep, n_dem = 100, 20

    def _run_method(mname, n_seeds):
        seed_delays = []
        for seed in range(n_seeds):
            env = SDNEnv(topology='nsfnet', seed=seed, n_regime_shifts=3, total_episodes=n_ep)
            if mname == 'Static':
                router = StaticRouter()
            elif mname == 'React-UCB':
                router = ReactUCBRouter(c=1.0, gamma=0.95)
            elif mname == 'TS':
                router = TSRouter(seed=seed)
            elif mname == 'V1-LCB':
                router = LCBRouter(beta=1.0)
            elif mname == 'Adaptive':
                router = AdaptiveSignLinkRouter(beta0=2.0, calibrate_eps=20)

            pair_rng = np.random.default_rng(seed + 1000)
            ep_delays = []
            for ep in range(n_ep):
                total = 0
                for _ in range(n_dem):
                    s = int(pair_rng.integers(0, 14))
                    d = int(pair_rng.integers(0, 13))
                    if d >= s: d += 1
                    paths = env.get_paths(s, d)
                    if not paths: continue
                    pi = min(router.select_path(paths, src=s, dst=d), len(paths)-1)
                    delay = env.sample_path_delay(paths[pi], ep)
                    router.observe(pi, delay, paths=paths, src=s, dst=d)
                    total += delay
                ep_delays.append(total / n_dem)
                env.step_episode()
            seed_delays.append(np.mean(ep_delays))
        return np.mean(seed_delays)

    return {
        'Static':     _run_method('Static', n_seeds),
        'React-UCB':  _run_method('React-UCB', n_seeds),
        'TS':         _run_method('TS', n_seeds),
        'V1-LCB':    _run_method('V1-LCB', n_seeds),
        'Adaptive':   _run_method('Adaptive', n_seeds),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*85)
    print("  BAPR-HRO Final: Adaptive-Sign vs All Specialists × 4 Domains")
    print("="*85)

    domains = [
        ("1. Transit Routing", run_transit, "V1-LCB"),
        ("2. Power Dispatch",  run_power,   "V1-LCB"),
        ("3. VRP",             run_vrp,     "V1-LCB"),
        ("4. SDN Routing",     run_sdn,     "React-UCB"),
    ]

    all_results = {}
    for title, fn, expected_best in domains:
        t0 = _t.time()
        results = fn()
        elapsed = _t.time() - t0

        static_key = [k for k in results if 'Static' in k][0]
        static_v = results[static_key]

        print(f"\n--- {title} ({elapsed:.0f}s) ---")
        print(f"  {'Method':<14} {'Value':>10} {'vs Static':>10}")
        for m, v in results.items():
            delta = (v - static_v) / static_v * 100
            marker = " <-- best" if m == min(results, key=results.get) else ""
            print(f"  {m:<14} {v:>10.1f} {delta:>+9.1f}%{marker}")

        all_results[title] = results

    # Summary table
    print("\n" + "="*85)
    print("  SUMMARY")
    print("="*85)
    print(f"\n  {'Domain':<20} {'Best Specialist':>16} {'Adaptive':>16} {'Gap':>8}")
    print("  " + "-"*64)

    for title, fn, expected_best in domains:
        results = all_results[title]
        if 'Adaptive' not in results:
            print(f"  {title:<20} {'(no Adaptive)':>16}")
            continue

        static_key = [k for k in results if 'Static' in k][0]
        static_v = results[static_key]

        # Best non-static, non-adaptive
        specialists = {k: v for k, v in results.items()
                       if 'Static' not in k and k != 'Adaptive'}
        best_spec = min(specialists, key=specialists.get)
        best_v = specialists[best_spec]
        adaptive_v = results['Adaptive']

        spec_delta = (best_v - static_v) / static_v * 100
        adap_delta = (adaptive_v - static_v) / static_v * 100
        gap = adap_delta - spec_delta  # positive = adaptive worse

        spec_str = f"{best_spec} {spec_delta:+.1f}%"
        adap_str = f"Adaptive {adap_delta:+.1f}%"
        print(f"  {title:<20} {spec_str:>20} {adap_str:>20} {gap:>+7.1f}pp")
