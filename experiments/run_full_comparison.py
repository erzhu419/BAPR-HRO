"""Full comparison: all methods × small + large networks × all scenarios.

Produces the complete Table 1 (primary) and Table 2 (extended + new baselines)
for the paper, plus sensitivity analyses (network scale, disruption intensity).

Methods compared:
  Proposed:
    LCB-V1   : Normal-Gamma posterior, fixed β=1.5
    LCB-V2   : Ensemble LCB, dynamic β (best practical method)
    DRO      : Wasserstein DRO (formally equivalent to V1-LCB)
  Classic Bayesian RL (2012–2021):
    PS-SSP   : Posterior Sampling (Thompson Sampling over full MDP)
    BAMCP-60 : Bayes-Adaptive MCTS, 60 simulations
  Static / periodic:
    Static   : Durner hyperpath, no adaptation
  New 2024 baselines:
    SW-LCB   : Sliding-Window LCB (Garivier & Moulines 2011 / Luo et al. 2024)
    EXP3     : EXP3-IX adversarial bandit (Neu et al. 2010 / Chen et al. 2024)
    Oracle   : Perfect-information upper bound (knows true regime)
"""

from __future__ import annotations

import sys
import os
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.synthetic_network import create_bus_story_network, create_regime_distributions
from src.large_network import create_grid_network, create_grid_regime_distributions
from src.router import StaticRouter
from src.bandit_router import BanditRouter
from src.bandit_router_v2 import BanditRouterV2
from src.ssp_mdp import PosteriorSamplingRouter
from src.bamcp_router import BAMCPRouter
from src.dro_router import DRORouter
from src.sw_lcb_router import SWLCBRouter
from src.exp3_router import EXP3Router
from src.oracle_router import OracleRouter
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import RegimeSchedule, set_regime_dist_fn, _regime_dist_cache


# -----------------------------------------------------------------------
# Helper: build regime schedule by name
# -----------------------------------------------------------------------

SMALL_SCHEDULES = {
    "normal":    RegimeSchedule(shifts=[(0, "normal")]),
    "disrupted": RegimeSchedule(shifts=[(0, "normal"), (490, "disrupted_402"), (540, "normal")]),
    "rush_hour": RegimeSchedule(shifts=[(0, "normal"), (480, "rush_hour"), (570, "normal")]),
    "multi":     RegimeSchedule(shifts=[(0, "normal"), (485, "rush_hour"),
                                        (510, "disrupted_402"), (540, "normal")]),
}

LARGE_SCHEDULES = {
    "normal":    RegimeSchedule(shifts=[(0, "normal")]),
    "disrupted": RegimeSchedule(shifts=[(0, "normal"), (490, "central_disruption"), (560, "normal")]),
    "full_chaos":RegimeSchedule(shifts=[(0, "normal"), (485, "central_disruption"),
                                        (520, "full_chaos"), (560, "normal")]),
}


def make_router(name: str, graph, net: str, seed: int, schedule: RegimeSchedule,
                regime_dist_fn):
    """Instantiate a router by name."""
    if name == "Static":
        return StaticRouter(graph)
    elif name == "LCB-V1":
        return BanditRouter(graph)
    elif name == "LCB-V2":
        return BanditRouterV2(graph, n_estimators=5, beta_base=1.0, beta_ood=1.0, seed=seed)
    elif name == "DRO":
        return DRORouter(graph, beta=1.5, gamma=60.0)
    elif name == "PS-SSP":
        return PosteriorSamplingRouter(graph)
    elif name == "BAMCP-60":
        return BAMCPRouter(graph, n_simulations=60)
    elif name == "BAMCP-120":
        return BAMCPRouter(graph, n_simulations=120)
    elif name == "BAMCP-240":
        return BAMCPRouter(graph, n_simulations=240)
    elif name == "SW-LCB":
        return SWLCBRouter(graph, window_size=20, beta=1.5, gamma=60.0)
    elif name == "EXP3":
        return EXP3Router(graph, gamma=0.1, eta=0.05, cancel_cost=60.0)
    elif name == "Oracle":
        return OracleRouter(
            graph,
            regime_dist_fn=regime_dist_fn,
            regime_schedule_fn=schedule.get_regime,
            cancel_cost=60.0,
        )
    else:
        raise ValueError(f"Unknown method: {name}")


# -----------------------------------------------------------------------
# Core experiment runner
# -----------------------------------------------------------------------

def run_experiment(
    net: str,
    scenario: str,
    methods: list[str],
    n_journeys: int = 100,
    seed: int = 42,
) -> dict[str, dict]:
    """Run one (network × scenario) cell for all methods.

    Returns dict: method_name → {mean, median, p95, timeout_pct, n_obs}
    """
    if net == "small":
        create_graph = create_bus_story_network
        schedules = SMALL_SCHEDULES
        regime_dist_fn = create_regime_distributions
        s_source, s_dest = 0, 9
        max_time = 120
    elif net == "large":
        create_graph = create_grid_network
        schedules = LARGE_SCHEDULES
        regime_dist_fn = create_grid_regime_distributions
        s_source, s_dest = 0, 48
        max_time = 180
    else:
        raise ValueError(f"Unknown network: {net}")

    schedule = schedules[scenario]
    set_regime_dist_fn(regime_dist_fn)

    results = {}
    rng = np.random.default_rng(seed)

    for method in methods:
        _regime_dist_cache.clear()
        travel_times = []
        obs_counts = []
        t0 = time.time()

        for i in range(n_journeys):
            graph = create_graph()
            router = make_router(method, graph, net, seed + i, schedule, regime_dist_fn)
            t_dep = 480 + rng.integers(0, 20)
            journey_rng = np.random.default_rng(seed + i * 31337)

            result = simulate_bandit_journey(
                graph=graph,
                router=router,
                s_source=s_source,
                s_dest=s_dest,
                t_depart=t_dep,
                regime_schedule=schedule,
                rng=journey_rng,
                max_time=max_time,
            )
            travel_times.append(result.arrival_time - result.departure_time)
            obs_counts.append(result.n_replans)

        arr = np.array(travel_times)
        results[method] = {
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "p95":    float(np.percentile(arr, 95)),
            "timeout_pct": float((arr >= max_time).mean() * 100),
            "n_obs":  float(np.mean(obs_counts)),
            "wall_s": time.time() - t0,
        }

    return results


# -----------------------------------------------------------------------
# Printing helpers
# -----------------------------------------------------------------------

def print_table(results_grid: dict, net: str, methods: list[str], scenarios: list[str]):
    """Print a LaTeX-style comparison table."""
    header = f"\n{'='*90}\n{net.upper()} NETWORK\n{'='*90}"
    print(header)
    col_w = 10

    # Header row
    scen_header = "".join(f"{'  '+s:<22}" for s in scenarios)
    print(f"{'Method':<14} {scen_header}")
    sub = "".join(f"{'Mean':>7} {'P95':>6} {'TO%':>6}  " for _ in scenarios)
    print(f"{'':14} {sub}")
    print("-" * 90)

    for method in methods:
        row = f"{method:<14}"
        for scenario in scenarios:
            r = results_grid.get((net, scenario, method), {})
            if r:
                row += f" {r['mean']:>7.1f} {r['p95']:>6.0f} {r['timeout_pct']:>5.1f}%  "
            else:
                row += f" {'N/A':>7} {'':>6} {'':>6}  "
        print(row)

    # Improvement over Static
    print(f"\nImprovement over Static (mean travel time):")
    for scenario in scenarios:
        static = results_grid.get((net, scenario, "Static"), {}).get("mean", None)
        if static is None:
            continue
        print(f"  [{scenario}]")
        for method in methods:
            if method == "Static":
                continue
            r = results_grid.get((net, scenario, method), {})
            if r:
                imp = static - r["mean"]
                pct = imp / static * 100
                print(f"    {method:<12}: {imp:>+6.1f} min ({pct:>+5.1f}%)")


def print_latex_table(results_grid: dict, net: str, methods: list[str],
                      scenarios: list[str]):
    """Print LaTeX tabular code for paper insertion."""
    print(f"\n% ---- LaTeX table: {net} network ----")
    n_scen = len(scenarios)
    col_spec = "l" + "rr" * n_scen
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    # Multi-column scenario headers
    scen_cols = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{s.replace('_', ' ').title()}}}"
        for s in scenarios)
    print(f"Method & {scen_cols} \\\\")
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(n_scen))
    print(cmidrules)
    sub = " & ".join("Mean & P95" for _ in scenarios)
    print(f" & {sub} \\\\")
    print("\\midrule")
    for method in methods:
        cells = []
        for scenario in scenarios:
            r = results_grid.get((net, scenario, method), {})
            if r:
                cells.append(f"{r['mean']:.1f} & {r['p95']:.0f}")
            else:
                cells.append("-- & --")
        print(f"{method} & {' & '.join(cells)} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


# -----------------------------------------------------------------------
# Sensitivity: network scale (small / large) × disruption intensity
# -----------------------------------------------------------------------

def run_scale_sensitivity(seed: int = 42):
    """Test key methods across grid sizes 3x3, 5x5, 7x7."""
    print("\n" + "="*70)
    print("SENSITIVITY: Network scale (key methods, disrupted scenario)")
    print("="*70)
    print(f"{'Method':<14} {'3x3':>8} {'5x5':>8} {'7x7':>8}  (mean travel time, disrupted)")
    print("-" * 50)

    key_methods = ["Static", "LCB-V2", "SW-LCB", "Oracle"]
    grid_sizes = [(3, 3), (5, 5), (7, 7)]

    rows: dict[str, list] = {m: [] for m in key_methods}

    for rows_n, cols_n in grid_sizes:
        from src.large_network import create_grid_regime_distributions
        set_regime_dist_fn(create_grid_regime_distributions)
        _regime_dist_cache.clear()

        def make_graph():
            return create_grid_network(grid_rows=rows_n, grid_cols=cols_n)

        s_dest = rows_n * cols_n - 1
        schedule = RegimeSchedule(shifts=[(0, "normal"),
                                          (490, "central_disruption"), (560, "normal")])
        rng = np.random.default_rng(seed)

        for method in key_methods:
            times = []
            for i in range(50):
                graph = make_graph()
                router = make_router(method, graph, "large", seed + i,
                                     schedule, create_grid_regime_distributions)
                t_dep = 480 + rng.integers(0, 20)
                jrng = np.random.default_rng(seed + i * 31337)
                result = simulate_bandit_journey(
                    graph=graph, router=router,
                    s_source=0, s_dest=s_dest,
                    t_depart=t_dep, regime_schedule=schedule,
                    rng=jrng, max_time=180,
                )
                times.append(result.arrival_time - result.departure_time)
            rows[method].append(np.mean(times))

    for method in key_methods:
        vals = "  ".join(f"{v:>8.1f}" for v in rows[method])
        print(f"{method:<14} {vals}")


# -----------------------------------------------------------------------
# Sensitivity: disruption intensity
# -----------------------------------------------------------------------

def run_disruption_sensitivity(seed: int = 42):
    """Test key methods as disruption severity increases (0%, 30%, 60%, 90% cancel)."""
    import copy
    from src.synthetic_network import create_regime_distributions as base_dist

    print("\n" + "="*70)
    print("SENSITIVITY: Disruption severity (small network, route 402)")
    print("="*70)
    cancel_levels = [0.0, 0.3, 0.6, 0.85, 0.99]
    key_methods = ["Static", "LCB-V1", "LCB-V2", "SW-LCB", "Oracle"]

    print(f"{'Method':<14} " + " ".join(f"p={p:.0%}" for p in cancel_levels))
    print("-" * 70)

    rows: dict[str, list] = {m: [] for m in key_methods}

    for cancel_p in cancel_levels:
        def custom_dist(regime: str) -> dict:
            d = base_dist(regime)
            if regime == "disrupted_402":
                d["402"]["cancel_prob"] = cancel_p
            return d

        set_regime_dist_fn(custom_dist)
        _regime_dist_cache.clear()
        schedule = RegimeSchedule(shifts=[(0, "normal"), (490, "disrupted_402"), (540, "normal")])
        rng = np.random.default_rng(seed)

        for method in key_methods:
            times = []
            for i in range(100):
                graph = create_bus_story_network()
                router = make_router(method, graph, "small", seed + i,
                                     schedule, custom_dist)
                t_dep = 480 + rng.integers(0, 20)
                jrng = np.random.default_rng(seed + i * 31337)
                result = simulate_bandit_journey(
                    graph=graph, router=router,
                    s_source=0, s_dest=9,
                    t_depart=t_dep, regime_schedule=schedule,
                    rng=jrng, max_time=120,
                )
                times.append(result.arrival_time - result.departure_time)
            rows[method].append(np.mean(times))

    for method in key_methods:
        vals = "  ".join(f"{v:>8.1f}" for v in rows[method])
        print(f"{method:<14} {vals}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

METHODS_FULL = [
    "Static",
    "LCB-V1",
    "LCB-V2",
    "DRO",
    "PS-SSP",
    "BAMCP-60",
    "SW-LCB",
    "EXP3",
    "Oracle",
]

METHODS_FAST = [
    "Static",
    "LCB-V1",
    "LCB-V2",
    "DRO",
    "SW-LCB",
    "EXP3",
    "Oracle",
]

SMALL_SCENARIOS = ["normal", "disrupted", "rush_hour", "multi"]
LARGE_SCENARIOS = ["normal", "disrupted", "full_chaos"]


def main(fast: bool = False, n_journeys: int = 100):
    methods = METHODS_FAST if fast else METHODS_FULL
    seed = 42

    results_grid: dict[tuple, dict] = {}

    print("=" * 90)
    print("BAPR-HRO FULL COMPARISON")
    print(f"Methods: {methods}")
    print(f"Journeys per cell: {n_journeys}")
    print("=" * 90)

    # --- Small network ---
    print("\nRunning small network experiments...")
    for scenario in SMALL_SCENARIOS:
        print(f"  [{scenario}] ", end="", flush=True)
        r = run_experiment("small", scenario, methods, n_journeys, seed)
        for m, v in r.items():
            results_grid[("small", scenario, m)] = v
        print(f"done ({list(r.values())[0]['wall_s']:.0f}s per method avg)")

    # --- Large network ---
    print("\nRunning large network experiments...")
    for scenario in LARGE_SCENARIOS:
        print(f"  [{scenario}] ", end="", flush=True)
        r = run_experiment("large", scenario, methods, n_journeys // 2, seed)
        for m, v in r.items():
            results_grid[("large", scenario, m)] = v
        print(f"done")

    # --- Print tables ---
    print_table(results_grid, "small", methods, SMALL_SCENARIOS)
    print_table(results_grid, "large", methods, LARGE_SCENARIOS)

    # --- LaTeX tables ---
    print_latex_table(results_grid, "small", methods, ["normal", "disrupted"])
    print_latex_table(results_grid, "large", methods, ["normal", "disrupted"])

    # --- Sensitivity analyses ---
    run_disruption_sensitivity(seed)
    run_scale_sensitivity(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip BAMCP-60 and PS-SSP (slow) for quick runs")
    parser.add_argument("--n", type=int, default=100,
                        help="Journeys per (network, scenario, method) cell")
    args = parser.parse_args()
    main(fast=args.fast, n_journeys=args.n)
