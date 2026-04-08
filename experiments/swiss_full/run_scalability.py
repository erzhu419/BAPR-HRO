"""Experiment 2: Scalability benchmark.

Measures routing computation time as network size grows:
100 / 300 / 500 / 1000 / 2000 / 3000 connections.

Also benchmarks neural surrogate (V-hat) speedup at each scale.

Output: results/scalability.json
"""

import sys, os, json, time, pickle, copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.gtfs_parser import build_transit_graph
from src.durner.topocsa import topocsa
from src.pmf import PMF


def run_scalability(gtfs_dir='data/zurich_gtfs', n_trials=5):
    sizes = [100, 300, 500, 1000, 2000, 3000]
    results = []

    for max_conn in sizes:
        print(f"\n--- {max_conn} connections ---")

        # Build graph
        t0 = time.time()
        g = build_transit_graph(
            gtfs_dir,
            time_start=480, time_end=540,
            region_lat=(47.35, 47.42), region_lon=(8.49, 8.58),
            max_connections=max_conn,
            verbose=False,
        )
        build_time = time.time() - t0

        n_stops = len(g.stops)
        n_conns = len(g.connections)
        n_routes = len(set(c.route for c in g.connections))
        n_transfers = len(g.get_transfer_stops())

        print(f"  Built: {n_stops} stops, {n_conns} conns, "
              f"{n_routes} routes, {n_transfers} transfers ({build_time:.1f}s)")

        # Apply default delays
        g.assign_distributions({})

        # Find a connected pair
        transfers = g.get_transfer_stops()
        s1, s2 = None, None
        for a in transfers[:10]:
            for b in transfers[-10:]:
                if a == b:
                    continue
                r = topocsa(g, a, b, 490)
                if r.mean_arrival < float('inf') and r.mean_arrival < 540:
                    s1, s2 = a, b
                    break
            if s1:
                break

        if not s1:
            print("  No connected pair found, skipping")
            results.append({
                "max_connections": max_conn,
                "actual_connections": n_conns,
                "stops": n_stops,
                "routes": n_routes,
                "transfers": n_transfers,
                "build_time_s": build_time,
                "routing_time_ms": None,
                "connected": False,
            })
            continue

        # Benchmark routing time
        routing_times = []
        for _ in range(n_trials):
            t0 = time.time()
            r = topocsa(g, s1, s2, 490)
            routing_times.append((time.time() - t0) * 1000)

        mean_rt = np.mean(routing_times)
        std_rt = np.std(routing_times)
        print(f"  Routing: {mean_rt:.1f} ± {std_rt:.1f} ms "
              f"({g.stops[s1].name} -> {g.stops[s2].name})")

        results.append({
            "max_connections": max_conn,
            "actual_connections": n_conns,
            "stops": n_stops,
            "routes": n_routes,
            "transfers": n_transfers,
            "build_time_s": round(build_time, 2),
            "routing_time_ms": round(mean_rt, 1),
            "routing_std_ms": round(std_rt, 1),
            "hyperpath_size": len(r.hyperpath_connections),
            "connected": True,
            "od": f"{g.stops[s1].name} -> {g.stops[s2].name}",
        })

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Scalability Benchmark")
    print("=" * 60)

    results = run_scalability()

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Conns':>6s} {'Stops':>6s} {'Routes':>6s} {'Routing(ms)':>12s} {'Build(s)':>9s}")
    for r in results:
        rt = f"{r['routing_time_ms']:.1f}" if r['routing_time_ms'] else "N/A"
        print(f"{r['actual_connections']:>6d} {r['stops']:>6d} "
              f"{r['routes']:>6d} {rt:>12s} {r['build_time_s']:>9.1f}")

    # Save
    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    with open('experiments/swiss_full/results/scalability.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiments/swiss_full/results/scalability.json")
