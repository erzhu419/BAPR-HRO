"""Experiment 5: CSA-MEAT baseline comparison.

Implements a simplified CSA-MEAT (Connection Scan Algorithm for Minimum
Expected Arrival Time) as a baseline from Dibbelt et al. (2013).

CSA-MEAT computes a deterministic optimal route using expected arrival times
(no stochastic hyperpath, no alternatives). When a connection is delayed/
canceled, the passenger is stuck until the next connection on the same route.

This provides a lower-bound comparison: CSA-MEAT is what happens when you
don't use hyperpaths at all (single-path routing).

Output: results/csa_meat_baseline.json
"""

import sys, os, json, time, pickle, copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.transit_graph import TransitGraph, StopLabel, Connection
from src.durner.topocsa import topocsa
from src.pmf import PMF
from src.gtfs_parser import load_routes
from src.gtfs_rt_parser import load_distributions
from src.bandit_router import BanditRouter
from src.dro_router import DRORouter
from src.router import StaticRouter, RoutingDecision
from src.simulate_bandit import simulate_bandit_journey
from src.simulator import (RegimeSchedule, set_regime_dist_fn,
                           _regime_dist_cache, sample_actual_delay,
                           JourneyEvent, JourneyResult)


class CSAMEATRouter:
    """Simplified CSA-MEAT: deterministic shortest path using expected times.

    Unlike Durner's stochastic hyperpath (which maintains alternatives),
    CSA-MEAT computes a single fastest path based on expected arrival times.
    No fallback alternatives, no uncertainty handling.
    """

    def __init__(self, graph: TransitGraph):
        self.graph = graph
        self.best_path = None  # list of connection IDs
        self.cached_result = None

    def route(self, s_source, s_dest, t_source):
        """Compute single-path routing via expected times."""
        # Use topocsa but only take the BEST connection at each stop
        self.cached_result = topocsa(self.graph, s_source, s_dest, t_source)

        # Extract the single best path (greedy: always take top-ranked)
        if self.cached_result.mean_arrival < float('inf'):
            self.best_path = self._extract_best_path(s_source, s_dest, t_source)
        return RoutingDecision(hyperpath=self.cached_result, regime_id=0, regime_name="normal", confidence=1.0, surprise=0.0, recomputed=False, computation_ms=0.0)

    def _extract_best_path(self, s_source, s_dest, t_source):
        """Extract the single best path from the hyperpath."""
        path = []
        current_stop = s_source
        current_time = t_source

        for _ in range(20):  # max hops
            if current_stop == s_dest:
                break
            labels = self.cached_result.stop_labels.get(current_stop, [])
            if not labels:
                break

            # Take THE BEST label only (no alternatives)
            best = None
            for lab in reversed(labels):
                c = self.graph.connections[lab.connection_id]
                if c.dep_time >= current_time:
                    best = lab
                    break

            if best is None:
                break

            c = self.graph.connections[best.connection_id]
            path.append(best.connection_id)
            current_stop = c.arr_stop
            current_time = c.arr_time + 1

        return path

    def select_connection(self, stop_id, current_time, rng, top_k=5):
        """CSA-MEAT: only offer THE SINGLE BEST connection, no alternatives."""
        if self.cached_result is None:
            return None

        labels = self.cached_result.stop_labels.get(stop_id, [])
        if not labels:
            return None

        # Only return the best label (no top-K, no alternatives)
        for lab in reversed(labels):
            c = self.graph.connections[lab.connection_id]
            if c.dep_time >= current_time - 1:
                return lab, lab.mean_dest_arrival

        return None

    def observe_delay(self, route, delay):
        pass  # CSA-MEAT doesn't learn

    def observe_cancel(self, route):
        pass

    @property
    def total_observations(self):
        return 0


def run_baseline_comparison(g, s1, s2, normal_by_name, disrupted_by_name,
                            N=20, seed=42):
    def real_day_regime(name):
        src = normal_by_name if name == 'normal' else disrupted_by_name
        result = {}
        for rname, d in src.items():
            delays = np.arange(-5, 65)
            mean, std = d['mean'], max(d['std'], 0.5)
            probs = np.exp(-0.5 * ((delays - mean) / std) ** 2)
            probs /= probs.sum()
            cancel = d.get('cancel_rate', 0)
            if cancel > 0:
                probs *= (1 - cancel)
            info = {'delay_probs': probs, 'delay_offset': -5}
            if cancel > 0.01:
                info['cancel_prob'] = cancel
            result[rname] = info
        return result

    set_regime_dist_fn(real_day_regime)

    # Need to patch simulate_bandit to handle CSAMEATRouter
    from src.simulate_bandit import simulate_bandit_journey

    methods = {
        'CSA-MEAT': CSAMEATRouter,
        'Static': StaticRouter,
        'LCB': BanditRouter,
        'DRO': DRORouter,
    }

    scenarios = {
        'normal': RegimeSchedule(shifts=[(0, 'normal')]),
        'disrupted': RegimeSchedule(shifts=[(0, 'disrupted')]),
    }

    results = {}
    for scen_name, sched in scenarios.items():
        results[scen_name] = {}
        for mname, Cls in methods.items():
            _regime_dist_cache.clear()
            times = []
            for i in range(N):
                ri = Cls(copy.deepcopy(g))
                rng = np.random.default_rng(seed + i)
                res = simulate_bandit_journey(
                    ri.graph, ri, s1, s2, 490, sched, rng, 120)
                times.append(res.arrival_time - res.departure_time)
            arr = np.array(times)
            results[scen_name][mname] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
            }
            print(f"  {scen_name} {mname:10s}: mean={arr.mean():.1f} p95={np.percentile(arr,95):.1f}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("CSA-MEAT Baseline Comparison")
    print("=" * 60)

    with open('data/zurich_wide.pkl', 'rb') as f:
        g = pickle.load(f)
    with open('data/day_distributions.pkl', 'rb') as f:
        day_dists = pickle.load(f)
    route_names_map = load_routes('data/swiss_gtfs')

    normal_by_name = {}
    disrupted_by_name = {}
    for rid, d in day_dists['normal_day'].items():
        name = route_names_map.get(rid)
        if name:
            normal_by_name[name] = d
    for rid, d in day_dists['disrupted_day'].items():
        name = route_names_map.get(rid)
        if name and (name not in disrupted_by_name or
                     d['mean'] > disrupted_by_name[name]['mean']):
            disrupted_by_name[name] = d

    for c in g.connections:
        d = normal_by_name.get(c.route)
        mean = d['mean'] if d else 0.5
        std = max(d['std'] if d else 1.5, 0.5)
        delays = np.arange(-5, 20)
        probs = np.exp(-0.5 * ((delays - mean) / std) ** 2)
        probs /= probs.sum()
        c.dep_distribution = PMF.from_delays(c.dep_time, probs, -5)
        c.arr_distribution = PMF.from_delays(c.arr_time, probs, -5)

    # Patch simulate_bandit to recognize CSAMEATRouter
    import src.simulate_bandit as sb
    original_check = "CSAMEATRouter"
    if "CSAMEATRouter" not in str(sb.simulate_bandit_journey.__code__.co_consts):
        # Monkey-patch: CSAMEATRouter has select_connection like bandit
        pass  # It works because CSAMEATRouter has select_connection/observe_delay

    s1, s2 = 67001060, 201257157
    print(f"OD: {g.stops[s1].name} -> {g.stops[s2].name}")

    results = run_baseline_comparison(
        g, s1, s2, normal_by_name, disrupted_by_name, N=20)

    os.makedirs('experiments/swiss_full/results', exist_ok=True)
    with open('experiments/swiss_full/results/csa_meat_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiments/swiss_full/results/csa_meat_baseline.json")
