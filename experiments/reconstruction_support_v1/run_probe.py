"""Exact design probes and one local query for the HRO reconstruction.

Run from the repository root with Python 3. Reads only a compact OD summary
and the existing 1 MB graph; writes diagnostic JSON next to this script.
The exact examples are constructed models, not passenger-benefit experiments.
"""
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import importlib
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def exact_probes():
    # At p=1/2 a compromise is better than either mode's specialist.
    costs = {"mode_0_specialist": (0, 10),
             "mode_1_specialist": (10, 0), "compromise": (4, 4)}
    mean = {a: sum(c) / 2 for a, c in costs.items()}
    specialists = set(min(costs, key=lambda a: costs[a][z]) for z in (0, 1))
    hidden = min(mean.values())
    specialist_hidden = min(mean[a] for a in specialists)
    revealed = sum(min(c[z] for c in costs.values()) for z in (0, 1)) / 2
    assert (hidden, specialist_hidden, revealed) == (4, 5, 0)

    # A two-step network. Both paths cost 6 under the restricted menu.
    # The best full path costs 2. The chosen restricted-optimal path is via v.
    edges = {"s": {"u": (1, "u"), "v": (2, "v")},
             "u": {"fast": (1, "d"), "slow": (5, "d")},
             "v": {"fast": (1, "d"), "slow": (4, "d")}, "d": {}}
    menus = {"s": ("u", "v"), "u": ("slow",), "v": ("slow",)}
    policy = {"s": "v", "u": "slow", "v": "slow"}

    @lru_cache(None)
    def value(state, restricted):
        if state == "d":
            return 0
        actions = menus[state] if restricted else edges[state]
        return min(edges[state][a][0] + value(edges[state][a][1], restricted)
                   for a in actions)

    state, total, restricted_advantage, full_local_rank = "s", 0, 0, 0
    trace = []
    while state != "d":
        action = policy[state]
        cost, next_state = edges[state][action]
        q_h = cost + value(next_state, True)
        q_full = cost + value(next_state, False)
        min_menu_q_full = min(edges[state][a][0] + value(edges[state][a][1], False)
                              for a in menus[state])
        restricted_advantage += q_h - value(state, True)
        full_local_rank += q_full - min_menu_q_full
        trace.append({"state": state, "action": action, "cost": cost,
                      "restricted_advantage": q_h - value(state, True)})
        total += cost
        state = next_state
    support_gap = value("s", True) - value("s", False)
    assert total - value("s", False) == support_gap + restricted_advantage
    assert (support_gap, restricted_advantage, full_local_rank) == (4, 0, 1)

    # Adding one edge of a two-edge alternative gives no complete path.
    def improvement(extra):
        return 10 if {"s-u", "u-d"} <= set(extra) else 0
    edge_gains = {str(tuple(s)): improvement(s) for k in range(3)
                  for s in combinations(("s-u", "u-d"), k)}
    assert improvement(("s-u", "u-d")) > improvement(("s-u",)) + improvement(("u-d",))
    return {"belief_menu": {"cost_vectors": costs, "uniform_belief_costs": mean,
                            "full_menu_cost": hidden, "mode_top1_union_cost": specialist_hidden,
                            "mode_revealed_before_choice_cost": revealed},
            "journey_decomposition": {"full_optimum": value("s", False),
                                      "restricted_optimum": value("s", True),
                                      "executed_cost": total, "support_gap": support_gap,
                                      "within_menu_gap": restricted_advantage,
                                      "full_Q_local_rank_despite_restricted_optimality": full_local_rank,
                                      "trace": trace},
            "edge_complementarity": edge_gains}


def local_label_probe():
    module = importlib.import_module("src.durner.topocsa")
    with (ROOT / "data/zurich_wide.pkl").open("rb") as handle:
        graph = pickle.load(handle)
    summary = json.loads((ROOT / "experiments/swiss_full/results/swiss_multi_od_v3.json").read_text())
    od = summary["viable_ods"][0]
    by_name = {}
    for sid, stop in graph.stops.items():
        by_name.setdefault(stop.name, sid)
    original = module._insert_departure_label
    dropped, examples = 0, []
    differing = {"route": 0, "departure": 0, "feasibility": 0}

    def record(labels, stop, label, connection):
        nonlocal dropped
        original(labels, stop, label, connection)
        if label.mean_dest_arrival == float("inf"):
            return
        if any(item is label for item in labels[stop]):
            return
        dropped += 1
        other = min(labels[stop], key=lambda x: abs(x.mean_dest_arrival - label.mean_dest_arrival))
        kept = graph.connections[other.connection_id]
        differences = {"route": connection.route != kept.route,
                       "departure": connection.dep_time != kept.dep_time,
                       "feasibility": abs(label.feasibility - other.feasibility) > 1e-9}
        for name, differs in differences.items():
            differing[name] += int(differs)
        if any(differences.values()) and len(examples) < 5:
            examples.append({"stop": stop,
                             "dropped": {"connection": label.connection_id, "route": connection.route,
                                         "departure": connection.dep_time, "mean": label.mean_dest_arrival,
                                         "feasibility": label.feasibility},
                             "kept": {"connection": other.connection_id, "route": kept.route,
                                      "departure": kept.dep_time, "mean": other.mean_dest_arrival,
                                      "feasibility": other.feasibility}})

    module._insert_departure_label = record
    try:
        result = module.topocsa(graph, by_name[od["s1_name"]], by_name[od["s2_name"]], 490)
    finally:
        module._insert_departure_label = original
    return {"origin": od["s1_name"], "destination": od["s2_name"], "departure": 490,
            "connections_processed": result.n_connections_processed,
            "mean_duplicate_drops": dropped, "drops_with_differing_attributes": differing,
            "examples": examples}


if __name__ == "__main__":
    report = {"exact_constructed_examples": exact_probes(), "existing_graph_query": local_label_probe()}
    output = Path(__file__).with_name("probe_results.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
