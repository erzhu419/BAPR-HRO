"""Exact finite-bank mechanism panel; no sampling or hindsight route choice."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.information_planner import Scenario, Outcome, FiniteScenarioModel, ExactPlanner


def timing_model(relationship, early, timeout=120):
    pairs = {"opposite": [(10, 50), (50, 10)],
             "independent": [(10, 10), (10, 50), (50, 10), (50, 50)],
             "together": [(10, 10), (50, 50)]}[relationship]
    worlds = tuple(Scenario(str(i), 1 / len(pairs), {"A": a, "B": b})
                   for i, (a, b) in enumerate(pairs))

    def actions(state):
        return {"origin": ("direct", "hub"), "hub": ("A", "B")}.get(state, ())

    def transition(world, state, action):
        if state == "origin":
            if action == "direct":
                return Outcome("destination", None, min(25, timeout), True)
            signal = (world.factors["A"], world.factors["B"]) if early else None
            return Outcome("hub", signal, min(2, timeout))
        return Outcome("destination", None, min(world.factors[action], max(0, timeout - 2)), True)

    return FiniteScenarioModel(worlds, "origin", actions, transition, 2,
                               dependencies=lambda state, action: {"A", "B"}
                               if state == "hub" or (early and action == "hub") else set())


def run():
    rows = []
    for relation, early in [("opposite", True), ("independent", True),
                            ("together", True), ("opposite", False)]:
        model = timing_model(relation, early)
        for cap in (20, 40, 120):
            model = timing_model(relation, early, cap)
            started = perf_counter()
            tracemalloc.start()
            full = ExactPlanner(model)
            optimum = full.solve().value
            full_ms = (perf_counter() - started) * 1000
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            for retained in (("A",), ("B",), ("A", "B")):
                rule = lambda node, retained=retained: retained if node.state == "hub" else model.actions(node.state)
                restricted = ExactPlanner(model, rule)
                # Expected marginal costs ignore future signals; chooses direct.
                marginal_policy = lambda node: "direct" if node.state == "origin" else retained[0]
                cert = restricted.certificate(marginal_policy)
                rows.append({"relationship": relation, "message_before_choice": early,
                             "timeout": cap, "hub_menu": retained,
                             **asdict(cert), "full_ms": full_ms,
                             "full_stats": dict(full.stats), "peak_bytes": peak,
                             "restricted_nodes": restricted.stats["information_nodes_solved"],
                             "telescoping_loss": restricted.decision_loss(marginal_policy),
                             "clairvoyant_lower_bound": full.clairvoyant_lower_bound()})
                if abs(cert.within_candidate_loss - restricted.decision_loss(marginal_policy)) > 1e-10:
                    raise AssertionError("Bellman advantages failed to telescope")
                if cert.support_loss < -1e-10 or cert.within_candidate_loss < -1e-10:
                    raise AssertionError("A restricted causal policy beat its feasible superset")
    output = Path(__file__).with_name("mechanism_results.json")
    output.write_text(json.dumps({"scope": "exact constructed models; not observed transit benefits",
                                  "rows": rows}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows),
                      "full_120": [(x["relationship"], x["message_before_choice"],
                                    x["full_optimum"], x["support_loss"])
                                   for x in rows if x["timeout"] == 120 and x["hub_menu"] == ("A",)]}))


if __name__ == "__main__":
    run()
