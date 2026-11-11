"""Known-model 2x2 probability/rescue valuation attribution, exact causal suffix.

The fixed toy transit duration bank is constructed, not observed transit data.
Only the origin choice uses the four scores; every suffix uses the same exact
causal optimal continuation. No realized-world minimum is used for rescue.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from math import isclose
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.committed_transit import OBSERVE, build_committed_model
from src.information_planner import ExactPlanner, InformationState
from src.simulator import JourneyScenario, VehicleEvent
from src.transit_graph import Connection, Stop, TransitGraph


class PublicStatePlanner(ExactPlanner):
    """Exact memoization for this experiment's state-dependent H and policies.

    CommittedState already retains every message/feedback; the posterior keeps
    any information carried by action timing. Future transitions and H here
    depend on this state and posterior, not a redundant list of WAIT actions.
    This specialization is not valid for arbitrary history-dependent H.
    """
    @staticmethod
    def canonical(node):
        return replace(node, history=()) if node.history else node

    def _solve(self, node, compressed=False):
        return super()._solve(self.canonical(node), compressed)

    def _action_value(self, node, action, compressed=False, outcomes=None):
        return super()._action_value(self.canonical(node), action, compressed, outcomes)

    def evaluate(self, policy, node=None):
        return super().evaluate(policy, self.canonical(self.root if node is None else node))

    def decision_loss(self, policy, node=None):
        return super().decision_loss(policy, self.canonical(self.root if node is None else node))


def make_model(relationship, cancel_probability, cap, backup_arrival):
    pairs = {"same": ((0, 0), (6, 6)),
             "independent": ((0, 0), (0, 6), (6, 0), (6, 6)),
             "opposite": ((0, 6), (6, 0))}[relationship]
    graph = TransitGraph()
    for sid in range(3):
        graph.add_stop(Stop(sid, str(sid), min_transfer_time=2))
    for connection in [Connection(0, "incoming", "incoming", 0, 1, 0, 6),
                       Connection(1, "target", "target", 1, 2, 9, 17),
                       Connection(2, "backup", "backup", 1, 2, 16, backup_arrival),
                       Connection(3, "direct", "direct", 0, 2, 0, 28)]:
        graph.add_connection(connection)
    bank, specifications = [], []
    cancel_states = ((False, 1 - cancel_probability), (True, cancel_probability))
    for incoming_duration_shift, target_shift in pairs:
        for canceled, cancel_weight in cancel_states:
            if not cancel_weight:
                continue
            weight = cancel_weight / len(pairs)
            # Incoming departure delay is zero; only its travel duration varies.
            # This custom bank is not JourneyScenario.sample's common-shift model.
            scene = JourneyScenario({
                0: VehicleEvent(0, False, 0, 6 + incoming_duration_shift),
                1: VehicleEvent(999 if canceled else target_shift, canceled,
                                9 if canceled else 9 + target_shift,
                                17 if canceled else 17 + target_shift),
                2: VehicleEvent(0, False, 16, backup_arrival),
                3: VehicleEvent(0, False, 0, 28),
            })
            bank.append((weight, scene))
            specifications.append({"probability": weight,
                                   "incoming_duration_shift": incoming_duration_shift,
                                   "target_shift": target_shift, "target_canceled": canceled,
                                   "events": {cid: asdict(event) for cid, event in scene.events.items()}})
    return build_committed_model(graph, bank, 0, 2, 0, cap), specifications


def root_rule(model, incoming_only=False):
    def rule(node):
        actions = model.actions(node.state)
        if (node.state.initialized and node.state.stop == 0
                and node.state.current_trip is None):
            return tuple(action for action in actions if action[0] == "select"
                         and action[1] in ((0,) if incoming_only else (0, 3)))
        return actions
    return rule


def execute_in_world(planner, world_index):
    """Execute one shared causal policy, conditioning only on actual observations."""
    model, node = planner.model, planner.root
    total = 0.0
    for _ in range(model.horizon):
        action = planner.optimal_action(node)
        if action is None:
            total += model.terminal_cost(model.scenarios[world_index], node.state)
            break
        outcome = model.transition(model.scenarios[world_index], node.state, action)
        total += outcome.cost
        members = []
        for i, probability in zip(node.support, node.belief):
            # An unobserved world's outcome is used only to update posterior
            # compatibility, never to select the executed action.
            if model.transition(model.scenarios[i], node.state, action) == outcome:
                members.append((i, probability))
        mass = sum(p for _, p in members)
        node = InformationState(node.stage + 1, outcome.state,
                                tuple(i for i, _ in members), tuple(p / mass for _, p in members),
                                node.history + ((action, outcome.state, outcome.observation,
                                                 outcome.cost, outcome.terminal),))
        if outcome.terminal:
            total += model.terminal_cost(model.scenarios[world_index], node.state)
            break
    boarded_target = any(event[0] == "board" and event[1] == 1 for event in node.state.feedback)
    return {"world_index": world_index, "cost": total, "target_caught": boarded_target,
            "feedback": node.state.feedback,
            "public_messages": [asdict(message) for message in node.state.messages]}


def independent_catch_probability(specifications):
    incoming, target = {}, {}
    for world in specifications:
        probability = world["probability"]
        arrival = world["events"][0]["arrival"]
        incoming[arrival] = incoming.get(arrival, 0) + probability
        event = world["events"][1]
        key = event["departure"], event["canceled"]
        target[key] = target.get(key, 0) + probability
    return sum(pi * pt for arrival, pi in incoming.items() for (departure, canceled), pt in target.items()
               if not canceled and arrival + 2 <= departure <= arrival + 2 + 12)


def run_case(relationship, cancel_probability, cap, backup_arrival, model_builder=make_model):
    started = perf_counter()
    model, specifications = model_builder(relationship, cancel_probability, cap, backup_arrival)
    full = PublicStatePlanner(model, root_rule(model))
    optimum = full.solve().value
    incoming = PublicStatePlanner(model, root_rule(model, incoming_only=True))
    incoming_value = incoming.solve().value
    executions = [execute_in_world(incoming, i) for i in range(len(specifications))]
    p_true = sum(world["probability"] for world, run in zip(specifications, executions) if run["target_caught"])
    success = sum(world["probability"] * run["cost"] for world, run in zip(specifications, executions) if run["target_caught"])
    failure = sum(world["probability"] * run["cost"] for world, run in zip(specifications, executions) if not run["target_caught"])
    cs_true = success / p_true
    r_true = failure / (1 - p_true) if p_true < 1 - 1e-12 else None
    p_independent = independent_catch_probability(specifications)
    structural_p = sum(world["probability"] for world in specifications
                       if not world["events"][1]["canceled"]
                       and world["events"][0]["arrival"] + 2
                       <= world["events"][1]["departure"]
                       <= world["events"][0]["arrival"] + 2 + 12)
    if not isclose(p_true, structural_p, abs_tol=1e-9):
        raise AssertionError("Optimal causal suffix did not preserve the stated target catch event")
    if not isclose(success + failure, incoming_value, abs_tol=1e-9):
        raise AssertionError("Conditional cost mixture differs from exact incoming Q")
    # Root action cost with one shared optimal causal continuation after it.
    actual = {}
    for root_action in (0, 3):
        def policy(node):
            if node.state.initialized and node.state.stop == 0 and node.state.current_trip is None:
                return "select", root_action
            return full.optimal_action(node)
        value = full.evaluate(policy)
        decision_loss = full.decision_loss(policy)
        if not isclose(value - optimum, decision_loss, abs_tol=1e-9):
            raise AssertionError("Bellman decision loss does not equal actual excess cost")
        actual[root_action] = {"J": value, "within_candidate_loss": decision_loss}
    variants = {}
    for probability_name, probability in (("independent", p_independent), ("true", p_true)):
        for rescue_name in ("cap", "causal"):
            if rescue_name == "causal" and r_true is None and probability < 1 - 1e-12:
                variants[f"{probability_name}/{rescue_name}"] = {
                    "identified": False,
                    "reason": "true failure branch has probability zero; its conditional rescue cost is not identified"}
                continue
            rescue = cap if rescue_name == "cap" else r_true
            estimate = cs_true if probability >= 1 - 1e-12 else probability * cs_true + (1 - probability) * rescue
            selected = 0 if estimate <= 28 else 3
            if r_true is None:
                components = None
            else:
                components = {"catch_probability": (probability - p_true) * (cs_true - r_true),
                              "success_conditional_cost": 0.0,
                              "failure_conditional_rescue": (1 - probability) * (rescue - r_true)}
                if not isclose(sum(components.values()), estimate - incoming_value, abs_tol=1e-9):
                    raise AssertionError("Signed probability/success/rescue error identity failed")
            variants[f"{probability_name}/{rescue_name}"] = {
                "identified": True, "p_hat": probability, "success_cost_fixed_true": cs_true,
                "rescue_hat": rescue, "estimated_incoming_Q": estimate,
                "incoming_Q_signed_error": estimate - incoming_value,
                "signed_error_components": components,
                "selected_root_connection": selected,
                "true_J": actual[selected]["J"],
                "within_candidate_loss": actual[selected]["within_candidate_loss"]}
    return {"relationship": relationship, "target_cancel_probability": cancel_probability,
            "deadline_cap": cap, "backup_arrival": backup_arrival,
            "worlds": specifications, "incoming_policy_executions": executions,
            "full_same_information_root_restricted_optimum": optimum,
            "true_incoming_Q": incoming_value, "p_true": p_true,
            "p_independent_marginal_recombination": p_independent,
            "Cs_true": cs_true, "R_true_causal_suffix": r_true,
            "R_note": "conditional total journey cost on failure of target catch under one causal optimal suffix; failure label is used only for reporting",
            "variants": variants, "full_stats": dict(full.stats),
            "incoming_stats": dict(incoming.stats),
            "elapsed_ms": (perf_counter() - started) * 1000}


def summarize(cases):
    harmful_single_repairs, opposing_error_examples = [], []
    for i, case in enumerate(cases):
        variants = case["variants"]
        baseline = variants["independent/cap"]
        for repair in ("true/cap", "independent/causal"):
            variant = variants[repair]
            if variant["identified"] and variant["true_J"] > baseline["true_J"] + 1e-9:
                harmful_single_repairs.append({"case_index": i, "repair": repair,
                                               "before_J": baseline["true_J"], "after_J": variant["true_J"],
                                               "before_incoming_error": baseline["incoming_Q_signed_error"],
                                               "after_incoming_error": variant["incoming_Q_signed_error"]})
        components = baseline.get("signed_error_components")
        # Finite-bank .2 weights can leave an exact-zero component at 1e-15.
        # Such arithmetic noise is not an opposing-error mechanism example.
        if (components and abs(components["catch_probability"]) > 1e-9
                and abs(components["failure_conditional_rescue"]) > 1e-9
                and components["catch_probability"] * components["failure_conditional_rescue"] < 0):
            opposing_error_examples.append({"case_index": i, "components": components,
                                            "combined_error": baseline["incoming_Q_signed_error"]})
    return {"cases": len(cases),
            "fully_identified_factorials": sum(all(v["identified"] for v in c["variants"].values()) for c in cases),
            "harmful_single_repairs": harmful_single_repairs,
            "opposing_signed_errors": opposing_error_examples,
            "interpretation": "partial correction can remove cancellation between two valuation errors; no claim that every single-component correction is harmful",
            "research_gate": "known-model mechanism only; no new estimator validation, city benefit or new theoretical contribution established"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("exact_attribution_results.json"))
    args = parser.parse_args()
    # Detect any loss from the specialized state key before using it in the
    # factorial. A mismatch requires retaining the full-history reference.
    check_model, _ = make_model("opposite", 0, 40, 36)
    generic = ExactPlanner(check_model, root_rule(check_model))
    specialized = PublicStatePlanner(check_model, root_rule(check_model))
    generic_value, specialized_value = generic.solve().value, specialized.solve().value
    forced_incoming = lambda node: (("select", 0) if node.state.initialized
                                    and node.state.stop == 0 and node.state.current_trip is None
                                    else generic.optimal_action(node))
    forced_incoming_specialized = lambda node: (("select", 0) if node.state.initialized
                                                and node.state.stop == 0 and node.state.current_trip is None
                                                else specialized.optimal_action(node))
    generic_incoming = generic.evaluate(forced_incoming)
    specialized_incoming = specialized.evaluate(forced_incoming_specialized)
    if not isclose(generic_value, specialized_value, abs_tol=1e-9) or not isclose(generic_incoming, specialized_incoming, abs_tol=1e-9):
        raise AssertionError("Public-state memoization differs from full-history reference")
    memoization_check = {"case": "opposite/no_cancel/cap40/backup36", "generic_optimum": generic_value,
                         "public_state_optimum": specialized_value,
                         "generic_incoming_Q": generic_incoming,
                         "public_state_incoming_Q": specialized_incoming}
    cases = []
    for relationship in ("same", "independent", "opposite"):
        for cap in (40, 60, 90):
            for backup_arrival in (36, 48):
                for cancel_probability in (0, 0.2):
                    case = run_case(relationship, cancel_probability, cap, backup_arrival)
                    cases.append(case)
                    print(json.dumps({"completed": len(cases), "relationship": relationship,
                                      "cap": cap, "backup_arrival": backup_arrival,
                                      "cancel_probability": cancel_probability,
                                      "p_true": case["p_true"], "p_independent": case["p_independent_marginal_recombination"],
                                      "optimum": case["full_same_information_root_restricted_optimum"]}), flush=True)
    result = {"scope": "exact known-model constructed transit duration/cancellation bank; no observed events",
              "root_choice": "after forced local OBSERVE, incoming connection 0 or direct connection 3; root WAIT excluded from every policy and reference",
              "suffix": "same-information causal optimum; rescue is not a realized-world minimum",
              "information": "committed_transit local last10/max5 publication and twelve-minute commitment",
              "incoming_duration_model": "departure remains zero; arrival 6+D, D in {0,6}; VehicleEvent.delay=0 denotes incoming departure delay; this is not the common-shift scenario sampler",
              "estimation_design": "2x2 independent/true catch probability by cap/causal rescue; successful conditional journey cost fixed to its true value",
              "conditional_endpoint": "same/no-cancel has p=1; R_true is null and independent/causal score is not identified rather than inventing an impossible-branch value",
              "tuning": "none; 3 relations x 3 caps x 2 backup arrivals x 2 cancel probabilities fixed in source",
              "memoization": "same complete public state, posterior and stage; H and evaluated policies depend on public state only; no new approximation",
              "memoization_reference_check": memoization_check,
              "summary": summarize(cases), "cases": cases}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases),
                      "harmful_single_repairs": len(result["summary"]["harmful_single_repairs"])}), flush=True)


if __name__ == "__main__":
    main()
