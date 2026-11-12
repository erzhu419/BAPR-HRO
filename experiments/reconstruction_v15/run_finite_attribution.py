"""Exact G2 attribution for explicitly declared finite city laws."""
import argparse
from dataclasses import replace
from functools import lru_cache
import json
from math import comb, isclose
import os
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v13 import factored_run as v13
from src.causal_prefix import PrefixBudgetExceeded, causal_prefix_lower
from src.event_platform_solver import EventPlatform
from src.factored_lower import product_trip_bank
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map


BLOCKS = (0, 2, 4, 6)
MAX_NODES = 100_000
WITNESS_NODES = 5_000
DRAW_COUNT = 4
SOURCE_REVISION = "28d97a37988ca014619cc76eefe28d18185caa2e"
ARCHIVED_RESULTS = HERE.parent / "reconstruction_v13" / "factored_results.json"


def rest_world(graph, clocks):
    departures, arrivals, canceled = clocks
    return JourneyScenario({
        c.id: VehicleEvent(int(departures[c.id]) - c.dep_time, bool(canceled[c.id]),
                           int(departures[c.id]), int(arrivals[c.id]))
        for c in graph.connections
    }, minute=60, publications={})


def cross_bank(graph, rests, factors):
    """Each rest has the same independent A/B support."""
    worlds, clock_rows, labels = [], [], []
    for rest_index, clocks in rests:
        common = rest_world(graph, clocks)
        factor_bank, factor_labels = product_trip_bank(graph, common, factors)
        for row, (_, scene) in enumerate(factor_bank):
            first, second = (factor_labels[trip][row] for trip in factors)
            clocks_i = tuple(array.copy() for array in clocks)
            for trip in factors:
                for cid, event in factors[trip][factor_labels[trip][row]].items():
                    clocks_i[0][cid], clocks_i[1][cid], clocks_i[2][cid] = (
                        event.departure, event.arrival, event.canceled)
            clock_rows.append(clocks_i)
            labels.append((rest_index, first, second))
            worlds.append(scene)
    n = len(worlds)
    return tuple((1. / n, scene) for scene in worlds), tuple(map(np.stack, zip(*clock_rows))), labels


def build_problem(panel, rests, factors):
    graph, root, goal, deadline, scan, transfers, _, _, _, _ = panel
    bank, clocks, labels = cross_bank(graph, rests, factors)
    platform = EventPlatform(graph, bank, root, goal, deadline)

    @lru_cache(maxsize=100_000)
    def remaining(i, stop, time, trip, last_arrival, last_connection, used):
        state = replace(root, stop=stop, time=time, current_trip=trip, last_arrival=last_arrival,
                        last_connection=last_connection, used_connections=used)
        return float(scan.platform_arrivals(*(field[i:i + 1] for field in clocks),
                                            state, goal, deadline, transfers)[0] - state.time)

    def oracle(i, state):
        return remaining(i, state.stop, state.time, state.current_trip, state.last_arrival,
                         state.last_connection, state.used_connections)

    return platform, oracle, labels


def factor_weights(labels, rest, first, second):
    """Conditional on rest, with first/second denoting mass of atom 0."""
    return tuple((first if i == 0 else 1. - first) * (second if j == 0 else 1. - second)
                 if r == rest else 0. for r, i, j in labels)


def pathwise_witness(platform, oracle, indices):
    """Find a legal common policy attaining every world's clairvoyant loss."""
    root = platform.model.initial_state, indices
    complete, pending, stack = {}, {}, [root]
    nodes = transitions = 0

    def choices(state, support):
        nonlocal transitions
        actions = platform.model.actions(state)
        values = {i: oracle(i, state) for i in support}
        for action in sorted(actions, key=lambda a: a[0] == 'wait'):
            groups = {}
            for i, child in platform.outcomes(state, support, action, platform.deadline):
                transitions += 1
                if child.cost + oracle(i, child.state) != values[i]:
                    break
                groups.setdefault(child.state, []).append(i)
            else:
                yield [(child, tuple(members)) for child, members in groups.items()]

    while stack:
        key = stack[-1]
        if key in complete:
            stack.pop()
            continue
        state, support = key
        if key not in pending:
            if nodes >= WITNESS_NODES:
                return False, nodes, transitions
            nodes += 1
            if not platform.model.actions(state):
                complete[key] = all(platform.model.terminal_cost(platform.model.scenarios[i], state)
                                    == oracle(i, state) for i in support)
                continue
            pending[key] = choices(state, support), None
        options, selected = pending[key]
        if selected is None or any(complete.get(child) is False for child in selected):
            selected = next(options, None)
            if selected is None:
                complete[key] = False
                del pending[key]
                continue
            pending[key] = options, selected
            continue
        missing = next((child for child in selected if child not in complete), None)
        if missing is None:
            complete[key] = True
            del pending[key]
        else:
            stack.append(missing)
    return complete[root], nodes, transitions


def solve(platform, oracle, *, prefix, weights=None):
    before = perf_counter()
    receipt = {}
    if not prefix:
        if weights is None:
            weights = tuple(scenario.probability for scenario in platform.model.scenarios)
        support = tuple(i for i, weight in enumerate(weights) if weight)
        exact, nodes, transitions = pathwise_witness(platform, oracle, support)
        receipt.update(witness_nodes=nodes, witness_transitions=transitions)
        if exact:
            value = sum(weights[i] * oracle(i, platform.initial) for i in support)
            return dict(status="exact", value=value, method="pathwise_witness",
                        wall_seconds=perf_counter() - before, **receipt)
    try:
        result = platform.solve(oracle, elapsed_horizon=v13.HORIZON if prefix else None,
                                max_nodes=MAX_NODES, stop_after_first_ride=prefix, weights=weights)
    except PrefixBudgetExceeded as error:
        return dict(status="budget_exceeded", nodes=error.nodes - 1,
                    transition_evaluations=error.transitions, wall_seconds=perf_counter() - before, **receipt)
    return dict(status="exact", value=result.value, nodes=result.information_nodes,
                transition_evaluations=result.transition_evaluations,
                oracle_evaluations=result.oracle_evaluations, wall_seconds=perf_counter() - before, **receipt)


def exact_resampling(platform, oracle, labels, rest, conditional, clairvoyant):
    """Enumerate binomial multiplicities for both four-draw empirical factors."""
    draws = []
    probabilities = [comb(DRAW_COUNT, n) / 2**DRAW_COUNT for n in range(DRAW_COUNT + 1)]
    for a in range(DRAW_COUNT + 1):
        for b in range(DRAW_COUNT + 1):
            weights = factor_weights(labels, rest, a / DRAW_COUNT, b / DRAW_COUNT)
            if conditional.get("value") == clairvoyant:
                # At equal positive weights an exact policy reaches each atom's
                # clairvoyant lower loss. It remains optimal under any reweighting.
                entry = dict(status="exact", value=sum(w * oracle(i, platform.model.initial_state)
                                                       for i, w in enumerate(weights) if w),
                             method="pathwise_sandwich")
            elif (a, b) == (2, 2):
                entry = conditional.copy()
                entry["method"] = "already_solved"
            else:
                entry = solve(platform, oracle, prefix=True, weights=weights)
            entry.update(a_atom0_count=a, b_atom0_count=b, mass=probabilities[a] * probabilities[b])
            draws.append(entry)
    assert sum(d["mass"] for d in draws) == 1.
    value = sum(d["mass"] * d["value"] for d in draws) if all(d["status"] == "exact" for d in draws) else None
    return dict(value=value, resampling=draws)


def analyze_block(panel, index):
    graph, root, goal, deadline, scan, transfers, plan, compiled, rest, factors = panel
    rests = [(i, tuple(field[i] for field in rest)) for i in (index, index + 1)]
    selected = {trip: draws[index * v13.ATOMS:index * v13.ATOMS + 2] for trip, draws in factors.items()}
    before = perf_counter()
    platform, oracle, labels = build_problem(panel, rests, selected)
    setup_seconds = perf_counter() - before
    baseline_rows = []
    full = []
    ready = max(root.time, root.last_arrival + graph.stops[root.stop].min_transfer_time)
    for i, scenario in enumerate(platform.model.scenarios):
        observed = v13.replay_arm(graph, plan, root, goal, deadline, platform.model, scenario,
                                  "first_visible", compiled["first_visible"])
        assert observed["published"] and not observed["rejected"]
        assert observed["loss_upper"] == observed["loss_lower"]
        assert observed["publish_at"] <= ready
        loss = observed["loss_upper"]
        lower = oracle(i, root)
        assert lower <= loss
        baseline_rows.append(dict(atom_indices=labels[i], loss=loss, clairvoyant=lower,
                                  published_at=observed["publish_at"]))
        full.append(lower)
    J = mean(row["loss"] for row in baseline_rows)
    conditional = []
    for r in (index, index + 1):
        weights = factor_weights(labels, r, .5, .5)
        upper = sum(w * row["loss"] for w, row in zip(weights, baseline_rows, strict=True))
        clairvoyant = sum(w * lower for w, lower in zip(weights, full, strict=True))
        if upper == clairvoyant:
            VRtau = dict(status="exact", value=clairvoyant, method="first_visible_clairvoyant_sandwich")
        else:
            VRtau = solve(platform, oracle, prefix=True, weights=weights)
        if VRtau.get("value") == upper:
            VR = dict(status="exact", value=upper, method="first_visible_prefix_sandwich")
        else:
            VR = solve(platform, oracle, prefix=False, weights=weights)
        ell = exact_resampling(platform, oracle, labels, r, VRtau, clairvoyant)
        if VR.get("value") is not None and VRtau.get("value") is not None:
            assert VR["value"] >= VRtau["value"] >= clairvoyant
        if ell["value"] is not None and VRtau.get("value") is not None:
            assert VRtau["value"] >= ell["value"] >= clairvoyant
        conditional.append(dict(rest=r, J=upper, full_causal=VR, original_prefix=VRtau,
                                clairvoyant=clairvoyant, expected_small_bank=ell))

    def average(key):
        values = [c[key].get("value") for c in conditional]
        return None if None in values else mean(values)

    V_R, V_Rtau, ell = (average(k) for k in ("full_causal", "original_prefix", "expected_small_bank"))
    if J in (V_R, V_Rtau):
        V = dict(status="exact", value=J, method="first_visible_revealed_sandwich")
    else:
        V = solve(platform, oracle, prefix=False)
    primitives = dict(J=J, V_star=V.get("value"), V_R=V_R, V_Rtau=V_Rtau, ell=ell)
    known = [value for value in primitives.values() if value is not None]
    assert all(a >= b for a, b in zip(known, known[1:]))
    terms = None
    if all(v is not None for v in primitives.values()):
        terms = dict(A=J-V["value"], B=V["value"]-V_R, C=V_R-V_Rtau, D=V_Rtau-ell)
        assert isclose(sum(terms.values()), J-ell, abs_tol=1e-9)
    return dict(block=index, rest_indices=(index, index + 1),
                factor_draw_indices=(index * v13.ATOMS, index * v13.ATOMS + 1),
                setup_seconds=setup_seconds, J_worlds=baseline_rows, full_causal=V,
                conditional=conditional, primitives=primitives, terms=terms,
                status="exact" if terms is not None else "incomplete")


def audit_archived_prefix(panel, archived):
    graph, root, _, _, _, _, _, _, rest, factors = panel
    index = archived["bank"]
    rests = [(index, tuple(field[index] for field in rest))]
    selected = {trip: draws[index * v13.ATOMS:(index + 1) * v13.ATOMS]
                for trip, draws in factors.items()}
    platform, oracle, _ = build_problem(panel, rests, selected)
    before = perf_counter()
    slow = causal_prefix_lower(platform.model, oracle, elapsed_horizon=v13.HORIZON,
                               max_nodes=v13.MAX_NODES)
    slow_seconds = perf_counter() - before
    fast = solve(platform, oracle, prefix=True)
    assert slow.value == fast["value"] == archived["value"]
    return dict(bank=index, archived_value=archived["value"], slow_value=slow.value,
                slow_seconds=slow_seconds, slow_nodes=slow.information_nodes, event_driven=fast)


def load_panel(case, data, fit, inventories, plans, journeys, kernel):
    day, query, decision = case
    graph, _, stops, _, metadata, _ = v13.dated_graph(data, day, "gtfs_station_max", True)
    history, journey = inventories[case], journeys[day, query]
    prefix = tuple(history["next_transfer_boundary"]["prefix"])
    belief, root = v13.prepare_waiting_context(graph, metadata, journey, decision, fit,
                                               prefix, history["recorded_incumbent"]["rides"])
    goal, deadline = stops[journey["od"]["destination"]], journey["od"]["deadline_seconds"]
    parameters = {c.route: fit["route_priors"].get(c.route, fit["route_priors"]["pooled"])
                  for c in graph.connections}
    plan = v13.load_plan(plans[case]["fixed_primary_tail_rescue"]["plan"], prefix)
    hidden = graph.connections[plan.primary[0]].trip_id, graph.connections[plan.rescue[0]].trip_id
    seed = [20261113, int(day.replace("-", "")), query, decision, 13]
    rest = sample_clocks(belief, parameters, np.random.default_rng(np.random.SeedSequence(seed + [0])),
                         v13.BANKS, templates=template_trip_map(belief, parameters))
    factors = {trip: v13.hidden_draws(belief, parameters, trip, seed + [1, j])
               for j, trip in enumerate(hidden)}
    scan = TemporalScan(graph, kernel)
    transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
    return graph, root, goal, deadline, scan, transfers, plan, v13.compile_arms(graph, plan, root, deadline), rest, factors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, nargs="+", choices=BLOCKS, default=BLOCKS)
    args = parser.parse_args()
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    before = perf_counter()
    data, _ = v13.load_city(v13.SOURCE)
    fit = v13.fit_training(data, retain_no_pickup=True)
    inventories = {v13.key(row): row for row in v13.read(v13.INVENTORY)["rows"]}
    plans = {v13.key(row): row for row in v13.read(v13.V10)["rows"]}
    journeys = {(row["date"], row["query"]): row for row in v13.read(v13.ARCHIVE)["rows"]
                if row["method"] == "evolving_rollout"}
    archived = {v13.key(row): row for row in v13.read(ARCHIVED_RESULTS)["rows"]
                if row["arm"] == "product" and row["bank"] == 0}
    out = dict(status="running", frozen_v13_commit=SOURCE_REVISION,
        scope="Exact attribution conditional on each explicitly declared 2×2×2 finite law; "
              "the law is not the original fitted city posterior, and block results are not a population interval.",
        selection="Chronological V13 bank index i in 0,2,4,6. Rest draws i and i+1, "
                  "first two A and B draws of V13 bank i. No selection by policy values.",
        probabilities="Rest/A/B each uniform on two separately indexed atoms. "
                      "Resampling uses independent size-four multinomial draws with replacement for A/B, "
                      "conditional on a single uniformly drawn revealed rest atom.",
        attribution="A=J−V*, B=V*−V_R, C=V_R−V_Rτ, D=V_Rτ−E[Z_4×4]. "
                    "Full causal solves stop only at the journey terminal, and τ uses original V13 first ride or 720s.",
        max_nodes_per_solve=MAX_NODES, max_nodes_per_pathwise_witness=WITNESS_NODES,
        selected_blocks=list(args.blocks), cpu=cpu, roots=[])
    path = HERE / "finite_attribution.json"

    def save():
        out["elapsed_seconds"] = perf_counter() - before
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    with TemporaryDirectory(prefix="bapr_v15_finite_") as tmp:
        kernel = build_temporal_kernel(Path(tmp) / "scan.so")
        # Visit one block for every root before computing any additional block.
        panels = {}
        roots = {}
        for case in v13.CASES:
            panel = load_panel(case, data, fit, inventories, plans, journeys, kernel)
            panels[case] = panel
            check = audit_archived_prefix(panel, archived[case])
            roots[case] = dict(case=case, root_time=panel[1].time, stop=panel[1].stop,
                               deadline=panel[3], hidden_trips=tuple(panel[-1]),
                               original_bank_check=check, blocks=[])
            out["roots"].append(roots[case])
            save()
        for index in args.blocks:
            for case in v13.CASES:
                row = analyze_block(panels[case], index)
                roots[case]["blocks"].append(row)
                save()
                print(case, index, row["status"], row["terms"], flush=True)
    out["status"] = "complete" if all(row["status"] == "exact" for root in out["roots"]
                                       for row in root["blocks"]) else "incomplete"
    save()


if __name__ == "__main__":
    main()
