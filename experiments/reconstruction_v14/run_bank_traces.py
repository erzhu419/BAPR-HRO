"""Reconstruct the optimal causal-prefix policy on all frozen V13 product banks."""
from collections import Counter, defaultdict
from heapq import heappop, heappush
from itertools import count
import json
from math import isclose
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
from src.factored_lower import factored_prefix_lower, product_trip_bank
from src.platform_transit import build_platform_model
from src.simulator import JourneyScenario, VehicleEvent
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map


BASE = HERE.parent / "reconstruction_v13"
SOURCE_COMMIT = "28d97a37988ca014619cc76eefe28d18185caa2e"


def oracle_path(graph, scene, state, goal, deadline):
    """Return a witness for the V13 platform scan's relaxed continuation."""
    serial = count()
    best = {}
    queue = []

    def offer(stop, trip, clock, path):
        key = stop, trip
        if clock <= deadline and clock < best.get(key, deadline + 1):
            best[key] = clock
            heappush(queue, (clock, next(serial), stop, trip, path))

    ready = state.time
    if state.current_trip is not None:
        offer(state.stop, state.current_trip, state.time, ())
        ready += graph.stops[state.stop].min_transfer_time
    elif state.last_connection is not None:
        ready = max(ready, state.last_arrival + graph.stops[state.stop].min_transfer_time)
    if state.stop == goal:
        return state.time, ()
    offer(state.stop, None, ready, ())
    while queue:
        clock, _, stop, trip, path = heappop(queue)
        if clock != best[stop, trip]:
            continue
        if stop == goal:
            return clock, path
        for connection in graph.get_connections_from(stop):
            cid = connection.id
            if cid in state.used_connections:
                continue
            if trip is None and not connection.pickup_allowed:
                continue
            if trip is not None and trip != connection.trip_id:
                continue
            event = scene.events[cid]
            if event.canceled or not clock <= event.departure <= event.arrival <= deadline:
                continue
            target = connection.arr_stop
            route = path + (cid,)
            offer(target, connection.trip_id, event.arrival, route)
            if target != goal:
                offer(target, None, event.arrival + graph.stops[target].min_transfer_time, route)
    return deadline, ()


def ride(graph, scene, cid, hidden):
    connection, event = graph.connections[cid], scene.events[cid]
    label = dict(zip(hidden, ("A", "B"))).get(connection.trip_id, "R")
    return dict(cid=cid, trip=connection.trip_id, route=connection.route, role=label,
                origin=connection.dep_stop, destination=connection.arr_stop,
                departure=event.departure, arrival=event.arrival)


def replay_optimal(model, graph, bank, labels, goal, deadline, trace, oracle):
    """Follow the selected actions from the initial state on every product row."""
    root = model.initial_state
    hidden = tuple(labels)
    worlds = []
    for index, (_, scene) in enumerate(bank):
        state, support = root, tuple(range(len(bank)))
        public, steps, prefix = set(), [], []
        elapsed = 0.
        while True:
            options, boundary = trace[state, support]
            events = [event for event in state.public_log if event not in public]
            public.update(events)
            if boundary is not None:
                break
            ranked = sorted(options, key=lambda option: option[1])
            action, value = ranked[0]
            if events or len(ranked) > 1 or action[0] != "wait":
                step = dict(clock=state.time, stop=state.stop, support=list(support),
                            public_events=events, visible_connections=state.visible_connections,
                            action=action, expected_remaining_loss=value)
                if len(ranked) > 1:
                    step["runner_up"] = ranked[1][0]
                    step["runner_up_margin_seconds"] = ranked[1][1] - value
                    step["alternative_action_values"] = ranked[1:]
                steps.append(step)
            outcome = model.transition(model.scenarios[index], state, action)
            if action[0] == "board":
                prefix.append(ride(graph, scene, action[1], hidden))
            support = tuple(i for i in support
                            if model.transition(model.scenarios[i], state, action) == outcome)
            state = outcome.state
            elapsed += outcome.cost
        terminal, path = oracle_path(graph, scene, state, goal, deadline)
        assert terminal - state.time == oracle(index, state), (index, terminal, state)
        worlds.append(dict(world=index, factor_indices=[labels[trip][index] for trip in hidden],
                           steps=steps, prefix=prefix,
                           boundary=dict(kind=boundary, clock=state.time, stop=state.stop,
                                         support=list(support), new_public_events=events,
                                         current_trip=state.current_trip),
                           clairvoyant_tail=[ride(graph, scene, cid, hidden) for cid in path],
                           clairvoyant_tail_loss=terminal - state.time,
                           loss=elapsed + terminal - state.time))
    return worlds


def audit_bank(graph, initial, goal, deadline, scan, plan, compiled, rest, factors, index, frozen):
    dep, arr, canceled = (field[index] for field in rest)
    common = JourneyScenario({
        c.id: VehicleEvent(int(dep[c.id]) - c.dep_time, bool(canceled[c.id]), int(dep[c.id]), int(arr[c.id]))
        for c in graph.connections
    }, minute=60, publications={})
    atoms = {trip: draws[index * v13.ATOMS:(index + 1) * v13.ATOMS] for trip, draws in factors.items()}
    bank, labels = product_trip_bank(graph, common, atoms)
    packed = tuple(np.repeat(field[None, :], len(bank), axis=0) for field in (dep, arr, canceled))
    for trip in factors:
        for row, atom in enumerate(labels[trip]):
            for cid, event in atoms[trip][atom].items():
                packed[0][row, cid], packed[1][row, cid], packed[2][row, cid] = (
                    event.departure, event.arrival, event.canceled)
    model = build_platform_model(graph, bank, initial.stop, goal, initial.time, deadline,
                                 initial_state=initial, poll_interval=1)
    transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]

    def oracle(i, public):
        return float(scan.platform_arrivals(*(field[i:i + 1] for field in packed),
                                            public, goal, deadline, transfers)[0] - public.time)

    trace = {}

    def record(state, indices, options, boundary):
        trace[state, indices] = options, boundary

    result = factored_prefix_lower(model, oracle, graph=graph, factor_labels=labels,
                                   elapsed_horizon=v13.HORIZON, max_nodes=v13.MAX_NODES, trace=record)
    assert isclose(result.value, frozen["value"], abs_tol=1e-7)
    assert isclose(result.clairvoyant_root_mean, frozen["clairvoyant_root_mean"], abs_tol=1e-7)
    worlds = replay_optimal(model, graph, bank, labels, goal, deadline, trace, oracle)
    assert isclose(mean(w["loss"] for w in worlds), result.value, abs_tol=1e-7)

    counts, gains = Counter(), Counter()
    first_rides = Counter()
    for world, scenario, archived_loss in zip(worlds, model.scenarios,
                                               frozen["policies"]["first_visible"]["losses"], strict=True):
        policy = v13.replay_arm(graph, plan, initial, goal, deadline, model, scenario,
                               "first_visible", compiled["first_visible"])
        assert policy["loss_upper"] == policy["loss_lower"] == archived_loss
        world["first_visible_loss"] = archived_loss
        world["first_visible_route"] = [
            ride(graph, bank[world["world"]][1], row["ridden_connection"], tuple(labels))
            for row in policy["rides"]
        ]
        difference = archived_loss - world["loss"]
        role = world["prefix"][0]["role"] if world["prefix"] else "no_prefix_ride"
        if world["prefix"]:
            first = world["prefix"][0]
            first_rides[first["role"], first["trip"], first["route"], first["cid"]] += 1
        counts[role] += 1
        gains[role] += difference
        world["first_visible_minus_relaxation"] = difference
    return dict(bank=index, value=result.value, clairvoyant_root_mean=result.clairvoyant_root_mean,
                replay_value=mean(w["loss"] for w in worlds),
                point_gap=mean(w["first_visible_minus_relaxation"] for w in worlds),
                first_prefix_role_counts=dict(counts), loss_difference_sum_by_role=dict(gains),
                first_rides=[dict(role=k[0], trip=k[1], route=k[2], cid=k[3], count=n)
                             for k, n in first_rides.most_common()], worlds=worlds)


def summarize_banks(root):
    roles, boundaries, r_boarding = defaultdict(Counter), Counter(), Counter()
    fixed_menu = {root["main_cid"], root["rescue_cid"]}
    worlds = [world for bank in root["banks"] for world in bank["worlds"]]
    for world in worlds:
        prefix = world["prefix"]
        role = prefix[0]["role"] if prefix else "no_prefix_ride"
        gain = world["first_visible_minus_relaxation"]
        roles[role].update(rows=1, positive=int(gain > 0), negative=int(gain < 0),
                           tied=int(gain == 0), loss_difference_sum=gain)
        boundaries[world["boundary"]["kind"]] += 1
        if role == "R":
            step = next(step for step in world["steps"] if step["action"][0] == "board")
            options = step["alternative_action_values"]
            fixed_board_now = any(action[0] == "board" and action[1] in fixed_menu
                                  for action, _ in options)
            r_boarding.update(rows=1, fixed_ab_boardable_now=int(fixed_board_now),
                              strictly_better_than_runner_up=int(step["runner_up_margin_seconds"] > 0))
    return dict(n_worlds=len(worlds), by_first_prefix_role=dict(roles),
                boundaries=dict(boundaries), r_boarding=dict(r_boarding),
                point_gap=mean(world["first_visible_minus_relaxation"] for world in worlds))


def main():
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    start = perf_counter()
    frozen = {(v13.key(r), r["bank"]): r for r in v13.read(BASE / "factored_results.json")["rows"]
              if r["arm"] == "product"}
    data, _ = v13.load_city(v13.SOURCE)
    fit = v13.fit_training(data, retain_no_pickup=True)
    inventory = {v13.key(r): r for r in v13.read(v13.INVENTORY)["rows"]}
    plans = {v13.key(r): r for r in v13.read(v13.V10)["rows"]}
    journeys = {(r["date"], r["query"]): r for r in v13.read(v13.ARCHIVE)["rows"]
                if r["method"] == "evolving_rollout"}
    out = dict(source_commit=SOURCE_COMMIT, original_protocol="experiments/reconstruction_v13/factored_protocol.json",
               scope="V13 frozen product-bank policy traces. R and boundary tails have future information.",
               public_events="Only actually logged platform events are shown. The rest realization is also revealed by the bank design.",
               ranking="Expected remaining loss conditional on the V13 information node. Does not identify a causal effect of any one event.",
               path="First prefix rides follow exact selected actions. Tail is one matching clairvoyant path under the declared relaxation.",
               cpu=cpu, roots=[])
    with TemporaryDirectory(prefix="bapr_v14_trace_") as tmp:
        kernel = build_temporal_kernel(Path(tmp) / "scan.so")
        for case in v13.CASES:
            day, query, decision = case
            graph, _, stops, _, metadata, _ = v13.dated_graph(data, day, "gtfs_station_max", True)
            record, journey = inventory[case], journeys[day, query]
            prefix = tuple(record["next_transfer_boundary"]["prefix"])
            belief, initial = v13.prepare_waiting_context(graph, metadata, journey, decision, fit,
                                                         prefix, record["recorded_incumbent"]["rides"])
            goal, deadline = stops[journey["od"]["destination"]], journey["od"]["deadline_seconds"]
            parameters = {c.route: fit["route_priors"].get(c.route, fit["route_priors"]["pooled"])
                          for c in graph.connections}
            plan = v13.load_plan(plans[case]["fixed_primary_tail_rescue"]["plan"], prefix)
            hidden = (graph.connections[plan.primary[0]].trip_id, graph.connections[plan.rescue[0]].trip_id)
            seed = [20261113, int(day.replace("-", "")), query, decision, 13]
            rest = sample_clocks(belief, parameters, np.random.default_rng(np.random.SeedSequence(seed + [0])),
                                 v13.BANKS, templates=template_trip_map(belief, parameters))
            factors = {trip: v13.hidden_draws(belief, parameters, trip, seed + [1, j])
                       for j, trip in enumerate(hidden)}
            scan = TemporalScan(graph, kernel)
            compiled = v13.compile_arms(graph, plan, initial, deadline)
            results = []
            for index in range(v13.BANKS):
                result = audit_bank(graph, initial, goal, deadline, scan, plan, compiled,
                                    rest, factors, index, frozen[case, index])
                results.append(result)
                print(case, index, result["replay_value"], result["first_prefix_role_counts"], flush=True)
            root = dict(date=day, query=query, decision=decision, root_time=initial.time,
                        root_stop=initial.stop, deadline=deadline, main_cid=plan.primary[0],
                        rescue_cid=plan.rescue[0], hidden_trips=hidden, banks=results)
            root["summary"] = summarize_banks(root)
            out["roots"].append(root)
    out["elapsed_seconds"] = perf_counter() - start
    (HERE / "bank_traces.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
