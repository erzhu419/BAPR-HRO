"""Private conditional futures on the frozen V13 city waiting problems."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
from math import isclose
import os
from pathlib import Path
from statistics import mean
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v13 import factored_run as v13
from experiments.reconstruction_v14.run_city_ablation import make_event, suffix_draw
from experiments.reconstruction_v15.run_attribution_banks import pack_product, prepare, rng, scanner
from src.causal_prefix import PrefixBudgetExceeded
from src.continuation_confidence import continuation_lower
from src.event_platform_solver import EventPlatform
from src.temporal_scan import TemporalScan, build_temporal_kernel, sample_clocks, template_trip_map


ALPHA = .05 / (3 * 6)
MAX_NODES = 50_000
ARCHIVED = HERE.parent / "reconstruction_v13" / "factored_results.json"


def exact_example():
    """Independent public departures and private ride durations; B arrives at 10."""
    atoms = tuple(product((2, 3), (1, 20)))

    def optimum(rows, private_exact=False):
        dep_times = {dep for dep, _ in rows}
        return sum(
            sum(when == dep for when, _ in rows) / len(rows)
            * min(10, dep + (10.5 if private_exact else mean(t for when, t in rows if when == dep)))
            for dep in dep_times
        )

    values, stratified = [], []
    for rows in product(atoms, repeat=4):
        values.append(optimum(rows))
        stratified.append(optimum(rows, private_exact=True))
    exact = optimum(atoms)
    assert exact == 10. and mean(values) == 8.52734375 and set(stratified) == {10.}
    return dict(population=exact, policy_loss=10., original_expected_bank=mean(values),
                analytic_private_expected_bank=mean(stratified), banks=len(values),
                optimism_before=exact-mean(values), optimism_after=exact-mean(stratified))


def event_map(law, slots, states):
    return {slots[i].cid: make_event(law, slots, states, i) for i in range(0, len(slots), 2)}


def conditional_arrival(law, departure_slot, anchor):
    arrival = departure_slot + 1
    logs = (law._transition(departure_slot)[anchor] + law.emissions[arrival]
            + law._backward_from(len(law.times)-1, arrival))
    prob = np.exp(logs - logsumexp(logs))
    return prob / prob.sum()


def strata(prob):
    """Two nonempty bins with probability split nearest to one half."""
    supported = np.flatnonzero(prob > 0.)
    if len(supported) < 2:
        return (supported,)
    split = 1 + np.argmin(abs(np.cumsum(prob[supported])[:-1] - .5))
    return supported[:split], supported[split:]


def conditional_factor(graph, belief, parameters, root, cid, seed, residual_seed, deadline):
    trip, cut = belief._positions[cid, "departure"]
    law, slots = belief._law(trip, parameters), belief._trips[trip]
    original = law.draw_uniforms_batch(
        rng(seed).random((v13.BANKS * v13.ATOMS, len(slots)))[:, ::-1])
    later_local = [slot.cid for slot in slots[cut+1:]
                   if (graph.connections[slot.cid].dep_stop if slot.kind == "departure"
                       else graph.connections[slot.cid].arr_stop) == root.stop]
    output = []
    for atom, states in enumerate(original):
        departure_anchor, original_arrival = int(states[cut]), int(states[cut+1])
        prob = conditional_arrival(law, cut, departure_anchor)
        assert prob[original_arrival] > 0.
        original_events = event_map(law, slots, states)
        copies = []
        for k, indices in enumerate(strata(prob)):
            weights = prob[indices]
            mass = float(weights.sum())
            if original_arrival in indices:
                selected = original_arrival
                events = original_events
            else:
                residual = rng(residual_seed + [atom, k])
                selected = int(residual.choice(indices, p=weights / mass))
                revised = states.copy()
                revised[cut+1] = selected
                revised[cut+2:] = suffix_draw(law, cut+1, selected,
                                               residual.random(len(slots)))[cut+2:]
                events = event_map(law, slots, revised)
            assert events[cid].departure == original_events[cid].departure
            assert events[cid].canceled == original_events[cid].canceled
            assert events[cid].arrival == law.times[cut+1][selected]
            copies.append(dict(weight=mass, events=events, selected_arrival=int(law.times[cut+1][selected]),
                               state_count=len(indices), is_original=selected == original_arrival))
        assert isclose(sum(c["weight"] for c in copies), 1.)
        arrivals = law.times[cut+1]
        clipped = np.minimum(deadline, arrivals)
        value = float(prob @ clipped)
        output.append(dict(original=original_events, copies=copies, clipped_mean=value,
                           clipped_sd=float(np.sqrt(prob @ ((clipped-value)**2))),
                           next_state_support=int(np.count_nonzero(prob)),
                           latent_departure=int(law.times[cut][departure_anchor]),
                           canceled=bool(law.canceled[cut][departure_anchor])))
    return trip, later_local, output


def selections(rows):
    """A subsample chooses one private suffix independently for each prefix."""
    bins = [row["copies"] for row in rows]
    sizes = [len(group) for group in bins]
    offsets = np.cumsum([0, *sizes[:-1]])
    for indices in product(*(range(size) for size in sizes)):
        weights = np.zeros(sum(sizes))
        weights[offsets + np.array(indices)] = 1. / len(rows)
        selected = [group[i] for group, i in zip(bins, indices, strict=True)]
        mass = float(np.prod([copy["weight"] for copy in selected]))
        yield weights, mass, all(copy["is_original"] for copy in selected)


def weighted_waiting_values(platform, oracle, variants, horizon):
    """Evaluate multiple priors on one public waiting tree, with no ride actions pruned."""
    root = platform.initial
    assert root.current_trip is None
    cutoff = min(root.time+horizon, platform.deadline)
    root_key = root, tuple(range(variants.shape[1]))
    finished, pending, stack = {}, {}, [root_key]
    nodes = oracle_evaluations = 0
    while stack:
        key = stack[-1]
        if key in finished:
            stack.pop()
            continue
        state, support = key
        if key not in pending:
            nodes += 1
            if nodes > MAX_NODES:
                raise PrefixBudgetExceeded(MAX_NODES, nodes, 0)
            actions = platform.model.actions(state)
            weights = variants[:, support]
            if state.time >= cutoff or not actions:
                finished[key] = weights @ np.array([oracle(i, state) for i in support])
                oracle_evaluations += len(support)
                continue
            boards = []
            waits = []
            for action in actions:
                outcomes = list(platform.outcomes(state, support, action, cutoff))
                assert tuple(i for i, _ in outcomes) == support
                if action[0] == "board":
                    boards.append(weights @ np.array([out.cost + oracle(i, out.state) for i, out in outcomes]))
                    oracle_evaluations += len(support)
                else:
                    assert action[0] in ("observe", "wait") and not waits
                    children = defaultdict(list)
                    for i, out in outcomes:
                        children[out.state].append(i)
                    waits = [(child, tuple(indices)) for child, indices in children.items()]
            assert waits
            wait_costs = sum(variants[:, members].sum(axis=1) * (child.time-state.time)
                             for child, members in waits)
            pending[key] = boards, waits, wait_costs
        boards, waits, wait_costs = pending[key]
        missing = next((child for child in waits if child not in finished), None)
        if missing is None:
            wait = wait_costs + sum(finished[child] for child in waits)
            finished[key] = np.minimum.reduce([wait, *boards])
            del pending[key]
        else:
            stack.append(missing)
    return finished[root_key], nodes, oracle_evaluations


def run_bank(graph, scan, transfers, root, goal, deadline, clocks, factors, eligible,
             targets, bank_index, archived):
    before = perf_counter()
    atoms = {trip: rows[bank_index*v13.ATOMS:(bank_index+1)*v13.ATOMS]
             for trip, rows in factors.items()}
    unweighted = {trip: [atom["original"] for atom in rows] for trip, rows in atoms.items()}
    old_bank, old_clocks, _ = pack_product(graph, clocks, unweighted)
    old = EventPlatform(graph, old_bank, root, goal, deadline)
    old_oracle = scanner(scan, old_clocks, root, goal, deadline, transfers)
    old_result = old.solve(old_oracle, elapsed_horizon=v13.HORIZON, max_nodes=MAX_NODES)
    if archived is not None:
        assert old_result.value == archived, (bank_index, old_result.value, archived)
    old_seconds = perf_counter()-before

    # Exact one-step conditional centering; the remaining clairvoyant tail is sampled.
    original_indices = tuple(product(*(range(len(rows)) for rows in atoms.values())))
    means = [{cid: atoms[trip][index]["clipped_mean"]
              for trip, cid, index in zip(atoms, targets, indices, strict=True) if eligible[trip]}
             for indices in original_indices]

    def centered(i, state):
        remaining = old_oracle(i, state)
        if state.used_connections == root.used_connections or state.last_connection not in means[i]:
            return remaining
        adjusted = remaining + means[i][state.last_connection] - state.time
        return min(adjusted, deadline-state.time)

    before = perf_counter()
    centered_result = old.solve(centered, elapsed_horizon=v13.HORIZON, max_nodes=MAX_NODES)
    center_seconds = perf_counter()-before

    # Fixed branch probabilities, private suffixes, and the full physical action model.
    before = perf_counter()
    choices = {trip: [(copy["weight"]/v13.ATOMS, copy["events"])
                      for atom in rows for copy in atom["copies"]]
               for trip, rows in atoms.items()}
    new, packed, _ = pack_product(graph, clocks, {k: [events for _, events in v] for k, v in choices.items()})
    probabilities = tuple(a[0]*b[0] for a, b in product(*choices.values()))
    total_mass = sum(probabilities)
    assert isclose(total_mass, 1.)
    weighted = tuple((p/total_mass, scene) for p, (_, scene) in zip(probabilities, new, strict=True))
    platform = EventPlatform(graph, weighted, root, goal, deadline)
    oracle = scanner(scan, packed, root, goal, deadline, transfers)
    build_seconds = perf_counter()-before
    before = perf_counter()
    paired = [(np.outer(a[0], b[0]).ravel(), a[1]*b[1], a[2] and b[2])
              for a, b in product(*(selections(rows) for rows in atoms.values()))]
    variants = np.vstack([np.array(probabilities)/total_mass, *[weights for weights, _, _ in paired]])
    selection_probabilities = np.array([p for _, p, _ in paired])
    assert isclose(selection_probabilities.sum(), 1.)
    conditional_difference = None
    try:
        values, nodes, evaluations = weighted_waiting_values(platform, oracle, variants, v13.HORIZON)
        original_index, = [i for i, (_, _, original) in enumerate(paired, start=1) if original]
        assert isclose(values[original_index], old_result.value, abs_tol=1e-7)
        if bank_index == 0:
            reference = platform.solve(oracle, elapsed_horizon=v13.HORIZON, max_nodes=MAX_NODES)
            assert isclose(values[0], reference.value, abs_tol=1e-7)
        conditional_original = float(selection_probabilities @ values[1:])
        difference = float(values[0] - conditional_original)
        assert difference >= -1e-7
        conditional_difference = dict(value=max(0., difference), mean_original_given_augmented=conditional_original,
                                      enumerated_subsamples=len(paired))
        stratified = dict(status="complete", value=float(values[0]), nodes=nodes, oracle_evaluations=evaluations)
    except PrefixBudgetExceeded as error:
        # A per-world clairvoyant minimum is a pointwise lower bound on
        # the corresponding fully solved conditional empirical bank value.
        fallback = sum(w * oracle(i, root) for i, (w, _) in enumerate(weighted))
        stratified = dict(status="fallback_clairvoyant", value=fallback, nodes=error.nodes-1)
    stratified.update(solve_seconds=perf_counter()-before, preparation_seconds=build_seconds,
                      worlds=len(weighted))
    private = {trip: [dict(next_state_support=atom["next_state_support"],
                          prefix_departure=atom["latent_departure"], canceled=atom["canceled"],
                          clipped_conditional_sd=atom["clipped_sd"],
                          clipped_conditional_mean=atom["clipped_mean"],
                          branches=[{k: v for k, v in branch.items() if k != "events"}
                                    for branch in atom["copies"]]) for atom in rows]
               for trip, rows in atoms.items()}
    return dict(bank=bank_index, original=dict(value=old_result.value, worlds=len(old_bank),
                                               nodes=old_result.information_nodes, wall_seconds=old_seconds),
                centered=dict(value=centered_result.value, nodes=centered_result.information_nodes,
                              wall_seconds=center_seconds), stratified=stratified,
                conditional_difference=conditional_difference or dict(value=0., fallback_zero=True), private=private)


def summarize(rows, cap):
    levels = {arm: np.array([row[arm]["value"] for row in rows])
              for arm in ("original", "centered", "stratified")}
    assert all(np.all((0. <= level) & (level <= cap)) for level in levels.values())
    result = {arm: dict(mean=float(level.mean()), lower=float(continuation_lower(level, cap, ALPHA)))
              for arm, level in levels.items()}
    for arm in ("centered", "stratified"):
        delta = levels[arm]-levels["original"]
        result[arm].update(mean_paired_increment=float(delta.mean()), improved=int(np.count_nonzero(delta > 1e-7)),
                           worsened=int(np.count_nonzero(delta < -1e-7)),
                           D_lower=max(0., float(continuation_lower(delta+cap, 2*cap, ALPHA)-cap)))
    rb = np.array([row["conditional_difference"]["value"] for row in rows])
    result["conditional_difference"] = dict(mean=float(rb.mean()), D_lower=float(continuation_lower(rb, cap, ALPHA)),
                                            nonzero=int(np.count_nonzero(rb > 1e-7)))
    return result


def evaluate(case, cpu, fresh):
    os.sched_setaffinity(0, {cpu})
    before = perf_counter()
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    day, query, decision = case
    date = int(day.replace("-", ""))
    base = [20261116, date, query, decision, 16, 31] if fresh else [20261113, date, query, decision, 13]
    targets = (plan.primary[0], plan.rescue[0])
    archived = {} if fresh else {r["bank"]: r["value"] for r in v13.read(ARCHIVED)["rows"]
                                 if v13.key(r) == case and r["arm"] == "product"}
    with TemporaryDirectory(prefix="bapr_v16_conditional_") as tmp:
        kernel = build_temporal_kernel(Path(tmp) / "scan.so")
        scan = TemporalScan(graph, kernel)
        transfers = [graph.stops[stop].min_transfer_time for stop in scan.stops]
        rest = sample_clocks(belief, parameters, rng(base+[0]), v13.BANKS,
                             templates=template_trip_map(belief, parameters))
        precompute_start = perf_counter()
        factors, later = {}, {}
        for j, cid in enumerate(targets):
            trip, later_local, options = conditional_factor(
                graph, belief, parameters, root, cid, base+[1,j],
                [20261116, date, query, decision, j, 41] if fresh else [20261116, date, query, decision, j], deadline)
            factors[trip], later[trip] = options, later_local
        factor_seconds = perf_counter()-precompute_start
        setup_seconds = perf_counter()-before
        assert root.current_trip is None
        eligible = {trip: not events for trip, events in later.items()}
        rows = []
        for bank_index in range(v13.BANKS):
            clocks = tuple(field[bank_index] for field in rest)
            rows.append(run_bank(graph, scan, transfers, root, goal, deadline, clocks, factors,
                                 eligible, targets, bank_index, archived.get(bank_index)))
    return dict(case=case, root_time=root.time, root_stop=root.stop, targets=targets,
                later_root_local_events=later, centering_eligible=eligible,
                factor_precompute_seconds=factor_seconds, setup_seconds=setup_seconds,
                total_seconds=perf_counter()-before, cap=deadline-root.time,
                rows=rows, summary=summarize(rows, deadline-root.time))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="Use an independent fitted-model tape.")
    args = parser.parse_args()
    results = dict(status="running", scope="Same V13 frozen model, revealed R and first-ride/720s oracle tail.",
        panel="V16 independent fitted-model tape" if args.fresh else "V13 development tape",
        exact_example=exact_example(), max_nodes=MAX_NODES, alpha_per_endpoint=ALPHA,
        contract="All original integer-clock platform actions and actual public observations retained. "
                 "No original A/B prefixes, visible observations or public action choices removed.",
        center="For an eligible A/B board, replace the clipped arrival cost with its exact conditional mean given "
               "the sampled latent departure anchor. Keep sampled oracle continuation; clip total loss at the cap. "
               "Enable only when no later same-trip event is observable at the waiting stop before boarding.",
        stratified="Each original departure-prefix atom has two private next-arrival strata of exact probability "
                   "under its chain conditional law. Keep original future in its own stratum and draw a true-law "
                   "residual conditional suffix in the other. Compare fixed-policy costs before minimizing. "
                   "Residual futures are sampled, not integrated exactly.",
        conditional_difference="Enumerate all independently selected one-future-per-prefix subbanks and average their "
                               "minima with exact stratum weights. Jensen makes the difference pointwise nonnegative; "
                               "its expectation is at most D. Substitute zero for an unfinished conditional comparison.",
        inference="Eight independent outer banks per root. Bonferroni .05 over three roots and six one-sided "
                  "endpoints per root. An upper bound on V_Rtau is not implied by empirical stratification. "
                  "The penalty is valid but need not improve Z. Stratification improvement holds in expectation.",
        roots=[])
    path = HERE/("conditional_bound_fresh_results.json" if args.fresh else "conditional_bound_results.json")
    cpus = sorted(os.sched_getaffinity(0))
    with ProcessPoolExecutor(max_workers=min(len(cpus), len(v13.CASES))) as pool:
        jobs = [pool.submit(evaluate, case, cpus[i % len(cpus)], args.fresh) for i, case in enumerate(v13.CASES)]
        for job in as_completed(jobs):
            result = job.result()
            results["roots"].append(result)
            results["roots"].sort(key=lambda row: row["case"])
            path.write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
            print(result["case"], result["summary"], flush=True)
    results["status"] = "complete"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
