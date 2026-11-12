"""Replay V18 on the three previously inspected actual platform records."""

from copy import copy
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from experiments.reconstruction_v13 import platform_run as v13
from experiments.reconstruction_v17.run_offgrid_audit import (
    compact_replay, missing_departure_invariance, read,
)
from experiments.reconstruction_v18.run_policy_pair import RELEASE_OFFSET, compile_comparators


HERE = Path(__file__).resolve().parent
REQUIRED_ARMS = {'first_visible', 'through_seconds', 'support_old_wait', 'support_through_wait'}


def gain_interval(reference, policy):
    return [reference['loss_lower'] - policy['loss_upper'],
            reference['loss_upper'] - policy['loss_lower']]


def audit_missing_departures(replay, setup, plan, model, replay_arm, baseline):
    """Identify a trace only if the unresolved offers cannot affect any root action."""
    policy = copy(setup['controller'])
    missing = set(replay['missing_connections'])
    wait_ids = getattr(policy.waiting_score.__self__, 'through_ids', ())
    if not missing or not missing <= policy.suffixes.keys() or missing & {
        *plan.primary, *plan.rescue, *wait_ids,
    }:
        return None

    # The alternative trace keeps the ordinary audit of issued A/B and ridden
    # suffixes. Its relaxation is only at the root menu, and is tested below.
    policy.censor_unknown_extras = False
    candidate = replay_arm(dict(setup, controller=policy))
    candidate['gain_interval_vs_baseline'] = gain_interval(baseline, candidate)
    # Use this V18 arm's actual wait callback, including through options when
    # enabled. The V17 helper enumerates current departure offers, not outcomes.
    policy.scores = SimpleNamespace(ready=policy.scores.ready, waiting_score=policy.waiting_score)
    audit = missing_departure_invariance(model, policy, candidate, sorted(missing))
    return dict(audit=audit, recorded_trace=candidate)


def run_case(case, dated, fit, journey, inventory, archived_plan, old_baseline):
    graph, actual, stops, _, metadata, _ = dated
    prefix = tuple(inventory['next_transfer_boundary']['prefix'])
    belief, root = v13.prepare_waiting_context(
        graph, metadata, journey, case[2], fit, prefix, inventory['recorded_incumbent']['rides'])
    goal = stops[journey['od']['destination']]
    deadline = journey['od']['deadline_seconds']
    plan = v13.load_plan(archived_plan['fixed_primary_tail_rescue']['plan'], prefix)
    parameters = {connection.route: fit['route_priors'].get(connection.route, fit['route_priors']['pooled'])
                  for connection in graph.connections}
    model = v13.build_model(graph, actual, root, goal, deadline)
    compiled = compile_comparators(graph, belief, parameters, root, goal, deadline, plan,
                                   censor_unknown_extras=True)
    assert REQUIRED_ARMS <= compiled.keys()
    ready = max(root.time, root.last_arrival + graph.stops[root.stop].min_transfer_time)
    publish = root.time + RELEASE_OFFSET

    def replay_arm(name, setup):
        replay = compact_replay(v13.replay_arm(graph, plan, root, goal, deadline, model,
                                model.scenarios[0], name, setup, actual=actual, metadata=metadata,
                                publish_not_before=publish))
        assert replay['publish_at'] == publish <= ready
        for selected in replay['selected_rescues']:
            kind, cid = selected['action']
            assert kind == 'board' and ready <= selected['time']
            assert selected['time'] == actual.events[cid].departure
        return replay

    rows = {name: replay_arm(name, setup) for name, setup in compiled.items()}
    baseline = rows['first_visible']
    assert all(baseline[key] == old_baseline[key] for key in ('status', 'loss_lower', 'loss_upper'))
    missing_audits = {}
    for name, replay in rows.items():
        if name != 'first_visible':
            replay['gain_vs_first_visible'] = gain_interval(baseline, replay)
        if name.startswith('support_'):
            replay['gain_vs_through_seconds'] = gain_interval(rows['through_seconds'], replay)
            audit = audit_missing_departures(replay, compiled[name], plan, model,
                                            lambda setup: replay_arm(name, setup), baseline)
            if audit is not None:
                missing_audits[name] = audit
    support = compiled['support_through_wait']['controller']
    early = []
    for cid in sorted(support.suffixes):
        if graph.connections[cid].dep_time < ready:
            receipt = metadata[cid]['departure']
            early.append(dict(cid=cid, status=receipt['status'], time=receipt['time_seconds']))
    return dict(case=case, clock=root.time, earliest_board=ready, deadline=deadline, publication=publish,
                scheduled_early_candidate_receipts=early, arms=rows, missing_departure_audits=missing_audits)


def main():
    before = perf_counter()
    data, _ = v13.load_city(v13.SOURCE)
    fit = v13.fit_training(data, retain_no_pickup=True)
    fit_seconds = perf_counter() - before
    inventory = {v13.key(r): r for r in read(v13.INVENTORY)['rows']}
    plans = {v13.key(r): r for r in read(v13.V10)['rows']}
    journeys = {(r['date'], r['query']): r for r in read(v13.ARCHIVE)['rows']
                if r['method'] == 'evolving_rollout'}
    baseline = {v13.key(r): r for r in read(v13.HERE/'platform_results.json')['rows']
                if r['panel'] == 'waiting' and r['source'] == 'actual' and r['method'] == 'first_visible'}
    by_day = {day: v13.dated_graph(data, day, 'gtfs_station_max', True)
              for day in sorted({case[0] for case in v13.CASES})}
    roots = [run_case(case, by_day[case[0]], fit, journeys[case[:2]], inventory[case], plans[case],
                      baseline[case]) for case in v13.CASES]
    result = dict(status='complete', shared_load_fit_seconds=fit_seconds,
                  scope='Previously inspected actual platform records. Only public actual-second events '
                        'are used by the issued policies. Missing relevant source rows censor at '
                        'elapsed-to-deadline unless their irrelevance is checked at every feasible '
                        'departure second. This is a historical diagnostic under the ideal local '
                        'event-response contract.', roots=roots)
    output = HERE/'actual_replay.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    for root in roots:
        print(root['case'], {arm: [row['status'], row['loss_lower'], row['loss_upper'],
                                  row['missing_connections']] for arm, row in root['arms'].items()})


if __name__ == '__main__':
    main()
