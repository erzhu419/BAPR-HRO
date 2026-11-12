"""Audit actual-second scores and execute bounded, source-aware replays."""

from collections import Counter
from copy import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter, process_time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v13 import platform_run as v13
from src.conditional_rescue_policy import ConditionalItineraries
from src.offgrid_conditional import ActualClockRescue, ActualClockScores
from src.platform_transit import OBSERVE, WAIT
from src.public_rescue_candidates import screen_public_itineraries
from src.tail_rescue import TailRescuePlan

V16 = HERE.parent / 'reconstruction_v16'
MARGIN = 120


def read(path):
    return json.loads(path.read_text())


def available_actual_scores(clock_scores, cid, root, departure, archived):
    """A local departure receipt is the only new information passed to either scorer."""
    visible = replace(root, time=departure, visible_connections=(cid,),
                      public_log=((departure, cid, 'departure'),), initialized=True)
    hidden = replace(visible, public_log=())
    assert clock_scores.at_visible(cid, hidden, 'anchor') is None
    assert clock_scores.at_visible(cid, replace(visible, visible_connections=()), 'projection') is None
    root_evidence = clock_scores.belief._known_times.copy()
    result = dict(cid=cid, departure=departure, original_exact_available=departure in archived)
    for mode in ('projection', 'anchor'):
        before = perf_counter()
        score = clock_scores.at_visible(cid, visible, mode)
        seconds = perf_counter() - before
        assert score is None or score.clock == departure
        result[mode] = None if score is None else asdict(score) | {'seconds': seconds}
    assert clock_scores.belief._known_times == root_evidence
    return result


def audit(clocks, root, extras, metadata, archived):
    records = []
    for cid in sorted(extras):
        row = metadata[cid]['departure']
        state = dict(cid=cid, departure=row['time_seconds'], source_status=row['status'])
        if row['status'] != 'REAL':
            state['scope'] = 'unresolved_departure'
        elif not clocks.scores.ready <= row['time_seconds'] < clocks.scores.deadline:
            state['scope'] = 'outside_boarding_interval'
        else:
            state['scope'] = 'measured_within_boarding_interval'
            table = {item['clock'] for item in archived['conditionals'][str(cid)]}
            state.update(available_actual_scores(clocks, cid, root, row['time_seconds'], table))
        records.append(state)
    return records


def compact_replay(row):
    keep = ('method', 'status', 'loss_lower', 'loss_upper', 'published', 'rejected', 'publish_at',
            'missing_connections', 'rides', 'worker_seconds', 'total_rule_pipeline_seconds',
            'rule_work', 'triggers')
    out = {name: row[name] for name in keep}
    out['selected_rescues'] = [d for d in row['trigger_trace'] if d['branch'] == 'conditional_rescue']
    return out


def missing_departure_invariance(model, policy, replay, missing):
    """Try every unknown extra at every possible second before the observed choice.

    The V16 reference and previous actions ignore extra identities except when
    comparing currently visible paths. Adding earlier rejected extra receipts
    cannot change the original A/B cursor or later scores. Thus checking each
    missing candidate separately suffices even if they could appear together.
    No missing clock or continuation is assigned a realized value.
    """
    if not missing or replay['status'] != 'arrived' or len(replay['selected_rescues']) != 1:
        return None
    chosen = replay['selected_rescues'][0]
    chosen_clock, chosen_cid = chosen['time'], chosen['action'][1]
    binding = policy.original.bind('invariance_reference')
    state = model.initial_state
    world = model.scenarios[0]
    tested = Counter()
    scored = Counter()
    smallest_margin = {}
    changes = []
    while state.time <= chosen_clock:
        time = state.time
        if model.actions(state) == (OBSERVE,):
            action = OBSERVE
        elif time < replay['publish_at']:
            assert time < policy.scores.ready
            action = WAIT
        else:
            action = binding.action(state)
            if time >= policy.scores.ready:
                if action[0] == 'board':
                    row = policy.clock_scores.at_visible(action[1], state, policy.mode)
                    reference = row.expected_arrival_or_deadline if row is not None else None
                else:
                    reference = policy.scores.waiting_score(state, binding.cursor)
                if time == chosen_clock:
                    known = policy.clock_scores.at_visible(chosen_cid, state, policy.mode)
                    assert known is not None and known.expected_arrival_or_deadline == chosen['score']
                    assert reference == chosen['reference']
                for cid in missing:
                    tested[cid] += 1
                    observed = replace(state, visible_connections=tuple(sorted((*state.visible_connections, cid))),
                        public_log=tuple(sorted((*state.public_log, (time, cid, 'departure')))))
                    row = policy.clock_scores.at_visible(cid, observed, policy.mode)
                    if row is None or reference is None:
                        continue
                    scored[cid] += 1
                    gap = row.expected_arrival_or_deadline + policy.margin-reference
                    smallest_margin[cid] = min(smallest_margin.get(cid, float('inf')), gap)
                    would_win = time < chosen_clock or (row.expected_arrival_or_deadline, cid) < (
                        chosen['score'], chosen_cid)
                    if gap < 0 and would_win:
                        changes.append(dict(cid=cid, time=time, margin_against_reference=gap))
            if time == chosen_clock:
                break
            assert action == WAIT  # Recorded candidate remained on the original platform until this choice.
        state = model.transition(world, state, action).state
    checked_count = chosen_clock-policy.scores.ready+1
    assert tested == {cid: checked_count for cid in missing} and time == chosen_clock
    return dict(invariant=not changes, checked_from=policy.scores.ready, checked_through=chosen_clock,
                tested_seconds=tested, scored_seconds=scored, minimum_rejection_margin=smallest_margin,
                alternative_winning_departures=len(changes), first_alternative=changes[0] if changes else None,
                identified_loss=[replay['loss_lower'], replay['loss_upper']] if not changes else None,
                identified_gain=replay['gain_interval_vs_baseline'] if not changes else None)


def negative_control(model, policy, replay, missing):
    """Inject one favorable hypothetical score and require the proof to fail."""
    first = missing[0]
    ready = policy.scores.ready

    class FavorableScore:
        def at_visible(self, cid, state, mode):
            row = policy.clock_scores.at_visible(cid, state, mode)
            if row is not None and cid == first and state.time == ready:
                return replace(row, expected_arrival_or_deadline=float(state.time))
            return row

    modified = copy(policy)
    modified.clock_scores = FavorableScore()
    result = missing_departure_invariance(model, modified, replay, missing)
    assert result is not None and not result['invariant']
    assert result['first_alternative']['cid'] == first and result['first_alternative']['time'] == ready
    return result['first_alternative']


def evaluate(case, data, fit, inventory, journeys, old_plans, archived, actual_baseline):
    started = perf_counter()
    day, query, decision = case
    graph, actual, stops, _, metadata, _ = v13.dated_graph(data, day, 'gtfs_station_max', True)
    old, journey = inventory[case], journeys[day, query]
    prefix = tuple(old['next_transfer_boundary']['prefix'])
    belief, root = v13.prepare_waiting_context(graph, metadata, journey, decision, fit,
                                             prefix, old['recorded_incumbent']['rides'])
    goal, deadline = stops[journey['od']['destination']], journey['od']['deadline_seconds']
    plan = v13.load_plan(old_plans[case]['fixed_primary_tail_rescue']['plan'], prefix)
    pars = {c.route: fit['route_priors'].get(c.route, fit['route_priors']['pooled']) for c in graph.connections}
    model = v13.build_model(graph, actual, root, goal, deadline)
    world = model.scenarios[0]
    baseline_compiled = v13.compile_arms(graph, plan, root, deadline)['first_visible']
    baseline = compact_replay(v13.replay_arm(graph, plan, root, goal, deadline, model, world,
                               'first_visible', baseline_compiled, actual=actual, metadata=metadata))
    assert (baseline['status'], baseline['loss_lower'], baseline['loss_upper']) == (
        actual_baseline['status'], actual_baseline['loss_lower'], actual_baseline['loss_upper'])
    setup = perf_counter() - started

    before, cpu = perf_counter(), process_time()
    paths = {p[0]: TailRescuePlan(p, first_action_now=True, departures=plan.departures)
             for p in (plan.primary, plan.rescue)}
    extras = screen_public_itineraries(graph, root, goal, deadline, paths, 720)
    paths.update((cid, TailRescuePlan(itinerary.path, first_action_now=True))
                 for cid, itinerary in extras.items())
    assert {str(cid): list(p.primary) for cid, p in paths.items()} == archived['paths']
    scores = ConditionalItineraries(belief, pars, graph, root, goal, deadline, paths)
    clock_scores = ActualClockScores(belief, pars, scores, goal)
    prepare, prepare_cpu = perf_counter() - before, process_time() - cpu
    assert set(extras) <= set(scores.rows)
    records = audit(clock_scores, root, extras, metadata, archived)
    results, invariance, falsification = {}, {}, None
    for strict in (True, False):
        scope = 'source_safe' if strict else 'recorded_only_sensitivity'
        for mode in ('projection', 'anchor'):
            before, cpu = perf_counter(), process_time()
            policy = ActualClockRescue(graph, plan, root, deadline, scores, extras, MARGIN,
                                       clock_scores, mode=mode, censor_unknown_extras=strict)
            compiled = dict(controller=policy, construction_seconds=prepare+perf_counter()-before,
                            construction_cpu_seconds=prepare_cpu+process_time()-cpu)
            replay = compact_replay(v13.replay_arm(graph, plan, root, goal, deadline, model, world,
                                    f'{scope}_{mode}', compiled, actual=actual, metadata=metadata))
            assert replay['publish_at'] <= scores.ready
            assert all(r['time'] == actual.events[r['action'][1]].departure
                       for r in replay['selected_rescues'])
            replay['gain_interval_vs_baseline'] = [baseline['loss_lower']-replay['loss_upper'],
                                                    baseline['loss_upper']-replay['loss_lower']]
            results.setdefault(scope, {})[mode] = replay
            if not strict:
                missing = results['source_safe'][mode]['missing_connections']
                if set(missing) <= set(extras):
                    invariance[mode] = missing_departure_invariance(model, policy, replay, missing)
                    if case == v13.CASES[0] and mode == 'projection':
                        falsification = negative_control(model, policy, replay, missing)
    measured = [r for r in records if r['scope'] == 'measured_within_boarding_interval']
    both = [r for r in measured if r['projection'] is not None and r['anchor'] is not None]
    discrepancy = [abs(r['projection']['expected_arrival_or_deadline']-r['anchor']['expected_arrival_or_deadline'])
                   for r in both]
    return dict(case=case, clock=root.time, earliest_board=scores.ready, deadline=deadline,
                setup_seconds=setup, prepare_seconds=prepare, prepare_cpu_seconds=prepare_cpu,
                departure_scope_counts=dict(Counter(r['scope'] for r in records)),
                measured_eligible=len(measured), original_scored=sum(r['original_exact_available'] for r in measured),
                projection_scored=sum(r['projection'] is not None for r in measured),
                anchor_scored=sum(r['anchor'] is not None for r in measured),
                score_discrepancy=dict(mean_absolute=mean(discrepancy), max_absolute=max(discrepancy)),
                median_score_seconds={mode: median(r[mode]['seconds'] for r in both)
                                      for mode in ('projection', 'anchor')},
                baseline=baseline, replays=results, missing_departure_invariance=invariance,
                missing_invariance_negative_control=falsification,
                departure_audit=records)


def main():
    before = perf_counter()
    data, _ = v13.load_city(v13.SOURCE)
    fit = v13.fit_training(data, retain_no_pickup=True)
    fit_seconds = perf_counter() - before
    inventory = {v13.key(r): r for r in read(v13.INVENTORY)['rows']}
    old_plans = {v13.key(r): r for r in read(v13.V10)['rows']}
    journeys = {(r['date'], r['query']): r for r in read(v13.ARCHIVE)['rows']
                if r['method'] == 'evolving_rollout'}
    archive = read(V16/'held_out_policy.json')
    assert archive['status'] == 'complete' and archive['margins'] == [MARGIN]
    roots = {tuple(r['case']): r for r in archive['roots']}
    baseline = {v13.key(r): r for r in read(v13.HERE/'platform_results.json')['rows']
                if r['panel'] == 'waiting' and r['source'] == 'actual' and r['method'] == 'first_visible'}
    output = dict(scope='Existing three waiting roots and archived routes. No new dates or pulls.',
        physical_contract='Use the actual observed departure second and legal visible menu. '
            'Never round the passenger clock, make an unobserved departure available, or read an actual future event.',
        model='Projection interpolates scores over the old minute-grid brackets; anchor rebuilds the '
            'observed trip with the existing exact receipt conditioning on an expanded grid. '
            'Both retain V16 waiting heuristic. Neither extends the old bounds to a new timestamp model.',
        censor='Source-safe replay refuses to infer absence of an unresolved extra candidate once '
            'boarding becomes possible. Baseline and downstream source audits use existing V13 rules. '
            'Recorded-only is sensitivity conditional on no unrecorded extra departure while waiting. '
            'The separate invariance audit checks every integer clock for each missing extra through '
            'the first recorded rescue and identifies its replay result only if none could change the action.',
        latency='Preparation is charged at publication. Decision CPU and wall are recorded. Exact-second '
            'actions follow the existing ideal event response contract. On-demand anchor timing does not '
            'establish boardability if a deployment has zero dwell or a stricter physical reaction deadline.',
        shared_load_fit_seconds=fit_seconds,
        roots=[evaluate(case, data, fit, inventory, journeys, old_plans, roots[case], baseline[case])
               for case in v13.CASES])
    assert [r['measured_eligible'] for r in output['roots']] == [13, 9, 15]
    assert all(r['original_scored'] == 0 for r in output['roots'])
    assert all(r['anchor_scored'] == r['projection_scored'] == r['measured_eligible'] for r in output['roots'])
    path = HERE/'offgrid_audit_results.json'
    path.write_text(json.dumps(output, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
