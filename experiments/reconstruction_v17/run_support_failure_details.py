"""Reconstruct the two archived July 27 support-expansion timeouts."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.reconstruction_v10.platform_sensitivity import adapter
from experiments.reconstruction_v13.platform_run import build_model
from experiments.reconstruction_v14.run_bank_traces import oracle_path
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from experiments.reconstruction_v17.run_support_pair import compile_policy
from src.background_policy import IssuedPolicy, execute_background
from src.platform_transit import ALIGHT, WAIT
from src.recourse_itinerary_router import build_trip_blocks
from src.temporal_scan import build_temporal_kernel, sample_clocks, template_trip_map


CASE = ('2023-07-27', 0, 1)
FAILURES = (3, 99)


def describe(graph, scene, path):
    rows = []
    for cid in path:
        c, event = graph.connections[cid], scene.events[cid]
        rows.append(dict(cid=cid, trip=c.trip_id, origin=graph.stops[c.dep_stop].name,
            destination=graph.stops[c.arr_stop].name, scheduled_departure=c.dep_time,
            scheduled_arrival=c.arr_time, departure=event.departure, arrival=event.arrival,
            canceled=event.canceled, pickup_allowed=c.pickup_allowed))
    return rows


def replay_witness(graph, scene, state, goal, deadline, path):
    model = build_model(graph, scene, state, goal, deadline)
    policy, _, _ = adapter(graph, path, state, 'fixed_hindsight_witness')
    return execute_background(model, model.scenarios[0], policy, policy, state.time)


def direct_paths(graph, stop, goal):
    blocks, positions = build_trip_blocks(graph)
    paths = {}
    for first in graph.get_connections_from(stop):
        if not first.pickup_allowed or first.trip_id is None:
            continue
        path, previous_stop = [], stop
        for cid in blocks[first.trip_id][positions[first.id]:]:
            c = graph.connections[cid]
            if c.dep_stop != previous_stop:
                break
            path.append(cid)
            previous_stop = c.arr_stop
            if previous_stop == goal:
                paths[first.id] = tuple(path)
                break
    return paths


def replay_late_public_rescue(graph, scene, start, goal, deadline):
    """Board the first legally visible scheduled-through trip after failure."""
    paths = direct_paths(graph, start.stop, goal)
    selected = ()

    def action(state):
        nonlocal selected
        if not selected:
            options = [cid for cid in state.visible_connections if cid in paths]
            if options:
                cid = min(options, key=lambda c: (graph.connections[paths[c][-1]].arr_time, c))
                selected = paths[cid]
        next_ride = next((cid for cid in selected if cid not in state.used_connections), None)
        return ('board', next_ride) if next_ride in state.visible_connections else WAIT

    model = build_model(graph, scene, start, goal, deadline)
    policy = IssuedPolicy(action, name='late_public_through_rescue')
    replay = execute_background(model, model.scenarios[0], policy, policy, start.time)
    return replay, selected


def investigate(graph, scene, root, goal, deadline, plan, compiled, archived, paths):
    candidate = archived['policies']['support']
    model = build_model(graph, scene, root, goal, deadline)
    world = model.scenarios[0]
    incumbent, _, _ = adapter(graph, plan.primary, root, 'incumbent')
    binding = compiled['controller'].bind('support')
    original = execute_background(model, world, incumbent, binding.policy, candidate['publication'])
    assert original['cost'] == candidate['loss'] == deadline-root.time
    alight = next(row for row in binding.follow.trace if row['reason'] == 'transfer_alight')
    failure = next(row for row in binding.follow.trace if row['events'])
    failure_reason, failed_cid, failure_clock = failure['events'][0]
    first = candidate['first']
    first_state, transfer_state, failed_state = None, None, None
    state = root
    for step in original['trace']:
        if state.time == failure_clock and state.stop == failure['stop'] and state.current_trip is None:
            failed_state = state
        updated = model.transition(world, state, step['action']).state
        if step['ridden_connection'] == first:
            first_state = updated
        if step['action'] == ALIGHT and state.time == alight['time']:
            transfer_state = updated
        state = updated
    assert first_state is not None and transfer_state is not None and failed_state is not None
    mct = graph.stops[transfer_state.stop].min_transfer_time
    ready = transfer_state.last_arrival+mct
    event = scene.events[failed_cid]
    assert not event.canceled and event.departure < ready
    best_clock, path = oracle_path(graph, scene, first_state, goal, deadline)
    witness = replay_witness(graph, scene, replace(first_state, initialized=False), goal, deadline, path)
    assert witness['state'].stop == goal and witness['state'].time == best_clock
    late, _ = replay_late_public_rescue(graph, scene, failed_state, goal, deadline)
    assert late['state'].stop == goal
    late_path = tuple(row['ridden_connection'] for row in late['trace'] if row['ridden_connection'] is not None)
    return dict(world=archived['world'], root_hindsight=archived['hindsight'],
        v16_loss=archived['policies']['v16']['loss'], support_loss=original['cost'],
        release_clock=candidate['publication'], conditional_board=candidate['decisions'][0],
        original_path=describe(graph, scene, paths[str(first)]),
        original_executed=[row['ridden_connection'] for row in original['trace']
                           if row['ridden_connection'] is not None],
        transfer=dict(stop=graph.stops[transfer_state.stop].name, alight_clock=alight['time'],
            mct_seconds=mct, ready_clock=ready, target_cid=failed_cid, target_departure=event.departure,
            target_minus_ready=event.departure-ready, failure_clock=failure_clock, public_reason=failure_reason),
        same_first_hindsight=dict(loss=best_clock-root.time, continuation=describe(graph, scene, path),
                                  platform_witness_loss=first_state.time-root.time+witness['cost']),
        late_public_through=dict(loss=late['state'].time-root.time, continuation=describe(graph, scene, late_path)))


def main():
    archive = json.loads((HERE/'support_pair_results.json').read_text())
    record = next(row for row in archive['roots'] if tuple(row['case']) == CASE)
    graph, belief, root, goal, deadline, plan, pars = prepare(CASE)
    compiled, _ = compile_policy(graph, belief, pars, root, goal, deadline, plan, 'support')
    with TemporaryDirectory(prefix='bapr_v17_support_failure_') as temp:
        build_temporal_kernel(Path(temp)/'scan.so')
        clocks = sample_clocks(belief, pars, rng(record['seed']), len(record['rows']),
                               templates=template_trip_map(belief, pars))
        rows = [investigate(graph, scene_from_clocks(graph, tuple(x[i] for x in clocks)),
            root, goal, deadline, plan, compiled, record['rows'][i], record['designs']['support']['paths'])
            for i in FAILURES]
    output = dict(case=CASE, seed=record['seed'], root_clock=root.time, deadline=deadline,
        scope='Two outcome-selected diagnostics, not a new validation sample. Same-first continuation '
              'is chosen with full hindsight and checked against physical platform execution. The '
              'late-through diagnostic starts when target failure is publicly known and chooses only '
              'currently visible, legally boardable scheduled-through trips, with no hindsight ranking.',
        rows=rows)
    (HERE/'support_failure_details.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
