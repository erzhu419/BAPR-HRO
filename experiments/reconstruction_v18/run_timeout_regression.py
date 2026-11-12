"""Reconstruct the two V17 support-expansion timeouts under V18."""

import json
from pathlib import Path

from experiments.reconstruction_v13.platform_run import build_model, replay_arm
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from experiments.reconstruction_v16.run_policy_pair import receipt
from experiments.reconstruction_v17.run_support_failure_details import CASE, FAILURES
from experiments.reconstruction_v18.run_policy_pair import PRIMARY, RELEASE_OFFSET, compile_comparator
from src.platform_transit import OBSERVE, WAIT
from src.temporal_scan import sample_clocks, template_trip_map


HERE = Path(__file__).resolve().parent
ARMS = ('support_old_wait', PRIMARY)


def compare_at_departure(model, world, compiled, root, clock, cid):
    old = compiled[ARMS[0]]['controller']
    binding = old.original.bind('ab_prefix')
    state = model.initial_state
    while state.time < clock:
        action = binding.action(state)
        assert action in (OBSERVE, WAIT)
        state = model.transition(world, state, action).state
    assert state.time == clock and binding.action(state) == WAIT
    out = {}
    for arm, item in compiled.items():
        controller = item['controller']
        board = controller.clock_scores.at_visible(cid, state, controller.mode)
        wait = controller.waiting_score(state, binding.cursor)
        assert board is not None and wait is not None
        out[arm] = dict(board_estimate=board.expected_arrival_or_deadline-root.time,
                        wait_estimate=wait-root.time,
                        clears_margin=board.expected_arrival_or_deadline+controller.margin < wait)
    return out


def main():
    archive = json.loads((HERE.parent/'reconstruction_v17'/'support_pair_results.json').read_text())
    source = next(row for row in archive['roots'] if tuple(row['case']) == CASE)
    graph, belief, root, goal, deadline, plan, parameters = prepare(CASE)
    compiled = {arm: compile_comparator(arm, graph, belief, parameters, root, goal, deadline,
                                        plan, censor_unknown_extras=False) for arm in ARMS}
    clocks = sample_clocks(belief, parameters, rng(source['seed']), len(source['rows']),
                           templates=template_trip_map(belief, parameters))
    publish = root.time+RELEASE_OFFSET
    rows = []
    for index in FAILURES:
        archived = source['rows'][index]
        original = archived['policies']['support']
        assert archived['world'] == index and not original['reached']
        choice = original['decisions'][0]
        clock, cid = choice['time'], original['first']
        scene = scene_from_clocks(graph, tuple(a[index] for a in clocks))
        model = build_model(graph, scene, root, goal, deadline)
        world = model.scenarios[0]
        estimates = compare_at_departure(model, world, compiled, root, clock, cid)
        results = {}
        for arm in ARMS:
            row = replay_arm(graph, plan, root, goal, deadline, model, world, arm, compiled[arm],
                             publish_not_before=publish)
            assert row['publish_at'] == publish and row['published'] and not row['rejected']
            compact = receipt(row)
            results[arm] = {k: compact[k] for k in ('loss', 'reached', 'first', 'decisions')}
        assert results[ARMS[0]]['first'] == cid and results[ARMS[0]]['loss'] == original['loss']
        assert abs(estimates[ARMS[0]]['wait_estimate']-(choice['reference']-root.time)) < 1e-9
        rows.append(dict(world=index, clock=clock, visible_departure=cid, estimates=estimates,
                         hindsight=archived['hindsight'], policies=results))
    out = dict(case=CASE, seed=source['seed'], publication=publish, margin=compiled[PRIMARY]['controller'].margin,
               scope='Regression on the two archived V17 timeout worlds, reconstructed from the '
                     'original posterior tape and replayed with the V18 common release.', rows=rows)
    (HERE/'timeout_regression.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
