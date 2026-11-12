"""Replay the three archived timeouts with the same fixed 9488 first ride."""

import json
from pathlib import Path

from experiments.reconstruction_v10.platform_sensitivity import adapter
from experiments.reconstruction_v13.platform_run import build_model
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from experiments.reconstruction_v19.run_policy_pair import PRIMARY, RELEASE_OFFSET, V18, compile_arm
from src.background_policy import IssuedPolicy, execute_background
from src.interruptible_recourse import InterruptibleRecourse, RecourseCursor
from src.local_transfer_rescue import compile_local_transfer_rescue
from src.platform_transit import WAIT
from src.tail_rescue import TailRescuePlan
from src.temporal_scan import sample_clocks, template_trip_map


HERE = Path(__file__).resolve().parent
CASE, FIRST = ('2023-07-27', 0, 1), 9488
PANELS = (('reconstruction_v17/support_pair_results.json', 'support', (3, 99)),
          ('reconstruction_v18/paired_results.json', 'support_old_wait', (178,)))
RANKING_ARMS = ('support_old_wait', V18, PRIMARY)


def replay(model, controller, publication, board_clock, goal):
    bound = controller.bind('forced_first')

    def choose(state):
        if FIRST in state.used_connections:
            return bound.action(state)
        if FIRST in state.visible_connections:
            assert state.time == board_clock
            bound.cursor = RecourseCursor(controller.plan, state.time)
            bound.previous = state
            return ('board', FIRST)
        return WAIT

    world = model.scenarios[0]
    result = execute_background(model, world, IssuedPolicy(lambda _: WAIT), IssuedPolicy(choose),
                                publication)
    rides = [dict(cid=step['ridden_connection'], departure=step['time'], arrival=step['next_time'])
             for step in result['trace'] if step['ridden_connection'] is not None]
    assert rides and rides[0]['cid'] == FIRST
    fallback = next((event for event in bound.trace if event['branch'] == 'local_transfer_rescue'), None)
    return dict(loss=result['cost'], arrived=result['state'].stop == goal, rides=rides,
                fallback=fallback)


def rank_and_replay(model, graph, plan, root, goal, clock, compiled):
    incumbent, _, _ = adapter(graph, plan.primary, root, 'incumbent')
    publication = root.time+RELEASE_OFFSET
    out = {}
    for arm, setup in compiled.items():
        controller = setup['controller']
        bound = controller.bind(arm)
        at_departure = {}

        def choose(state):
            action = bound.action(state)
            if state.time == clock and state.stop == root.stop and state.used_connections == root.used_connections:
                score = controller.clock_scores.at_visible(FIRST, state, controller.mode)
                wait = controller.waiting_score(state, bound.original.cursor)
                assert score is not None and wait is not None
                at_departure.update(board_9488=score.expected_arrival_or_deadline-root.time,
                                    wait=wait-root.time, margin=controller.margin, action=action)
            return action

        assert setup['construction_seconds'] < RELEASE_OFFSET
        policy = IssuedPolicy(choose, bound.policy.applicable, arm)
        result = execute_background(model, model.scenarios[0], incumbent, policy, publication)
        assert at_departure and result['published'] and not result['rejected']
        rides = [step['ridden_connection'] for step in result['trace'] if step['ridden_connection'] is not None]
        out[arm] = dict(at_departure=at_departure, loss=result['cost'],
                        arrived=result['state'].stop == goal, rides=rides)
    return dict(clock=clock, publication=publication, arms=out)


def main():
    graph, belief, root, goal, deadline, original, parameters = prepare(CASE)
    compiled = {arm: compile_arm(arm, graph, belief, parameters, root, goal, deadline, original)
                for arm in RANKING_ARMS}
    rows = []
    for filename, method, indices in PANELS:
        archive = json.loads((HERE.parent/filename).read_text())
        panel = next(row for row in archive['roots'] if tuple(row['case']) == CASE)
        paths = (panel['designs'][method]['paths'] if 'designs' in panel else panel['menus'][method])
        plan = TailRescuePlan(tuple(paths[str(FIRST)]))
        old = InterruptibleRecourse(graph, plan, root, deadline, mode='public_failure')
        new = compile_local_transfer_rescue(graph, plan, root, goal, deadline)
        clocks = sample_clocks(belief, parameters, rng(panel['seed']), len(panel['rows']),
                               templates=template_trip_map(belief, parameters))
        for index in indices:
            source = panel['rows'][index]['policies'][method]
            assert not source['reached'] and source['first'] == FIRST
            clock, published = source['decisions'][0]['time'], source['publication']
            scene = scene_from_clocks(graph, tuple(x[index] for x in clocks))
            model = build_model(graph, scene, root, goal, deadline)
            assert scene.events[FIRST].departure == clock
            base = replay(model, old, published, clock, goal)
            rescue = replay(model, new, published, clock, goal)
            assert not base['arrived'] and base['loss'] == source['loss']
            assert rescue['rides'][:len(base['rides'])] == base['rides']
            ranked = rank_and_replay(model, graph, original, root, goal, clock, compiled)
            rows.append(dict(source=filename, seed=panel['seed'], world=index, publication=published,
                             fixed_path=plan.primary, without_recovery=base, with_recovery=rescue,
                             policy_ranking=ranked))
    output = dict(case=CASE, scope='Outcome-selected regression on archived timeout worlds. The fixed '
                  'action replay changes only execution after a public miss. The ranking comparison '
                  'replays each complete policy at the V19 common publication.', rows=rows)
    (HERE/'miss_regression.json').write_text(json.dumps(output, indent=2)+'\n')
    for row in rows:
        a, b = row['without_recovery'], row['with_recovery']
        print(row['source'], row['world'], a['loss'], b['loss'], b['fallback'], flush=True)


if __name__ == '__main__':
    main()
