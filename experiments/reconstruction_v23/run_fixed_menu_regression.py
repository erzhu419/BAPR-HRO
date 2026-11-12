"""Replay the V22 fitted tapes through the corrected platform controllers."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean

from experiments.reconstruction_v13.platform_run import build_model, replay_arm
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from experiments.reconstruction_v20.run_policy_pair import compile_direct
from experiments.reconstruction_v21.run_policy_pair import compile_rollout
from experiments.reconstruction_v22.run_baselines import MINIMAL, NOMINAL, EXACT, compile_minimal
from src.first_opportunity_rescue import route_plan
from src.temporal_scan import sample_clocks, template_trip_map


HERE = Path(__file__).resolve().parent


def replay_root(frozen):
    case = tuple(frozen['case'])
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    args = graph, belief, parameters, root, goal, deadline, plan
    setups = {MINIMAL: compile_minimal(*args), EXACT: compile_direct(*args),
              NOMINAL: compile_rollout(NOMINAL, *args, frozen['planning_seed'])}
    search = setups[NOMINAL]['controller'].search
    for cid in plan.primary[:1] + plan.rescue[:1]:
        suffix = search.continuations[cid]
        suffix.plan = route_plan(cid, suffix.plan.primary, plan)
    exact_paths = {cid: path.primary for cid, path in setups[EXACT]['controller'].scores.paths.items()}
    assert setups[MINIMAL]['paths'] == exact_paths == setups[NOMINAL]['paths']
    dep, arr, canceled = sample_clocks(belief, parameters, rng(frozen['eval_seed']), len(frozen['rows']),
                                        templates=template_trip_map(belief, parameters))
    records, changes = [], []
    for previous, clocks in zip(frozen['rows'], zip(dep, arr, canceled, strict=True), strict=True):
        scene = scene_from_clocks(graph, clocks)
        model = build_model(graph, scene, root, goal, deadline)
        row = {}
        for arm, setup in setups.items():
            result = replay_arm(graph, plan, root, goal, deadline, model, model.scenarios[0], arm, setup,
                                publish_not_before=frozen['release'])
            assert result['published'] and not result['rejected'] and result['publish_at'] == frozen['release']
            assert result['loss_lower'] == result['loss_upper']
            rides = result['rides']
            row[arm] = dict(first=rides[0]['ridden_connection'] if rides else None,
                            first_clock=rides[0]['time'] if rides else None,
                            loss=result['loss_upper'], arrived=result['status'] == 'arrived')
            old = {key: previous['policies'][arm][key] for key in row[arm]}
            if old != row[arm]:
                changes.append(dict(world=previous['world'], arm=arm, before=old, after=row[arm]))
        records.append(row)
    summary = {arm: dict(loss=mean(row[arm]['loss'] for row in records),
                         arrivals=sum(row[arm]['arrived'] for row in records),
                         disagreements_vs_first=sum(row[arm] != row[MINIMAL] for row in records)) for arm in setups}
    return dict(case=case, worlds=len(records), release=frozen['release'], summary=summary, archive_changes=changes)


if __name__ == '__main__':
    roots = json.loads((HERE.parent / 'reconstruction_v22' / 'baseline_results.json').read_text())['roots']
    with ProcessPoolExecutor(max_workers=len(roots)) as pool:
        results = list(pool.map(replay_root, roots))
    (HERE / 'fixed_menu_regression.json').write_text(json.dumps(results, indent=2) + '\n')
