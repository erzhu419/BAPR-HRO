"""Fresh conditional replay of the decision to take the early 9488 at readiness."""

import json
from pathlib import Path
from statistics import mean, stdev

from experiments.reconstruction_v13.platform_run import build_model, replay_arm
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from experiments.reconstruction_v16.run_policy_pair import receipt
from experiments.reconstruction_v17.run_support_pair import compile_policy
from src.public_trip_conditionals import condition_on_departure
from src.temporal_scan import draw_chain_batch, sample_clocks, template_trip_map


CASE = ('2023-07-27', 0, 1)
CID, CLOCK, WORLDS = 9488, 29417, 256
SEED = [20261117, 20230727, 0, 1, 3]
ARMS = ('v16', 'support')


def main():
    here = Path(__file__).resolve().parent
    prior = json.loads((here/'support_pair_results.json').read_text())
    original = next(root for root in prior['roots'] if tuple(root['case']) == CASE)
    traces = [r['policies']['support']['decisions'][0] for r in original['rows']
              if not r['policies']['support']['reached']]
    reference = traces[0]['reference']
    assert all(row['time'] == CLOCK and row['action'] == ['board', CID]
               and row['reference'] == reference for row in traces)
    graph, belief, root, goal, deadline, plan, pars = prepare(CASE)
    policies = {arm: compile_policy(graph, belief, pars, root, goal, deadline, plan, arm)[0]
                for arm in ARMS}
    condition = condition_on_departure(belief, pars, CID, CLOCK)
    assert condition is not None
    trip = graph.connections[CID].trip_id
    # Retain all other trips' independent draws, and sample 9488's chain from
    # the exact masked law. The generic event update can add grid anchors.
    random = rng(SEED)
    clocks = sample_clocks(belief, pars, random, WORLDS, templates=template_trip_map(belief, pars))
    slots = belief._trips[trip]
    law = condition.law
    draws = draw_chain_batch(law, random.random((WORLDS, len(slots)))[:, ::-1])
    departures, arrivals, canceled = clocks
    for index in range(0, len(slots), 2):
        cid = slots[index].cid
        departures[:, cid] = law.times[index][draws[:, index]]
        arrivals[:, cid] = law.times[index+1][draws[:, index+1]]
        canceled[:, cid] = law.canceled[index][draws[:, index]]
    assert (departures[:, CID] == CLOCK).all() and not canceled[:, CID].any()
    rows = []
    for i in range(WORLDS):
        scene = scene_from_clocks(graph, tuple(x[i] for x in clocks))
        model = build_model(graph, scene, root, goal, deadline)
        results = {arm: receipt(replay_arm(graph, plan, root, goal, deadline, model,
                                          model.scenarios[0], arm, policies[arm])) for arm in ARMS}
        early = results['support']
        match = any(t['time'] == CLOCK and tuple(t['action']) == ('board', CID)
                    and abs(t['reference']-reference) < 1e-9 for t in early['decisions'])
        if match:
            assert early['first'] == CID
            rows.append(dict(world=i, policies=results))
    delta = [r['policies']['support']['loss']-r['policies']['v16']['loss'] for r in rows]
    out = dict(case=CASE, seed=SEED, drawn=WORLDS, matched=len(rows), receipt=dict(cid=CID, clock=CLOCK),
               scope='Fresh worlds conditioned on this public departure only, then selected by the '
                     'same public boarding decision and A/B proxy value, without selecting on outcomes. '
                     'Replay each whole policy from the root. This is a fitted-model diagnostic for '
                     'the matched receipt class, not an evaluation across all departure histories.',
               estimated_wait_loss=reference-root.time, estimated_board_loss=traces[0]['score']-root.time,
               mean_wait_gain=mean(delta), paired_standard_error=stdev(delta)/len(delta)**.5,
               wait_improved=sum(d>0 for d in delta), wait_harmed=sum(d<0 for d in delta),
               arms={arm: dict(mean_loss=mean(r['policies'][arm]['loss'] for r in rows),
                               arrivals=sum(r['policies'][arm]['reached'] for r in rows)) for arm in ARMS},
               rows=rows)
    (here/'wait_comparison.json').write_text(json.dumps(out, indent=2)+'\n')
    print({k: v for k, v in out.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    main()
