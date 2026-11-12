"""Condition only on reachable public prefixes where the stage-one arms split."""

import json
from math import prod
from pathlib import Path

from experiments.reconstruction_v13.platform_run import build_model
from experiments.reconstruction_v15.run_attribution_banks import prepare, rng, scene_from_clocks
from src.committed_transit import CommittedState
from src.first_opportunity_rescue import route_plan
from src.platform_public_particles import PlatformPublicParticles
from src.platform_transit import OBSERVE, WAIT
from src.risk_bounds import CommitmentBounds
from src.temporal_scan import sample_clocks, template_trip_map


HERE = Path(__file__).resolve().parent


def compare(root, disagreement, context, tape):
    graph, belief, initial, goal, deadline, issued, parameters = context
    paths = {int(cid): tuple(path) for cid, path in root['paths'].items()}
    tick = disagreement['first_offer_clock']
    offers = set(disagreement['first_offer_cids'])
    scene = scene_from_clocks(graph, tuple(a[disagreement['world']] for a in tape))
    model = build_model(graph, scene, initial, goal, deadline)
    world = model.scenarios[0]
    state = model.transition(world, initial, OBSERVE).state
    while state.time < tick:
        assert not (set(state.visible_connections) & paths.keys()) or state.time < root['release']
        state = model.transition(world, state, WAIT).state
    assert state.time == tick and set(state.visible_connections) & paths.keys() == offers

    compiler = PlatformPublicParticles(belief, parameters)
    history = compiler.bind()
    history.observe(state)
    bounds = CommitmentBounds(belief, parameters, graph, goal, deadline)
    committed = CommittedState(state.stop, tick, None, state.last_arrival, state.used_connections)
    expected = {}
    for cid in sorted(offers):
        trip = belief._positions[cid, 'departure'][0]
        base, _, _ = compiler.trip(trip)
        exact, absent = history._evidence(trip)
        law, start = compiler._conditioned_law(trip, 0, len(base.times)-1, exact, absent)
        assert start == 0
        bounds._laws[trip] = law
        plan = route_plan(cid, paths[cid], issued)
        mark = bounds._itinerary(committed, plan.primary, first_action_now=True,
                                lookahead=max(1500, graph.connections[cid].dep_time-tick),
                                departures=dict(plan.departures))
        expected[cid] = mark.upper + tick - initial.time

    # The full prefix conditions the cost above. This coarser event supplies a
    # reachable frequency ceiling without treating an abstract mask as history.
    factors = []
    assert len({belief._positions[cid, 'departure'][0] for cid in paths}) == len(paths)
    for cid in paths:
        times, weights, cancel = belief.event_distribution(cid, 'departure', parameters)
        eligible = (times >= root['ready']) & (times <= tick)
        factors.append(float(weights[times==tick].sum()*(1-cancel)) if cid in offers
                       else 1.-float(weights[eligible].sum()*(1-cancel)))
    return dict(disagreement, public_receipts=len(state.public_log),
                exact_conditional_loss=expected, first_offer_set_and_clock_probability=prod(factors),
                exact_gain_vs_minimal=expected[disagreement['minimal_first']]-expected[disagreement['arm_first']])


def main():
    data = json.loads((HERE/'baseline_results.json').read_text())
    assert data['status'] == 'complete'
    disagreements = []
    for root in data['roots']:
        if root['disagreements']:
            context = prepare(tuple(root['case']))
            belief, parameters = context[1], context[-1]
            tape = sample_clocks(belief, parameters, rng(root['eval_seed']), len(root['rows']),
                                 templates=template_trip_map(belief, parameters))
            disagreements.extend(dict(case=root['case'], **compare(root, row, context, tape))
                                 for row in root['disagreements'])
    result = dict(status='complete', scope='Action-level conditional integration in the fixed P60 model. '
                  'The full observed platform prefix is used for cost. Reported mass is for the coarser '
                  'clock and first offered menu set and bounds the exact prefix frequency.',
                  disagreements=disagreements)
    (HERE/'baseline_disagreements.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result)


if __name__ == '__main__':
    main()
