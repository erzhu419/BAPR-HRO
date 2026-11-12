"""Replay the V22 offer support with the original A/B paths eligible for comparison."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from math import fsum, isclose
from pathlib import Path
from time import perf_counter

from experiments.reconstruction_v22.run_offer_audit import (
    CASES, GRIDS, OfferAudit, prepare, root_menu, support_additions, through_itineraries,
)


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / 'reconstruction_v22' / 'offer_audit.json'
SHORT = ('2023-07-26', 0, 2)


def metrics(result):
    exact = result['disagreement']['exact']
    return dict(mean_loss=result['mean_loss'], first_histories=result['first_histories'],
                disagreement_mass=exact['mass'], wait_mass=exact['wait_mass'])


def changed_first_offers(archived, corrected):
    outcomes = defaultdict(list)
    for history, mass in corrected.first_offers():
        tick = history[-1][0]
        state = corrected.state(tick, history)
        if not corrected.ab.intersection(state.visible_connections):
            continue
        old, new = archived.exact_before_first(history), corrected.exact_before_first(history)
        before, after = old.action(state), new.action(state)
        if before == after:
            continue
        assert after[0] == 'board' and after[1] in corrected.ab
        expired = any(kind == 'attempt_expired' and cid == after[1]
                      for row in old.original.trace for kind, cid, _ in row.get('events', ()))
        active = tuple(cid for cid in corrected.cids if cid not in {c for _, c in history})
        gain = (archived.value('exact', before, tick, active, old.original.cursor)
                - corrected.value('exact', after, tick, active, new.original.cursor))
        outcomes[expired, before[0]].append((mass, mass * gain))
    return [dict(prior_attempt_expired=expired, previous_action=previous_action, histories=len(rows),
                 mass=fsum(row[0] for row in rows), loss_reduction=fsum(row[1] for row in rows))
            for (expired, previous_action), rows in sorted(outcomes.items())]


def run_grid(job):
    case, step, archived = job
    start = perf_counter()
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    original, extras = root_menu(graph, root, goal, deadline, plan)
    extras.update(support_additions(graph, belief, parameters, root, goal, deadline, original, extras))
    trips = {belief._trip_by_cid[cid] for cid in original.keys() | through_itineraries(graph, extras).keys()}
    law = belief if step is None else belief.with_grid_step(step, trips=trips)
    args = law, parameters, plan, root, goal, deadline, original, extras
    corrected = OfferAudit(*args, original_openers=True, archived_nominal=True)
    result = metrics(corrected.summarize())
    for policy in ('first_offer', 'nominal'):
        assert isclose(result['mean_loss'][policy], archived['mean_loss'][policy], abs_tol=1e-9, rel_tol=0.)
    row = dict(case=case, grid_step=step, archived=archived, corrected=result,
               exact_loss_reduction=archived['mean_loss']['exact'] - result['mean_loss']['exact'])
    if case == SHORT and step in (None, 10):
        historical = OfferAudit(*args, archived_nominal=True)
        replay = metrics(historical.summarize())
        assert replay == archived
        row.update(archived_replay_matches=True, changed_first_offers=changed_first_offers(historical, corrected))
    corrected.later_offers.cache_clear()
    corrected.continue_waiting.cache_clear()
    row['seconds'] = perf_counter() - start
    print(case, step, row['exact_loss_reduction'], flush=True)
    return row


if __name__ == '__main__':
    archive = {(tuple(root['case']), row['grid_step']): metrics(row)
               for root in json.loads(ARCHIVE.read_text())['roots'] for row in root['grids']}
    jobs = [(case, step, archive[case, step]) for case in CASES for step in GRIDS]
    jobs.sort(key=lambda job: GRIDS.index(job[1]), reverse=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run_grid, jobs))
    rows.sort(key=lambda row: (row['case'], GRIDS.index(row['grid_step'])))
    (HERE / 'original_offer_audit.json').write_text(json.dumps(dict(roots=rows), indent=2) + '\n')
