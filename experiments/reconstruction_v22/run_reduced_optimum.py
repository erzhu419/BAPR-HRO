"""Solve the frozen direct-menu offer process with every boarding optional."""

from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from math import fsum, prod
from time import perf_counter
from types import SimpleNamespace

from experiments.reconstruction_v13.factored_run import CASES
from experiments.reconstruction_v15.run_attribution_banks import prepare
from experiments.reconstruction_v18.run_policy_pair import root_menu, support_additions
from experiments.reconstruction_v22.run_offer_audit import GRIDS, HERE, FirstDeparture, OfferAudit
from src.public_wait_recourse import through_itineraries
from src.through_wait_value import ThroughWaitValue


def first_offer_value(audit, active, survive):
    savings, board_mass = [], []
    for cid in active:
        law = audit.factors[cid]
        for clock, mass in law.mass.items():
            if clock < audit.ready:
                continue
            weight = mass / survive[cid]
            for other in active:
                if other == cid:
                    continue
                competitor = audit.factors[other]
                remaining = competitor.after(clock)
                if audit.rank[cid] < audit.rank[other]:
                    remaining += competitor.mass.get(clock, 0.)
                weight *= remaining / survive[other]
            board_mass.append(weight)
            score = audit.scores.rows[cid][clock].expected_arrival_or_deadline
            savings.append(weight * (audit.deadline - score))
    no_offer = prod(audit.factors[cid].none / survive[cid] for cid in active)
    assert abs(fsum(board_mass) + no_offer - 1.) < 1e-10
    return audit.deadline - fsum(savings), no_offer


def evaluate(audit):
    early = {cid: fsum(p for _, p in law.early) for cid, law in audit.factors.items()}
    survive = {cid: fsum([law.none, *(p for t, p in law.mass.items() if t >= audit.ready)])
               for cid, law in audit.factors.items()}
    future = {cid: tuple((t, p / survive[cid], audit.scores.rows[cid][t].expected_arrival_or_deadline)
                         for t, p in law.mass.items() if t >= audit.ready)
              for cid, law in audit.factors.items()}
    planner = ThroughWaitValue(audit.scores, (), margin=0)
    rows = []
    for mask in product((False, True), repeat=len(audit.cids)):
        active = [cid for cid, alive in zip(audit.cids, mask, strict=True) if alive]
        mass = prod((survive if alive else early)[cid]
                    for cid, alive in zip(audit.cids, mask, strict=True))
        if not mass:
            continue
        optimum = planner._solve((), (), tuple(future[cid] for cid in active))
        first, no_offer = first_offer_value(audit, active, survive)
        assert optimum <= first + 1e-8
        rows.append(dict(active=active, mass=mass, no_offer=no_offer,
                         optimum=optimum-audit.root.time, first_offer=first-audit.root.time,
                         improvement=max(0., first-optimum)))
    assert abs(fsum(r['mass'] for r in rows) - 1.) < 1e-10
    return dict(first_offer_loss=fsum(r['mass']*r['first_offer'] for r in rows),
                optimal_loss=fsum(r['mass']*r['optimum'] for r in rows),
                headroom=fsum(r['mass']*r['improvement'] for r in rows),
                no_offer_mass=fsum(r['mass']*r['no_offer'] for r in rows),
                masks=len(rows), max_mask_gain=max(rows, key=lambda r: r['improvement']),
                largest_weighted_gain=max(rows, key=lambda r: r['mass']*r['improvement']))


def run_root(case):
    graph, belief, root, goal, deadline, plan, parameters = prepare(case)
    original, extras = root_menu(graph, root, goal, deadline, plan)
    extras.update(support_additions(graph, belief, parameters, root, goal, deadline, original, extras))
    trips = {belief._trip_by_cid[cid] for cid in original.keys() | through_itineraries(graph, extras).keys()}
    grids = []
    for step in GRIDS:
        started = perf_counter()
        law = belief if step is None else belief.with_grid_step(step, trips=trips)
        audit = OfferAudit(law, parameters, plan, root, goal, deadline, original, extras)
        result = evaluate(audit)
        grids.append(dict(grid_step=step, seconds=perf_counter()-started, **result))
        print(case, step, result['first_offer_loss'], result['optimal_loss'], result['headroom'], flush=True)
    return dict(case=case, grids=grids)


def main():
    scores = SimpleNamespace(deadline=150, rows={
        0: {0: SimpleNamespace(expected_arrival_or_deadline=100)},
        1: {t: SimpleNamespace(expected_arrival_or_deadline=t + 60) for t in (10, 50)},
    })
    example = evaluate(SimpleNamespace(cids=(0, 1), root=SimpleNamespace(time=0), ready=0, deadline=150,
        rank={0: (100, 0), 1: (60, 1)}, scores=scores,
        factors={0: FirstDeparture((0,), (1.,), 0., 0, 0, 150),
                 1: FirstDeparture((10, 50), (.5, .5), 0., 0, 0, 150)}))
    assert example['first_offer_loss'] == 100. and example['optimal_loss'] == 90.
    with ProcessPoolExecutor(max_workers=len(CASES)) as pool:
        roots = list(pool.map(run_root, CASES))
    result = dict(status='complete', scope='Independent first-departure factors on the frozen direct menu. '
                  'All legal menu offers after physical readiness, including A/B, can be declined. '
                  'The reward of boarding is the existing departure-conditioned, clipped expected fixed '
                  'tail arrival. Other public events and alternative tails are outside the reduced '
                  'information and action space. Floating point enumeration, not a rigorous enclosure.',
                  sanity_example=example, roots=roots)
    (HERE/'reduced_optimum.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
